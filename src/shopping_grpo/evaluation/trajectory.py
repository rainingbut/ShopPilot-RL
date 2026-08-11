"""把采集器原始轨迹整理成可审计、只含 Actor 可见信息的事件流。

评测前统一经过这里：工具调用会被解析成稳定字段，动作守卫拒绝会单独成为事件，
而环境的 hidden goal、raw observation 默认不会进入 Judge 输入。
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from copy import deepcopy
import json
import re

from shopping_grpo.evaluation.contracts import CONTRACT_VERSION


NORMALIZED_TRAJECTORY_VERSION = "shopping-normalized-trajectory-v1"


def _text(value: object) -> str:
    return value if isinstance(value, str) else ""


def _tool_call_payload(tool_call: object) -> tuple[str, dict, str | None]:
    """解析工具名和参数；历史坏数据转成 parse error 而不是中断整条评测。"""

    if not isinstance(tool_call, Mapping):
        return "", {}, "tool_call_not_object"
    function = tool_call.get("function")
    if not isinstance(function, Mapping):
        return "", {}, "tool_call_function_not_object"
    name = _text(function.get("name")).strip()
    raw_arguments = function.get("arguments", {})
    if isinstance(raw_arguments, Mapping):
        return name, deepcopy(dict(raw_arguments)), None
    if not isinstance(raw_arguments, str):
        return name, {}, "tool_call_arguments_not_object_or_json"
    try:
        arguments = json.loads(raw_arguments or "{}")
    except json.JSONDecodeError:
        return name, {}, "tool_call_arguments_invalid_json"
    if not isinstance(arguments, dict):
        return name, {}, "tool_call_arguments_not_object"
    return name, arguments, None


def _assistant_by_tool_call(messages: object) -> dict[str, str]:
    result = {}
    if not isinstance(messages, list):
        return result
    for message in messages:
        if not isinstance(message, Mapping) or message.get("role") != "assistant":
            continue
        content = _text(message.get("content"))
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue
        for tool_call in tool_calls:
            if not isinstance(tool_call, Mapping):
                continue
            tool_call_id = tool_call.get("id")
            if tool_call_id is not None:
                result[str(tool_call_id)] = content
    return result


def _query_from_trajectory(trajectory: Mapping) -> str:
    initial = trajectory.get("initial_result")
    if isinstance(initial, Mapping):
        instruction = initial.get("instruction")
        if isinstance(instruction, str) and instruction.strip():
            return instruction.strip()
    messages = trajectory.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, Mapping) or message.get("role") != "user":
                continue
            content = _text(message.get("content")).strip()
            return re.sub(r"^\s*Instruction:\s*", "", content, flags=re.I)
    return ""


def _tool_call_id(tool_call: object) -> str | None:
    if isinstance(tool_call, Mapping) and tool_call.get("id") is not None:
        return str(tool_call["id"])
    return None


def _terminal_view(trajectory: Mapping) -> dict:
    terminal = trajectory.get("terminal_result")
    terminal = terminal if isinstance(terminal, Mapping) else {}
    reward_detail = terminal.get("reward_detail")
    reward_detail = (
        deepcopy(dict(reward_detail))
        if isinstance(reward_detail, Mapping)
        else {}
    )
    purchase = terminal.get("purchase")
    purchase = deepcopy(dict(purchase)) if isinstance(purchase, Mapping) else {}
    progress = terminal.get("progress")
    progress = deepcopy(dict(progress)) if isinstance(progress, Mapping) else {}
    return {
        "done": bool(terminal.get("done", False)),
        "over": bool(terminal.get("over", False)),
        "reward": float(
            terminal.get("reward", trajectory.get("final_reward", 0.0)) or 0.0
        ),
        "termination_reason": (
            reward_detail.get("termination_reason")
            or terminal.get("termination_reason")
            or ""
        ),
        "reward_detail": reward_detail,
        "purchase": purchase,
        "progress": progress,
    }


def normalize_trajectory(
    trajectory: object,
    *,
    include_audit_raw_observations: bool = False,
) -> dict:
    """Convert one raw collector record to a stable, Judge-safe event stream.

    The default output includes only observations visible to the Actor.  Hidden
    target goals, persona fields, and full raw observations are deliberately
    excluded.  Raw observations can be retained under an explicitly audit-only
    field for local debugging, but should never be rendered into Judge input.
    """

    if not isinstance(trajectory, Mapping):
        raise TypeError("trajectory must be an object")
    source = trajectory
    messages = source.get("messages")
    assistant_by_call = _assistant_by_tool_call(messages)
    steps = source.get("steps")
    steps = steps if isinstance(steps, list) else []
    blocked = source.get("blocked_tool_calls")
    blocked = blocked if isinstance(blocked, list) else []
    warnings = []

    # 先按 source step 位置保存被守卫拒绝的调用，稍后与真正执行的步骤交错输出，
    # 这样 event_id 能反映模型真实经历的动作顺序。
    blocked_by_position = defaultdict(list)
    trailing_blocked = []
    for blocked_index, item in enumerate(blocked):
        if not isinstance(item, Mapping):
            warnings.append(f"blocked_tool_calls[{blocked_index}] is not an object")
            continue
        position = item.get("step_index")
        if isinstance(position, int) and not isinstance(position, bool) and 0 <= position <= len(steps):
            blocked_by_position[position].append((blocked_index, item))
        else:
            warnings.append(
                f"blocked_tool_calls[{blocked_index}] has invalid step_index"
            )
            trailing_blocked.append((blocked_index, item))

    events = []
    action_attempt = 0

    def append_guard(blocked_index: int, item: Mapping) -> None:
        nonlocal action_attempt
        action_attempt += 1
        tool_call = item.get("tool_call")
        name, parameters, parse_error = _tool_call_payload(tool_call)
        call_id = _tool_call_id(tool_call)
        event = {
            "event_id": f"e{len(events) + 1:04d}",
            "event_type": "guard_rejection",
            "action_attempt_id": f"a{action_attempt:04d}",
            "executed_step_id": None,
            "source_step_index": item.get("step_index"),
            "source_blocked_index": blocked_index,
            "assistant_text": assistant_by_call.get(call_id or "", ""),
            "tool_call_id": call_id,
            "tool_name": name,
            "parameters": parameters,
            "tool_call_parse_error": parse_error,
            "guard_reason": _text(item.get("reason")) or "unknown",
            "guard_consecutive_count": int(item.get("consecutive_count", 0) or 0),
            "latest_observation_truncated": bool(
                item.get("latest_observation_truncated", False)
            ),
            "actor_visible_observation": "",
            "reward": 0.0,
            "done": False,
        }
        events.append(event)

    def append_step(source_index: int, step_value: object) -> None:
        nonlocal action_attempt
        action_attempt += 1
        if not isinstance(step_value, Mapping):
            warnings.append(f"steps[{source_index}] is not an object")
            step = {}
        else:
            step = step_value
        tool_call = step.get("tool_call")
        parsed_name, parsed_parameters, parse_error = _tool_call_payload(tool_call)
        call_id = _tool_call_id(tool_call)
        tool_name = _text(step.get("tool_name")).strip() or parsed_name
        parameters = step.get("parameters")
        parameters = (
            deepcopy(dict(parameters))
            if isinstance(parameters, Mapping)
            else parsed_parameters
        )
        projection = step.get("projection")
        projection = (
            deepcopy(dict(projection))
            if isinstance(projection, Mapping)
            else {}
        )
        event = {
            "event_id": f"e{len(events) + 1:04d}",
            "event_type": "tool_step",
            "action_attempt_id": f"a{action_attempt:04d}",
            "executed_step_id": f"s{source_index:04d}",
            "source_step_index": source_index,
            "assistant_text": assistant_by_call.get(call_id or "", ""),
            "tool_call_id": call_id,
            "tool_name": tool_name,
            "parameters": parameters,
            "tool_call_parse_error": parse_error,
            "env_action": step.get("env_action"),
            "actor_visible_observation": _text(step.get("observation")),
            "observation_projection": projection,
            "raw_observation_available": isinstance(
                step.get("raw_observation"), str
            ),
            "reward": float(step.get("reward", 0.0) or 0.0),
            "done": bool(step.get("done", False)),
            "step_error": deepcopy(step.get("error")),
        }
        if include_audit_raw_observations:
            event["audit_only_raw_observation"] = _text(
                step.get("raw_observation")
            )
        events.append(event)

    # 每个动作尝试都有独立 ID：被拒绝的调用也计数，但没有 executed_step_id。
    for source_index in range(len(steps) + 1):
        for blocked_index, item in blocked_by_position.get(source_index, []):
            append_guard(blocked_index, item)
        if source_index < len(steps):
            append_step(source_index, steps[source_index])
    for blocked_index, item in trailing_blocked:
        append_guard(blocked_index, item)

    initial = source.get("initial_result")
    initial = initial if isinstance(initial, Mapping) else {}
    return {
        "schema_version": NORMALIZED_TRAJECTORY_VERSION,
        "evaluation_contract": CONTRACT_VERSION,
        "trajectory_id": str(source.get("trajectory_id") or ""),
        "task_id": int(source.get("task_id")),
        "attempt_index": int(source.get("attempt_index", 0) or 0),
        "created_at": source.get("created_at"),
        "actor_query": _query_from_trajectory(source),
        "status": _text(source.get("status")) or "unknown",
        "done": bool(source.get("done", False)),
        "final_reward": float(source.get("final_reward", 0.0) or 0.0),
        "infrastructure_invalid": bool(
            source.get("infrastructure_invalid", False)
        ),
        "environment": {
            "environment_version": initial.get("environment_version"),
            "reward_version": initial.get("reward_version"),
        },
        "events": events,
        "terminal": _terminal_view(source),
        "context": {
            "turn_tokens": deepcopy(source.get("context_turn_tokens") or []),
            "compactions": deepcopy(source.get("context_compactions") or []),
            "tool_call_truncations": deepcopy(
                source.get("tool_call_truncations") or []
            ),
        },
        "errors": {
            "trajectory_error": deepcopy(source.get("error")),
            "release_error": deepcopy(source.get("release_error")),
        },
        "normalization": {
            "warnings": warnings,
            "raw_observations_included": bool(
                include_audit_raw_observations
            ),
        },
    }
