"""Map ShopSimulator goals to private versioned TaskFacts records."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from shopping_grpo.evaluation.rubric import build_task_facts


def task_facts_from_environment(
    *,
    task_ids: Iterable[int],
    goals: list[Mapping],
    product_item_dict: Mapping,
) -> list[dict]:
    """Build private TaskFacts in requested task order without running an env."""

    requested = [int(task_id) for task_id in task_ids]
    if len(set(requested)) != len(requested):
        raise ValueError("task_ids contains duplicates")
    rows = []
    for task_id in requested:
        if task_id < 0 or task_id >= len(goals):
            raise IndexError(
                f"task_id {task_id} is outside goal range [0, {len(goals)})"
            )
        goal = goals[task_id]
        if not isinstance(goal, Mapping):
            raise ValueError(f"goal {task_id} must be an object")
        asin = goal.get("asin")
        product = product_item_dict.get(asin)
        if not isinstance(product, Mapping):
            raise ValueError(
                f"goal {task_id} target ASIN {asin!r} is missing"
            )
        query = str(goal.get("instruction_text") or "").strip()
        if not query:
            raise ValueError(f"goal {task_id} has no instruction_text")
        instruction_record = {
            "instruction": query,
            "attributes": goal.get("attributes") or [],
            "instruction_options": goal.get("goal_options") or [],
        }
        rows.append(
            build_task_facts(
                task_id=task_id,
                query=query,
                target_product=product,
                instruction_record=instruction_record,
                reward_goal=goal,
            )
        )
    return rows
