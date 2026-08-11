"""面向用户的轻量 CLI：运行 CPU 契约检查，或离线汇总已保存的轨迹。

这里不启动模型、不启动 ShopSimulator；需要 GPU 的训练和在线评测仍由仓库根目录
的 shell 入口负责。
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from shopping_grpo.evaluation.artifacts import iter_jsonl, write_jsonl_atomic
from shopping_grpo.evaluation.metrics import compute_deterministic_metrics
from shopping_grpo.evaluation.trajectory import normalize_trajectory
from shopping_grpo.smoke import run_cpu_smoke


def _offline_evaluate(args: argparse.Namespace) -> None:
    """规范化 JSONL 轨迹，计算确定性指标，并打印汇总。"""
    rows = []
    reward_types = Counter()
    strict_successes = 0
    executed_steps = 0
    for raw in iter_jsonl(args.trajectories):
        normalized = normalize_trajectory(raw)
        metrics = compute_deterministic_metrics(normalized)
        outcome = metrics["reward_and_outcome"]
        reward_types[str(outcome["reward_type"])] += 1
        strict_successes += int(outcome["strict_gold_success"])
        executed_steps += int(
            metrics["actions_and_efficiency"]["executed_tool_steps"]
        )
        rows.append(
            {
                "task_id": normalized["task_id"],
                "trajectory_id": normalized["trajectory_id"],
                "normalized_trajectory": normalized,
                "deterministic_metrics": metrics,
            }
        )
    count = len(rows)
    summary = {
        "schema_version": "shopping-offline-example-summary-v1",
        "trajectory_count": count,
        "strict_gold_success_count": strict_successes,
        "strict_gold_success_rate": (
            strict_successes / count if count else 0.0
        ),
        "reward_type_counts": dict(sorted(reward_types.items())),
        "executed_tool_steps": executed_steps,
        "mean_executed_tool_steps": executed_steps / count if count else 0.0,
    }
    if args.output:
        write_jsonl_atomic(args.output, rows, force=args.force)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def _smoke(args: argparse.Namespace) -> None:
    """运行不依赖模型和环境服务的纯 CPU 契约检查。"""
    result = run_cpu_smoke()
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return
    for check in result["checks"]:
        print(f"[ok] {check}")
    print(f"CPU smoke passed: {len(result['checks'])} checks")


def parse_args() -> argparse.Namespace:
    """定义公开子命令并返回解析后的参数。"""
    parser = argparse.ArgumentParser(
        prog="shopping-grpo",
        description="Shopping GRPO public CPU/offline utilities",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    command = commands.add_parser(
        "smoke",
        help="run pure-CPU contract checks without models or ShopSimulator",
    )
    command.add_argument("--json", action="store_true")
    command.set_defaults(handler=_smoke)

    command = commands.add_parser(
        "evaluate",
        help="normalize saved rollouts and report deterministic metrics",
    )
    command.add_argument("trajectories", type=Path)
    command.add_argument("--output", type=Path)
    command.add_argument("--force", action="store_true")
    command.set_defaults(handler=_offline_evaluate)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
