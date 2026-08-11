# Baseline → SFT → GRPO

All three models were evaluated with one deterministic rollout on the same 200
held-out tasks.

| Model | Done | Strict success | Purchase success | Mean reward | Mean steps | Guard rejections |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5-2B baseline | 18.0% | 0.0% | 0.0% | -0.1105 | 5.875 | 752 |
| LoRA SFT | 96.5% | 60.5% | 60.5% | 0.4729 | 12.335 | 52 |
| GRPO step 100 | 96.5% | 62.0% | 62.5% | 0.5158 | 11.850 | 38 |

## Interpretation

SFT provides the dominant gain. It teaches the base model to use the action
protocol, continue through multi-step shopping tasks and reach valid terminal
states. GRPO then adds a smaller improvement: three additional strict successes,
one valid alternative purchase, fewer wrong purchases, fewer loops and fewer
guard rejections than SFT.

The result supports a practical training order: first establish reliable tool
behavior with SFT, then use online RL for constraint satisfaction and policy
refinement.

## Limits

- Each model has one rollout per task, so the table does not estimate sampling
  variance.
- The SFT and GRPO recipes target large single GPUs; results may differ with
  other distributed layouts or dependency versions.
- Seven SFT and GRPO tasks ended without a Reward v3 terminal record and remain
  in the denominator.
- Full model weights and rollout logs are generated artifacts, not committed
  repository files.

Machine-readable settings and summaries are stored beside each stage:

- [`baseline/`](baseline/)
- [`sft/`](sft/)
- [`grpo/`](grpo/)
