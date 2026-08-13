# ShopPilot-RL 后续会话交接说明

更新时间：2026-08-13（GRPO step 75 正式评测完成后）

## 1. 项目目标与约束

本项目的唯一流程是：

```text
Baseline → SFT → GRPO → Evaluation
```

目标是在 AutoDL 单张 RTX 4090 48 GB vGPU 上完成可复现的购物 Agent 训练与评测：
SFT 学习合法的搜索、核验、规格选择和购买流程；GRPO 在线采样并优化终局 Reward；
最后在冻结的 200 题留出集上比较 Baseline、SFT 和 GRPO。

不可变约束：

- ShopSimulator Environment v2.1、Reward v3、observation v2、tool schema v2。
- 严格成功只统计完整 `gold_purchase` 且 `reward_valid=true`。
- `data/evaluation/tasks.jsonl` 只能评测，不能进入 SFT 或 GRPO 训练。
- 不增加旧协议兼容层、历史数据集、机器绝对路径或第二套实验流程。
- 未经用户明确授权，不启动训练、不合并模型、不运行完整 200 题评测。

## 2. 环境

本地工作区：

```text
D:\Projects\Agentic\ShopPilot-RL
branch: a100adaption
```

AutoDL：

```text
/root/autodl-tmp/ShopPilot-RL
GPU: RTX 4090 48 GB vGPU（约 49140 MiB）
主环境: .venv / Python 3.12.3
ShopSimulator: environments/ShopSimulator/.venv-shopsim
torch: 2.11.0+cu130
transformers: 5.15.0.dev0
vllm: 0.25.1
driver: 595.71.05
```

每次重新开机后建议执行：

```bash
cd /root/autodl-tmp/ShopPilot-RL
mkdir -p /root/autodl-tmp/cache/{huggingface,uv,pip}
mkdir -p /root/autodl-tmp/tmp outputs/logs
export HF_HOME=/root/autodl-tmp/cache/huggingface
export UV_CACHE_DIR=/root/autodl-tmp/cache/uv
export PIP_CACHE_DIR=/root/autodl-tmp/cache/pip
export TMPDIR=/root/autodl-tmp/tmp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

首次部署才需要：

```bash
python3 -m pip install -U uv
bash scripts/setup.sh
```

## 3. 当前真实进度

### 基础设施和数据审计已完成

- 主项目曾通过 `188 passed, 15 subtests passed`；此后又增加了测试，下一次同步后需
  重新跑完整 pytest。
- ShopSimulator：`43 passed`。
- evaluation、SFT train/validation、GRPO train/validation 两两无 task_id 重叠。
- GRPO parquet SHA-256 与 `data/grpo/metadata.json` 一致。

### Baseline 200 已完成

```text
completed: 200/200
errors: 0
strict success: 2/200 = 1.0%
done: 33/200 = 16.5%
context overflow: 0
critical footer failure: 0
```

AutoDL 产物：

```text
outputs/evaluation/baseline/trajectories.jsonl
outputs/evaluation/baseline/summary.json
outputs/logs/baseline-200.log
```

### SFT 训练和评测已完成

训练：792 train、198 validation、3 epochs、297 steps。

```text
train_loss: 0.31863
eval_loss: 0.35193 → 0.34032 → 0.34199
peak GPU: 约 10.46 GiB
耗时: 约 205.2 分钟
```

模型：

```text
outputs/models/sft-lora
outputs/models/sft-merged
```

10 题 pilot 严格成功 `7/10`。本次 AutoDL 正式 200 题严格成功
`122/200 = 61.0%`，正常终止 `199/200 = 99.5%`，无 context overflow。

注意：仓库 `experiments/sft/summary.json` 的归档数字是 `121/200 = 60.5%`，与本次
AutoDL 输出略有差别。正式报告以保存的本次 AutoDL 原始轨迹、summary、commit 和
配置为准。

### GRPO 训练、导出和正式评测已完成

历史故障：第一次使用 `a100_40g` profile（12,288 token、4 rollouts、4 并发）时发生
CUDA OOM，峰值约 `48176 / 49140 MiB`，随后仍需申请 `3.60 GiB`。该失败目录不是本次
正式结果。

之后改用已提交的 `rtx4090_48g` profile：

- 总序列 10,240（prompt 2,048 + response 8,192），`rollout.n=4`。
- `max_num_seqs=2`、Agent workers 2、vLLM 显存比例 0.20。
- 工具响应预算 4,096，并启用完整工具回合上下文压缩。
- `entropy_coeff=0` 时设置 `calculate_entropy=false`，避免不进入 loss 的全词表
  entropy 张量。

5-step smoke 使用 `outputs/smoke/grpo-rtx4090-48g-v2`，完成 5 个真实 optimizer
steps，产生 `global_step_5`，无 skipped update、OOM、overlong 或 infrastructure
invalid。GPU 峰值为 `36322 / 49140 MiB`，余量约 12.5 GiB。日志末尾的
`DataLoader worker ... Killed` 发生在训练、checkpoint 和 final validation 完成后的
`atexit` 收尾阶段，不影响结果。

正式训练使用：

```text
outputs/models/grpo-rtx4090-48g-100step
outputs/logs/grpo-rtx4090-48g-100step.log
```

训练完成 100 个 optimizer steps，无缺步；共记录 170 个 generation batches、100 个
optimizer steps 和 9 个 skipped updates。9 次跳过均不连续，最大连续次数为 1，远低于
上限 10。训练用 400 条 rollout 全部正常终止，无 overlong、infrastructure invalid
或 reward unverifiable。最大 response length 为 5,434；约 42% 的训练 rollout 触发
上下文压缩，无 critical footer failure。Torch 最大 allocated 约 22.92 GiB、reserved
约 35.07 GiB，CPU memory 指标最高约 95.94 GiB；总耗时约 1 小时 36 分。

各 checkpoint 的冻结 validation reward：

```text
step 0:   0.1216921
step 25:  0.1766312
step 50:  0.1984645
step 75:  0.2139387  <- 最优
step 100: 0.1673449
```

按预先约定的 validation 指标选择 `global_step_75`，不使用最后的 step 100。已导出为：

```text
outputs/models/grpo-step75-merged-v1
```

导出目录约 4.2 GiB，含 `model.safetensors`、`config.json`、tokenizer、chat template
和 LoRA adapter。10 题 pilot 完成 10/10，严格成功 7/10，mean reward 0.661995，
无 context overflow 或基础设施错误。

冻结 200 题正式评测已完成，详见第 7 节。重要：仓库已有
`experiments/grpo/summary.json` 仍是历史归档，不能作为本次 AutoDL 结果。本次结果只以
`outputs/evaluation/grpo-step75-200-v1` 的轨迹、summary、日志、commit 和配置为准。

## 4. 新会话首先执行

1. 完整阅读根目录 `AGENTS.md` 和本文档。
2. 检查 `git status --short`，不得覆盖用户已有修改。
3. 不要重跑训练、重新选 checkpoint、重复导出或重复运行 200 题评测。
4. 确认 WinSCP 备份及 SHA-256 校验完整，特别是训练 checkpoint、step 75 merged、
   diagnostics、日志以及正式评测轨迹和 summary。
5. 如需整理最终报告，只读取本次真实产物；不要引用历史
   `experiments/grpo/summary.json` 代替本次结果。

```bash
cd /root/autodl-tmp/ShopPilot-RL
git status --short
git rev-parse HEAD
df -h /root/autodl-tmp

cat outputs/evaluation/grpo-step75-200-v1/summary.json
wc -l outputs/evaluation/grpo-step75-200-v1/trajectories.jsonl
```

## 5. 各终端启动指令

### 终端 1：ShopSimulator

Baseline、评测和 GRPO 都需要它：

```bash
cd /root/autodl-tmp/ShopPilot-RL
bash scripts/start_environment.sh \
  2>&1 | tee outputs/logs/shop-environment.log
```

健康检查优先使用真实 smoke：

```bash
.venv/bin/python scripts/smoke_shop_env.py \
  --task-id 0 \
  --actions "search[乳胶枕]"
```

### 终端 2：模型服务（只用于 Baseline/SFT/GRPO 评测）

Baseline：

```bash
cd /root/autodl-tmp/ShopPilot-RL
bash scripts/serve_model.sh Qwen/Qwen3.5-2B \
  2>&1 | tee outputs/logs/serve-baseline.log
```

SFT：

```bash
cd /root/autodl-tmp/ShopPilot-RL
bash scripts/serve_model.sh outputs/models/sft-merged \
  2>&1 | tee outputs/logs/serve-sft.log
```

本次 GRPO step 75：

```bash
cd /root/autodl-tmp/ShopPilot-RL
bash scripts/serve_model.sh outputs/models/grpo-step75-merged-v1 \
  2>&1 | tee outputs/logs/serve-grpo-step75-v1.log
```

检查：

```bash
curl --fail --show-error http://127.0.0.1:8000/v1/models \
  | .venv/bin/python -m json.tool
```

**运行 GRPO 前必须停止终端 2 的 vLLM，确保 GPU 基本空闲。** 终端 1 的 CPU 环境
服务继续运行。

### 终端 3：训练或评测

本次正式评测已经完成，不应再次执行。原命令和输出目录为：

```bash
cd /root/autodl-tmp/ShopPilot-RL
EVAL_OUTPUT_DIR="$PWD/outputs/evaluation/grpo-step75-200-v1" \
  bash scripts/evaluate.sh grpo-step75 \
  2>&1 | tee outputs/logs/grpo-step75-200-v1.log
```

该目录已有 200 条完整轨迹和 summary；不要覆盖。评测器虽可跳过已完成题目，但没有
新的明确目的时不应重复执行。

### 终端 4：GPU 和磁盘监控

```bash
cd /root/autodl-tmp/ShopPilot-RL
nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader,nounits -l 2 \
  | tee outputs/logs/grpo-rtx4090-48g-gpu.csv
```

可另开窗口：

```bash
watch -n 5 'nvidia-smi; echo; df -h /root/autodl-tmp; echo; free -h'
```

## 6. 已完成的 GRPO 产物

必须保留：

```text
outputs/models/grpo-rtx4090-48g-100step/
outputs/models/grpo-step75-merged-v1/
outputs/logs/grpo-rtx4090-48g-100step.log
outputs/evaluation/grpo-step75-pilot-10-v1/
outputs/logs/grpo-step75-pilot-10-v1.log
outputs/evaluation/grpo-step75-200-v1/
outputs/logs/grpo-step75-200-v1.log
outputs/logs/serve-grpo-step75-v1.log
```

`grpo-rtx4090-48g-100step` 约 42 GiB，包含 step 25/50/75/100、optimizer state、
`data.pt` 和 `training_diagnostics.jsonl`。最佳 checkpoint 是 step 75，但在完整备份和
校验前不要删除其他 checkpoint。merged 模型约 4.2 GiB。

## 7. 冻结 200 题最终结果

本次真实 AutoDL 对比：

| 模型 | 完成 | 严格成功 | 严格成功率 |
|---|---:|---:|---:|
| Baseline | 200/200 | 2/200 | 1.0% |
| SFT | 200/200 | 122/200 | 61.0% |
| GRPO step 75 | 200/200 | 123/200 | 61.5% |

Baseline → SFT 是主要提升，绝对增加 60.0 个百分点。GRPO step 75 相比 SFT 只增加
1 个严格成功，绝对提升 0.5 个百分点、相对提升约 0.82%；应表述为持平或微增，不能
宣称显著提升。

GRPO step 75 正式评测：

```text
expected/completed: 200/200
done: 200/200 = 100.0%
strict success: 123/200 = 61.5%
purchase success: 123/200 = 61.5%
mean reward / terminal utility: 0.5188845791
mean weighted score: 0.7730666590
average steps: 9.42
gold_purchase: 123
partial_alternative_purchase: 46
wrong_purchase: 6
repeat_loop: 18
max_steps: 4
reward_unverifiable: 3
reward_valid: 197/200 = 98.5%
context overflow: 0
critical footer failure: 0
guard rejection: 16
evaluation exit code: 0
```

`max_context_input_tokens=29336` 统计的是压缩前原始上下文，不是实际提交给 vLLM 的
请求长度。正式协议输入预算为 `24576 - 512 - 512 = 23552`，超预算时先按完整工具
回合压缩；因此该值与 `context_overflow_tasks=0` 不矛盾。

## 8. 产物与磁盘

- AutoDL 数据盘已扩容为 125 GiB；正式评测结束时使用 91 GiB、剩余约 35 GiB。
- SFT 峰值约 10.46 GiB 不代表 GRPO；失败的 12K profile 曾接近 48 GiB，成功的
  10K profile smoke 峰值为 36,322 MiB。
- 定期运行 `df -h /root/autodl-tmp` 和 `du -sh outputs/* cache/*`。
- 保留 SFT merged model、正式 GRPO checkpoint、diagnostics、完整评测轨迹、summary、
  训练/GPU 日志和 commit hash。
- 不直接删除未检查的目录；失败重跑优先用新的版本化输出目录。
- 已在 `outputs/backups/<timestamp>` 生成备份元数据和 SHA-256 清单；WinSCP 本地备份
  正在/已经进行。所有 `SHA256SUMS` 校验为 OK 前，不删除 AutoDL 产物。
- 还要把本次新增的 pilot、正式 200 题目录及对应日志补充下载。
- 完成备份后可停止 vLLM 和 ShopSimulator；不要在 vLLM 占用约 44.5 GiB 显存时启动
  任何训练。

## 9. 新会话开场提示词

```text
请先完整阅读 AGENTS.md 和 docs/HANDOFF.md，再检查 git status，不要覆盖已有改动。
当前 Baseline、SFT、rtx4090_48g 10K GRPO 100-step、step 75 导出、10 题 pilot 和
冻结 200 题评测均已完成。最佳 checkpoint 是 global_step_75；本次真实严格成功率为
Baseline 1.0%、SFT 61.0%、GRPO step 75 61.5%，GRPO 仅微增 0.5 个百分点。下一步只做
备份校验、产物归档和最终报告，不要重跑训练、重新选 checkpoint、重复合并或重复评测，
也不要把 experiments/grpo/summary.json 当作本次结果。每个新阶段仍需先等我确认。
```
