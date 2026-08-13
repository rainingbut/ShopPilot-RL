# ShopPilot-RL 后续会话交接说明

更新时间：2026-08-13

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

### GRPO 仍处于 5-step 冒烟阶段

第一次使用 `a100_40g` profile（12,288 token、4 rollouts、4 并发、vLLM 0.28）时
发生 CUDA OOM：峰值约 `48176 / 49140 MiB`，随后仍需申请 `3.60 GiB`。失败后显存
已释放到约 1 MiB，无已知僵尸训练进程。

本地已准备、仍需同步到 AutoDL 的 48 GB 修改：

- 新增 `configs/hardware/rtx4090_48g.yaml`。
- 保留 `rollout.n=4`，总序列改为 10,240（prompt 2,048 + response 8,192）。
- `max_num_seqs` 和 Agent workers 改为 2，vLLM 显存比例改为 0.20。
- 工具响应预算改为 4,096，继续使用完整工具回合上下文压缩。
- `entropy_coeff=0` 时关闭 `calculate_entropy`，避免约 3.6 GiB、不进入 loss 的
  全词表 entropy 张量；preflight 会阻止该浪费重新出现。

潜在质量影响只主要来自 12K → 10K 上下文，极长轨迹会更早压缩。降低并发只影响
吞吐；关闭零权重 entropy 不改变训练目标；GRPO 组大小仍为 4。

重要：仓库已有 `experiments/grpo/summary.json` 是既有归档，不代表本次 AutoDL
GRPO 已完成。本次尚未成功跑完 5 steps，更没有本次正式 100-step 结果。

## 4. 新会话首先执行

1. 完整阅读根目录 `AGENTS.md` 和本文档。
2. 检查 `git status --short`，不得覆盖用户已有修改。
3. 确认 48 GB profile 修改已提交并推送到 `a100adaption`，再在 AutoDL 拉取。
4. 在 AutoDL 跑完整 pytest 和 dry-run。
5. 只重跑 5-step smoke；通过前不启动正式 100-step。

```bash
cd /root/autodl-tmp/ShopPilot-RL
git status --short
git switch a100adaption
git pull --ff-only origin a100adaption

.venv/bin/python -m pytest -q

PYTHONPATH=environments/ShopSimulator/shop_env \
  environments/ShopSimulator/.venv-shopsim/bin/python \
  -m pytest -q environments/ShopSimulator/shop_env/tests
```

期望主测试全部通过、ShopSimulator 43 passed；Ray deprecation warning 可忽略。

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

导出 GRPO 后：

```bash
cd /root/autodl-tmp/ShopPilot-RL
bash scripts/serve_model.sh outputs/models/grpo-merged \
  2>&1 | tee outputs/logs/serve-grpo.log
```

检查：

```bash
curl --fail --show-error http://127.0.0.1:8000/v1/models \
  | .venv/bin/python -m json.tool
```

**运行 GRPO 前必须停止终端 2 的 vLLM，确保 GPU 基本空闲。** 终端 1 的 CPU 环境
服务继续运行。

### 终端 3：训练或评测

完整评测示例（须用户明确授权）：

```bash
cd /root/autodl-tmp/ShopPilot-RL
bash scripts/evaluate.sh sft 2>&1 | tee outputs/logs/sft-200.log
```

不同模型使用不同 label：`baseline`、`sft`、`grpo`，避免覆盖。

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

## 6. 下一步：重跑 5-step GRPO smoke

先 dry-run：

```bash
cd /root/autodl-tmp/ShopPilot-RL
bash scripts/grpo.sh \
  --hardware-profile rtx4090_48g \
  --output outputs/smoke/grpo-rtx4090-48g-v1 \
  --dry-run
```

必须看到：

```text
data.max_response_length=8192
actor_rollout_ref.actor.calculate_entropy=false
actor_rollout_ref.rollout.n=4
actor_rollout_ref.rollout.gpu_memory_utilization=0.20
actor_rollout_ref.rollout.max_num_seqs=2
actor_rollout_ref.rollout.max_model_len=10240
```

停止模型服务并确认 GPU 空闲后：

```bash
cd /root/autodl-tmp/ShopPilot-RL
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p outputs/logs

bash scripts/grpo.sh \
  --hardware-profile rtx4090_48g \
  --output outputs/smoke/grpo-rtx4090-48g-v1 \
  -- \
  trainer.total_training_steps=5 \
  trainer.save_freq=5 \
  trainer.test_freq=50 \
  trainer.val_before_train=false \
  2>&1 | tee outputs/logs/grpo-rtx4090-48g-5step.log
```

不要复用第一次失败的 `outputs/smoke/grpo-a100-40g`。如果新目录非空，先检查内容，
再改用 `...-v2` 等新目录；不要直接删除未知产物。

完成后收集：

```bash
tail -n 200 outputs/logs/grpo-rtx4090-48g-5step.log

find outputs/smoke/grpo-rtx4090-48g-v1 \
  -maxdepth 3 -type f | sort | tail -100

wc -l outputs/smoke/grpo-rtx4090-48g-v1/training_diagnostics.jsonl

grep -E "CUDA out of memory|OutOfMemory|ERROR|Traceback|global_step|reward|effective|skipped" \
  outputs/logs/grpo-rtx4090-48g-5step.log | tail -160

nvidia-smi
df -h /root/autodl-tmp
```

通过标准：

- 无 OOM、Ray worker crash 或环境协议错误。
- 真正完成 5 个 optimizer steps，而非只生成 5 批 rollout。
- 产生 `global_step_5` 和 `training_diagnostics.jsonl`。
- Reward 有限且存在 reward-varying/effective groups。
- `skipped_update` 未连续触发上限。
- 峰值最好保留 3–5 GiB，抵御正式训练的长度波动。

## 7. Smoke 成功后的工作

先分析 generation batch、optimizer step、skipped update、有效 group、Reward 分布、
response length、overlong/context compaction、峰值显存和单步耗时，不立即启动正式训练。

若余量不足 3 GiB，应继续减小上下文或缓存，不要优先把 `rollout.n` 从 4 降到 2。
若余量明显充足且压缩过多，可以测试 10K 和 12K 之间的档位，但必须重新 smoke。

经用户明确同意后，目标 100-step 命令为：

```bash
bash scripts/grpo.sh \
  --hardware-profile rtx4090_48g \
  --output outputs/models/grpo-rtx4090-48g-100step \
  -- \
  trainer.total_training_steps=100 \
  trainer.save_freq=25 \
  trainer.test_freq=25
```

仓库默认是 500 steps，因此 100-step 必须显式覆盖。按 validation 指标选择 checkpoint，
不能只因为它是最后一步便自动选用。经用户明确同意后导出：

```bash
bash scripts/export_grpo.sh \
  outputs/models/grpo-rtx4090-48g-100step/global_step_<N>/actor \
  outputs/models/grpo-merged
```

启动导出模型后先跑 10 题 pilot；正常后再经明确授权运行冻结 200 题：

```bash
bash scripts/evaluate.sh grpo 2>&1 | tee outputs/logs/grpo-200.log
```

最终比较 strict success、purchase success、mean reward、weighted score、done、
repeat loop、wrong purchase、reward unverifiable、guard rejection、context overflow 和
平均步数，并报告 Baseline → SFT → GRPO 的绝对及相对提升。

## 8. 产物与磁盘

- SFT 峰值约 10.46 GiB 不代表 GRPO；GRPO 首次峰值已接近 48 GiB。
- 定期运行 `df -h /root/autodl-tmp` 和 `du -sh outputs/* cache/*`。
- 保留 SFT merged model、正式 GRPO checkpoint、diagnostics、完整评测轨迹、summary、
  训练/GPU 日志和 commit hash。
- 不直接删除未检查的目录；失败重跑优先用新的版本化输出目录。
- AutoDL 关机前把关键产物下载或同步至持久数据盘。

## 9. 新会话开场提示词

```text
请先完整阅读 AGENTS.md 和 docs/HANDOFF.md，再检查 git status，不要覆盖已有改动。
当前真实进度是 Baseline 和 SFT 已完成，GRPO 12K 5-step smoke 因 48GB OOM 失败；
rtx4090_48g 的 10K 配置已在本地准备。下一步是同步、完整测试和 dry-run，然后只跑
5-step smoke。不要把 experiments/grpo/summary.json 当成本次训练结果，也不要在未经
我确认时启动正式训练、合并模型或完整 200 题评测。
```
