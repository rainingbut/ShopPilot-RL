# SFT Data Collection Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Restore a reproducible Teacher-rollout-to-SFT-data pipeline that works with the repository's current ShopSimulator and training contracts.

**Architecture:** Reuse `evaluation/rollout.py` for model/environment interaction. Add one collection module for deterministic Reward v3 acceptance and SFT row construction, plus one resumable CLI that writes raw and derived artifacts together. Keep raw Teacher responses under `outputs/` and exclude held-out evaluation task IDs before producing train/validation files.

**Tech Stack:** Python standard library, existing OpenAI-compatible rollout client, ShopSimulator Environment v2.1, Reward v3, `unittest`.

---

### Task 1: Freeze the collection contract

**Files:**

- Create: `tests/test_sft_collection.py`
- Create: `src/shopping_grpo/collection/__init__.py`
- Create: `src/shopping_grpo/collection/sft.py`

1. Write failing tests for strict Reward v3 acceptance, action-only sanitization, held-out task exclusion and stable task-level splitting.
2. Run `env PYTHONPATH=src python3 -m unittest tests.test_sft_collection` and confirm imports fail.
3. Implement the minimum collection module using current `environment.actions`, `environment.tools` and `training.sft.dataset` interfaces.
4. Rerun the focused test and confirm it passes.

### Task 2: Add the resumable collection command

**Files:**

- Create: `tests/test_collect_sft_data_cli.py`
- Create: `scripts/collect_sft_data.py`

1. Write failing tests for batch paths, target-accepted stopping, worker forwarding and safe defaults.
2. Run the focused CLI tests and confirm the command is missing.
3. Implement one CLI that resumes from `raw.jsonl`, optionally collects concurrently, then rebuilds accepted, rejected, SFT, train, validation and metadata artifacts.
4. Rerun both focused test files.

### Task 3: Document and verify

**Files:**

- Modify: `docs/data-collection.md`
- Modify: `README.md`
- Modify: `README.en.md`

1. Document the runnable command, output layout and acceptance rules.
2. Run the collection, rollout, SFT dataset and CLI unit tests without contacting a model or ShopSimulator.
3. Run syntax and whitespace checks, review the final diff, commit and push `main`.
