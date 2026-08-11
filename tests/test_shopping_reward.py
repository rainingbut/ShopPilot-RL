"""Pure-function tests for the Environment v2.1 / Reward v3 contract."""

import unittest

from shopping_grpo.training.grpo.adapter.runtime import (
    make_runtime_state,
    record_action_attempt,
    reward_breakdown,
    validate_reward,
)


def reward_detail(*, reward_type="gold_purchase", utility=1.0):
    pass_gate = {
        "status": "pass",
        "passed": True,
        "verifiable": True,
        "comparator": "exact",
        "source_field": "catalog",
    }
    return {
        "reward_version": "shopsimulator-reward-v3",
        "reward_type": reward_type,
        "reward_valid": True,
        "termination_reason": reward_type,
        "target_asin_match": reward_type == "gold_purchase",
        "hard_gates": {"category": pass_gate, "budget": pass_gate},
        "weighted_score": 1.0,
        "evidence_coverage": 1.0,
        "dimension_scores": {
            "brand": 1.0,
            "model": 1.0,
            "core_functions": 1.0,
            "key_options": 1.0,
        },
        "terminal_utility": float(utility),
        "purchase_success": reward_type
        in {"gold_purchase", "valid_alternative_purchase"},
        "sampling_invalid": False,
    }


def terminal_state(*, steps=8, reward_type="gold_purchase", native_reward=1.0):
    detail = validate_reward(
        reward_detail(reward_type=reward_type, utility=native_reward)
    )
    state = make_runtime_state(task_id=1, max_steps=35)
    state["steps"] = [{"index": index} for index in range(steps)]
    state.update(
        {
            "done": True,
            "terminal_result": {"done": True, "over": True},
            "final_reward": native_reward,
            "reward_version": detail["reward_version"],
            "reward_type": detail["reward_type"],
            "reward_valid": detail["reward_valid"],
            "reward_detail": detail,
        }
    )
    return state


class ShoppingRewardTest(unittest.TestCase):
    def test_gold_purchase_uses_environment_terminal_utility(self):
        result = reward_breakdown(terminal_state())

        self.assertEqual(result["full"], 1.0)
        self.assertEqual(result["strict"], 1.0)
        self.assertEqual(result["semantic"], 1.0)
        self.assertEqual(result["total"], 1.0)
        self.assertFalse(result["sampling_invalid"])

    def test_valid_alternative_is_success_but_not_strict_gold(self):
        result = reward_breakdown(
            terminal_state(
                reward_type="valid_alternative_purchase",
                native_reward=0.55,
            )
        )

        self.assertEqual(result["full"], 0.0)
        self.assertEqual(result["strict"], 0.0)
        self.assertEqual(result["purchase_success"], 1.0)
        self.assertEqual(result["native"], 0.55)
        self.assertEqual(result["total"], 0.55)

    def test_unfinished_trajectory_is_sampling_invalid(self):
        state = make_runtime_state(task_id=1, max_steps=35)
        state["termination_reason"] = "assistant_finished_without_environment_done"
        state["error"] = state["termination_reason"]

        result = reward_breakdown(state)

        self.assertEqual(result["total"], 0.0)
        self.assertTrue(result["sampling_invalid"])
        self.assertTrue(result["infrastructure_invalid"])

    def test_reward_v3_rejects_wrong_version_and_invalid_numbers(self):
        wrong_version = reward_detail()
        wrong_version["reward_version"] = "legacy-reward"
        with self.assertRaisesRegex(ValueError, "unsupported reward_version"):
            validate_reward(wrong_version)

        nonfinite = reward_detail()
        nonfinite["terminal_utility"] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            validate_reward(nonfinite)

        out_of_bounds = reward_detail()
        out_of_bounds["weighted_score"] = 1.1
        with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
            validate_reward(out_of_bounds)

    def test_reward_v3_is_minimized_before_storage(self):
        detail = reward_detail()
        detail["hidden_answer"] = "must not survive"

        public = validate_reward(detail)

        self.assertNotIn("hidden_answer", public)
        self.assertEqual(public["reward_type"], "gold_purchase")

    def test_same_action_on_same_page_within_three_attempts_is_repeated(self):
        state = make_runtime_state(task_id=1, max_steps=35)

        record_action_attempt(state, "search_products", {"query": "mug"}, "search page")
        record_action_attempt(state, "open_product", {"asin": "123"}, "search page")
        record_action_attempt(state, "search_products", {"query": "mug"}, "search page")

        self.assertEqual(state["action_attempt_count"], 3)
        self.assertEqual(state["repeat_action_count"], 1)
        self.assertAlmostEqual(reward_breakdown(state)["repeat_action_rate"], 1 / 3)

    def test_different_parameters_or_page_are_not_repeated(self):
        state = make_runtime_state(task_id=1, max_steps=35)

        record_action_attempt(state, "search_products", {"query": "mug"}, "page 1")
        record_action_attempt(state, "search_products", {"query": "cup"}, "page 1")
        record_action_attempt(state, "search_products", {"query": "mug"}, "page 2")

        self.assertEqual(state["repeat_action_count"], 0)

    def test_think_is_not_an_environment_action_attempt(self):
        state = make_runtime_state(task_id=1, max_steps=35)

        record_action_attempt(state, "think", {"note": "plan"}, "page")

        self.assertEqual(state["action_attempt_count"], 0)
        self.assertEqual(state["recent_action_signatures"], [])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
