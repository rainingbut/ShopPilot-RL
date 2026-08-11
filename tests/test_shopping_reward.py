"""Shopping GRPO 终局奖励的纯函数测试。"""

import unittest

from shopping_grpo.training.grpo.adapter.runtime import (
    make_runtime_state,
    record_action_attempt,
    reward_breakdown,
    validate_reward_components,
)


def terminal_state(*, steps=8, components=None, native_reward=1.0):
    state = make_runtime_state(task_id=1, max_steps=35)
    state["steps"] = [{"index": index} for index in range(steps)]
    state.update(
        {
            "done": True,
            "terminal_result": {"done": True, "over": True},
            "final_reward": native_reward,
            "reward_components": components
            or {"r_type": 1.0, "r_att": 1.0, "r_option": 1.0, "r_price": 1.0},
        }
    )
    return state


class ShoppingRewardTest(unittest.TestCase):
    def test_full_success_gets_semantic_and_eight_step_efficiency_reward(self):
        result = reward_breakdown(terminal_state(steps=8))

        self.assertAlmostEqual(result["full"], 1.0)
        self.assertAlmostEqual(result["strict"], 1.0)
        self.assertAlmostEqual(result["semantic"], 1.7)
        self.assertAlmostEqual(result["efficiency"], 0.05 * (1 - 8 / 35))
        self.assertAlmostEqual(result["total"], 1.7 + 0.05 * (1 - 8 / 35))

    def test_full_success_at_step_limit_has_no_efficiency_or_overlong_penalty(self):
        result = reward_breakdown(terminal_state(steps=35))

        self.assertEqual(result["efficiency"], 0.0)
        self.assertEqual(result["penalty_overlong"], 0.0)
        self.assertEqual(result["total"], 1.7)

    def test_unfinished_assistant_gets_small_negative_reward(self):
        state = make_runtime_state(task_id=1, max_steps=35)
        state["termination_reason"] = "assistant_finished_without_environment_done"
        state["error"] = state["termination_reason"]

        result = reward_breakdown(state)

        self.assertEqual(result["semantic"], 0.0)
        self.assertEqual(result["penalty_unfinished"], 0.05)
        self.assertEqual(result["total"], -0.05)

    def test_reward_components_must_be_complete_finite_and_bounded(self):
        with self.assertRaisesRegex(ValueError, "missing"):
            validate_reward_components({"r_type": 1, "r_att": 1})
        with self.assertRaisesRegex(ValueError, "finite"):
            validate_reward_components(
                {"r_type": 1, "r_att": 1, "r_option": float("nan"), "r_price": 1}
            )
        with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
            validate_reward_components(
                {"r_type": 1, "r_att": 1, "r_option": 1, "r_price": 1.1}
            )

    def test_partial_purchase_preserves_native_and_product_reward(self):
        state = terminal_state(
            components={"r_type": 1, "r_att": 1, "r_option": 0.5, "r_price": 1},
            native_reward=0.6,
        )

        result = reward_breakdown(state)

        self.assertEqual(result["full"], 0.0)
        self.assertEqual(result["strict"], 0.5)
        self.assertEqual(result["native"], 0.6)
        self.assertAlmostEqual(result["semantic"], 0.5 * 0.5 + 0.2 * 0.6)

    def test_malformed_terminal_components_are_infrastructure_invalid(self):
        state = terminal_state()
        state["reward_components"]["r_option"] = float("nan")

        result = reward_breakdown(state)

        self.assertTrue(result["infrastructure_invalid"])
        self.assertEqual(result["total"], 0.0)

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
