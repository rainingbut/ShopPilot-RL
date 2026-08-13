import json
import tempfile
import unittest
from pathlib import Path

from scripts.hardware_profile import load_hardware_profile


class HardwareProfileTest(unittest.TestCase):
    def test_checked_in_profiles_have_audited_stage_settings(self):
        a100 = load_hardware_profile("a100_40g")
        a800 = load_hardware_profile("a800_80g")
        rtx4090 = load_hardware_profile("rtx4090_48g")

        self.assertIn("--qlora", a100["sft"]["arguments"])
        self.assertIn("--liger-kernel", a100["sft"]["arguments"])
        self.assertIn("--liger-kernel", a800["sft"]["arguments"])
        self.assertIn("--liger-kernel", rtx4090["sft"]["arguments"])
        self.assertEqual(
            a100["grpo"]["environment"]["SHOPPING_CONTEXT_WINDOW_TOKENS"],
            12288,
        )
        self.assertEqual(
            rtx4090["grpo"]["environment"]["SHOPPING_CONTEXT_WINDOW_TOKENS"],
            10240,
        )

    def test_profile_name_must_match_filename(self):
        payload = {
            "schema_version": "shopping-hardware-profile-v1",
            "name": "different",
            "grpo": {"hydra_overrides": [], "environment": {}},
            "sft": {"arguments": []},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.yaml"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "name must match filename"):
                load_hardware_profile(path)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
