import unittest
from unittest.mock import MagicMock, patch

import torch.nn as nn

from engram_peft.utils.peft_patches import apply_peft_patches


class Gemma4ClippableLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)


class TestPeftPatches(unittest.TestCase):
    def test_peft_patch_redirection(self):
        """Verify that Gemma4ClippableLinear is redirected to its inner .linear."""
        import peft.tuners.lora.model as lora_model

        # 1. Create a mock for the original method
        original_mock = MagicMock(return_value="success")

        # 2. Manually patch the class for the test
        with patch.object(lora_model.LoraModel, "_create_new_module", original_mock):
            # Ensure it's not marked as patched
            if hasattr(lora_model.LoraModel, "_is_engram_patched"):
                delattr(lora_model.LoraModel, "_is_engram_patched")

            # 3. Apply our patch
            apply_peft_patches()

            # 4. Create a dummy target
            target = Gemma4ClippableLinear()

            # 5. Call the now-patched method
            # In PEFT: _create_new_module(lora_config, adapter_name, target, ...)
            result = lora_model.LoraModel._create_new_module(
                "config", "adapter", target
            )

            # 6. Verify redirection
            # The original_mock should have been called with target.linear instead of target
            self.assertEqual(result, "success")
            args, kwargs = original_mock.call_args
            self.assertEqual(args[2], target.linear)
            self.assertNotEqual(args[2], target)

    def test_idempotency(self):
        """Verify that apply_peft_patches only applies once."""
        import peft.tuners.lora.model as lora_model

        apply_peft_patches()
        first_patch = lora_model.LoraModel._create_new_module

        apply_peft_patches()
        second_patch = lora_model.LoraModel._create_new_module

        self.assertEqual(first_patch, second_patch)


if __name__ == "__main__":
    unittest.main()
