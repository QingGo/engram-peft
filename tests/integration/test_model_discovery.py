import pytest
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

from engram_peft.config import EngramConfig
from engram_peft.model import EngramModel, get_engram_model


@pytest.mark.parametrize(
    "model_id, expected_path, expected_layer_count",
    [
        ("Qwen/Qwen2.5-0.5B", "model.layers", 24),
        ("mistralai/Mistral-7B-v0.1", "model.layers", 32),
    ],
)
def test_model_discovery_registry(model_id, expected_path, expected_layer_count):
    """
    Verifies that the layer container is correctly identified via the registry.
    We use publicly accessible models for this test.
    """
    print(f"\nTesting registry discovery for: {model_id}")
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)

    with torch.device("meta"):
        base_model = AutoModel.from_config(config, trust_remote_code=True)

    engram_config = EngramConfig(target_layers=[0, 1], tokenizer_name_or_path=model_id)

    model = get_engram_model(base_model, engram_config)
    found_layers = model._find_transformer_layers()
    assert isinstance(found_layers, nn.ModuleList)
    assert len(found_layers) == expected_layer_count
    assert len(model._hook_handles) == len(engram_config.target_layers) + 1


def test_explicit_path_override():
    """
    Verifies that layer_container_path explicitly overrides discovery.
    """
    # Use any model config (even a simple one)
    config = AutoConfig.from_pretrained("Qwen/Qwen2.5-0.5B")
    with torch.device("meta"):
        base_model = AutoModel.from_config(config)

    # Intentionally provide a wrong but existing path (should error)
    # Note: AutoModel returns the base model, so the root-level attrs are things like 'layers'
    engram_config = EngramConfig(layer_container_path="norm", target_layers=[0])

    with pytest.raises(ValueError, match="is not a nn.ModuleList"):
        get_engram_model(base_model, engram_config)

    # Provide correct explicit path
    engram_config.layer_container_path = "layers"
    model = get_engram_model(base_model, engram_config)
    assert len(model._find_transformer_layers()) == 24


def test_mock_registry_discovery():
    """
    Verifies that ARCH_LAYER_MAPPING works even without downloading configs.
    """
    from engram_peft.model import ARCH_LAYER_MAPPING

    class MockModel(nn.Module):
        def __init__(self, model_type):
            super().__init__()
            paths = ARCH_LAYER_MAPPING.get(model_type, ["model.layers"])
            # Use the first path in the list for the mock structure
            path = paths[0]
            # Create a nested structure matching the registry
            parts = path.split(".")
            curr = self
            for part in parts[:-1]:
                setattr(curr, part, nn.Module())
                curr = getattr(curr, part)
            setattr(
                curr, parts[-1], nn.ModuleList([nn.Linear(10, 10) for _ in range(3)])
            )

            self.config = type("Config", (), {"model_type": model_type})()

    # Test ChatGLM style
    glm_model = MockModel("chatglm")
    model = get_engram_model(glm_model, EngramConfig(target_layers=[0]))
    assert len(model._find_transformer_layers()) == 3


def test_heuristic_fallback():
    """
    Verifies that the heuristic scanner finds the largest ModuleList
    when the architecture is unknown.
    """

    # Create a custom model that is NOT in the registry
    class CustomModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.my_custom_blocks = nn.ModuleList([nn.Linear(10, 10) for _ in range(5)])
            self.other_list = nn.ModuleList([nn.Linear(10, 10) for _ in range(2)])
            self.config = type("Config", (), {"model_type": "unknown_fancy_model"})()

    base_model = CustomModel()
    engram_config = EngramConfig(target_layers=[0])

    model = get_engram_model(base_model, engram_config)
    found_layers = model._find_transformer_layers()

    assert len(found_layers) == 5
    # Since we don't have a registry for 'unknown_fancy_model', it should have used the heuristic
    # finding 'my_custom_blocks' as it is the largest.
