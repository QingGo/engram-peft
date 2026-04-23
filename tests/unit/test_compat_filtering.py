from typing import Any, cast

from transformers import TrainingArguments

from engram_peft.utils.compat import create_safe_training_args


def test_create_safe_training_args_filtering():
    # Standard arguments
    args = create_safe_training_args(
        output_dir="test_out", learning_rate=1e-4, per_device_train_batch_size=4
    )
    assert args.output_dir == "test_out"
    assert args.learning_rate == 1e-4
    assert args.per_device_train_batch_size == 4


def test_create_safe_training_args_with_garbage():
    # Pass unknown arguments that TrainingArguments would normally reject
    args = create_safe_training_args(
        output_dir="test_out", unknown_param="should_be_filtered", extra_garbage=123
    )
    assert args.output_dir == "test_out"
    # Verify it doesn't crash and returns a valid TrainingArguments object
    assert isinstance(args, TrainingArguments)
    # Check that unknown attributes are not set (though they shouldn't be anyway)
    assert not hasattr(args, "unknown_param")


def test_create_safe_training_args_from_dict():
    # Use cast to avoid TypedDict mismatch in tests
    args_dict = cast(
        "Any",
        {
            "output_dir": "test_dict",
            "num_train_epochs": 3.0,
            "invalid_field": "remove_me",
        },
    )
    args = create_safe_training_args(args_dict=args_dict, learning_rate=2e-5)
    assert args.output_dir == "test_dict"
    assert args.num_train_epochs == 3.0
    assert args.learning_rate == 2e-5
    assert not hasattr(args, "invalid_field")
