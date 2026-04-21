from unittest.mock import MagicMock, patch

from transformers import TrainingArguments

from engram_peft.trl import create_engram_sft_trainer, prepare_engram_for_sft


def test_prepare_engram_for_sft():
    """Verify that model preparation correctly sets training flags."""
    mock_model = MagicMock()
    # Mock nested config
    mock_model.base_model.config.use_cache = True

    prepare_engram_for_sft(mock_model)

    # Check if cache was disabled
    assert mock_model.base_model.config.use_cache is False
    # Check if gradient checkpointing and train mode were set
    mock_model.gradient_checkpointing_enable.assert_called_once()
    mock_model.train.assert_called_once()
    mock_model.print_trainable_parameters.assert_called_once()


@patch("engram_peft.trl.SFTTrainer")
def test_create_engram_sft_trainer(mock_trainer_class):
    """Verify that the factory function initializes SFTTrainer with correct defaults."""
    mock_model = MagicMock()
    mock_model.config = MagicMock()
    # Mocking required config attributes for EngramDataCollator
    mock_model.config.target_layers = [0]
    mock_model.config.engram_vocab_size_per_ngram = [1000]
    mock_model.config.ngram_sizes = [2]
    mock_model.config.n_head_per_ngram = 1
    mock_model.config.compressed_vocab_size = 1000
    mock_model.config.pad_id = 0
    mock_model.config.seed = 42

    mock_tokenizer = MagicMock()
    mock_tokenizer.pad_token_id = 0
    mock_dataset = MagicMock()
    training_args = TrainingArguments(output_dir="tmp_test_trl")

    create_engram_sft_trainer(
        model=mock_model,
        tokenizer=mock_tokenizer,
        train_dataset=mock_dataset,
        args=training_args,
    )

    # Verify model preparation was performed
    assert mock_model.base_model.config.use_cache is False

    # Verify SFTTrainer was instantiated
    mock_trainer_class.assert_called_once()
    _, kwargs = mock_trainer_class.call_args
    assert kwargs["model"] == mock_model
    assert kwargs["processing_class"] == mock_tokenizer
    assert "data_collator" in kwargs

    from engram_peft.collator import EngramDataCollator

    assert isinstance(kwargs["data_collator"], EngramDataCollator)
