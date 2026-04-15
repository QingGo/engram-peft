import torch
from transformers import AutoTokenizer, BertForSequenceClassification

from engram_peft import EngramConfig, get_engram_model


def test_bert_engram() -> None:
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Initialize BERT model
    print(f"Loading {model_name}...")
    base_model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # Configure Engram with Phase 2 features
    config = EngramConfig(
        target_layers=[2, 10],
        engram_vocab_size_per_ngram=[10000, 10000],
        hidden_size=base_model.config.hidden_size,
        embedding_dim=768,
        bidirectional_conv=True,  # Critical for BERT
        hashing_mode="centered",  # Optimal for bidirectional models
        stop_token_ids=[
            tokenizer.sep_token_id,
            tokenizer.cls_token_id,
        ],  # Boundary protection
        tokenizer_name_or_path=model_name,
        pad_id=tokenizer.pad_token_id,
    )

    print("Injecting Engram layers...")
    model = get_engram_model(base_model, config, tokenizer)
    model.print_trainable_parameters()

    # Dummy input
    inputs = tokenizer(
        "Hello, this is a test of Engram with BERT.", return_tensors="pt"
    )
    labels = torch.tensor([1])

    # Forward pass
    print("Running forward pass...")
    outputs = model(**inputs, labels=labels)
    loss = outputs.loss
    print(f"Loss: {loss.item()}")

    # Backward pass
    print("Running backward pass...")
    loss.backward()
    print("Backward pass successful!")

    # Verify hooks
    print(f"Hook handles attached: {len(model._hook_handles)}")
    assert len(model._hook_handles) > 0, "No hooks attached!"

    print("\nBERT Engram support verified successfully!")


if __name__ == "__main__":
    test_bert_engram()
