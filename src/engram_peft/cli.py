import logging
from pathlib import Path
from typing import Annotated, Any, cast

import typer
import yaml
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

from engram_peft.collator import EngramDataCollator
from engram_peft.config import EngramConfig
from engram_peft.model import get_engram_model
from engram_peft.trainer import EngramTrainer

app = typer.Typer(help="Engram-PEFT Command Line Interface")


@app.callback()
def callback() -> None:
    """
    Engram-PEFT: Sparse Memory Injection for Parameter-Efficient Fine-Tuning.
    """
    pass


logger = logging.getLogger(__name__)


def parse_override_value(value: str) -> Any:
    """Parses a string value into its inferred type (int, float, bool, or str)."""
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    if value.lower() == "none":
        return None

    try:
        # Check if it's an integer (no decimal point or scientific notation)
        if value.isdigit() or (value.startswith("-") and value[1:].isdigit()):
            return int(value)
        # Try float (handles decimals and scientific notation like 1e-4)
        return float(value)
    except ValueError:
        return value


def apply_overrides(config: dict[str, Any], overrides: list[str]) -> None:
    """Applies dot-notated overrides to a nested dictionary."""
    for override in overrides:
        if "=" not in override:
            logger.warning(f"Invalid override format (missing '='): {override}")
            continue
        key_path, value_str = override.split("=", 1)
        value = parse_override_value(value_str)

        keys = key_path.split(".")
        current = config
        for key in keys[:-1]:
            if key not in current or not isinstance(current[key], dict):
                current[key] = {}
            current = current[key]
        current[keys[-1]] = value


@app.command()
def train(
    config_path: Annotated[
        Path,
        typer.Option("--config", "-c", help="Path to the YAML configuration file."),
    ],
    overrides: Annotated[
        list[str] | None,
        typer.Option(
            "--overrides",
            "-o",
            help="Override config values (e.g. training_args.learning_rate=1e-5).",
        ),
    ] = None,
) -> None:
    """Trains an Engram-augmented model using the specified configuration."""
    if not config_path.exists():
        typer.secho(
            f"Error: Configuration file not found at {config_path}", fg=typer.colors.RED
        )
        raise typer.Exit(code=1)

    with open(config_path) as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            typer.secho(f"Error parsing YAML: {e}", fg=typer.colors.RED)
            raise typer.Exit(code=1) from e

    if overrides:
        apply_overrides(config, overrides)

    # Validate mandatory sections
    required_keys = ["model_name_or_path"]
    for k in required_keys:
        if k not in config:
            typer.secho(
                f"Error: Missing required config key '{k}'", fg=typer.colors.RED
            )
            raise typer.Exit(code=1)

    model_name = config["model_name_or_path"]
    engram_dict = config.get("engram_config", {})
    training_dict = config.get("training_args", {})
    data_dict = config.get("data_args", {})

    typer.echo(f"[*] Loading tokenizer & model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Build EngramConfig
    # We allow overrides to populate engram_config correctly
    engram_config = EngramConfig(tokenizer_name_or_path=model_name, **engram_dict)

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype="auto", device_map="auto"
    )

    typer.echo("[*] Wrapping model with Engram layers")
    model = get_engram_model(base_model, engram_config, tokenizer=tokenizer)

    typer.echo(f"[*] Loading dataset: {data_dict.get('dataset_name', 'unknown')}")
    dataset = load_dataset(
        data_dict.get("dataset_name"),
        data_dict.get("dataset_config_name"),
        split=data_dict.get("split", "train"),
    )

    def tokenize_function(examples: dict[str, Any]) -> dict[str, Any]:
        text_column = data_dict.get("text_column", "text")

        return cast(
            "dict[str, Any]",
            tokenizer(
                examples[text_column],
                truncation=True,
                max_length=data_dict.get("max_length", 512),
                padding=False,
            ),
        )

    typer.echo("[*] Tokenizing dataset")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Running tokenizer on dataset",
    )

    data_collator = EngramDataCollator(
        tokenizer=tokenizer, config=engram_config, mlm=False
    )

    typer.echo("[*] Initializing Trainer")
    # Ensure output_dir is present and not duplicated in **training_dict
    output_dir = training_dict.pop("output_dir", "./outputs")
    training_args = TrainingArguments(output_dir=output_dir, **training_dict)

    trainer = EngramTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    typer.echo("[*] Starting training...")
    trainer.train()

    typer.echo(f"[*] Saving adapter to {training_args.output_dir}")
    out_dir = training_args.output_dir
    assert out_dir is not None, "training_args.output_dir must be specified"
    model.save_pretrained(out_dir)
    typer.secho("Success: Training complete.", fg=typer.colors.GREEN, bold=True)


@app.command()
def evaluate() -> None:
    """[Placeholder] Evaluates an Engram-augmented model on a test set."""
    typer.echo("Evaluate functionality coming soon...")


if __name__ == "__main__":
    app()
