import importlib.resources
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, cast

import typer
import yaml
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    TrainingArguments,
)

from engram_peft.collator import EngramDataCollator
from engram_peft.config import EngramConfig
from engram_peft.model import TrainMode, get_engram_model
from engram_peft.trainer import EngramTrainer

if TYPE_CHECKING:
    import torch.nn as nn

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
def config_template(
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="Path to save the template.")
    ] = None,
) -> None:
    """Generates a clean YAML configuration template with sensible defaults."""

    # Load the template from package resources
    # importlib.resources.files is available in Python 3.9+
    template_file = importlib.resources.files("engram_peft").joinpath(
        "config_template.yaml"
    )
    content = template_file.read_text()

    if output:
        output.write_text(content)
        typer.secho(f"Template saved to {output}", fg=typer.colors.GREEN)
    else:
        typer.echo(content)


def generate_inference_script(output_dir: Path, model_name: str) -> None:
    """Generates a simple inference.py script in the output directory."""

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load the inference boilerplate from package resources
    template_file = importlib.resources.files("engram_peft").joinpath(
        "inference_template.py"
    )
    template = template_file.read_text()
    # Inject the model name into the placeholder
    script_content = template.replace("{{MODEL_NAME}}", model_name)

    (output_dir / "inference.py").write_text(script_content)


@app.command()
def train(
    config_path: Annotated[
        Path | None,
        typer.Option("--config", "-c", help="Path to the YAML configuration file."),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option("--model", "-m", help="Base model Name or Path."),
    ] = None,
    dataset_path: Annotated[
        str | None,
        typer.Option(
            "--dataset", "-d", help="Path to local dataset (json/parquet/csv)."
        ),
    ] = None,
    overrides: Annotated[
        list[str] | None,
        typer.Option(
            "--overrides",
            "-o",
            help="Override config values (e.g. training_args.learning_rate=1e-5).",
        ),
    ] = None,
) -> None:
    """Trains an Engram-augmented model."""
    config = {}
    if config_path:
        if not config_path.exists():
            typer.secho(
                f"Error: Configuration file not found at {config_path}",
                fg=typer.colors.RED,
            )
            raise typer.Exit(code=1)
        with open(config_path) as f:
            config = yaml.safe_load(f)
    elif model:
        config = {"model_name_or_path": model}
    else:
        typer.secho(
            "Error: Either --config or --model must be provided.", fg=typer.colors.RED
        )
        raise typer.Exit(code=1)

    if overrides:
        apply_overrides(config, overrides)

    model_name = config.get("model_name_or_path")
    if not model_name:
        typer.secho("Error: Missing 'model_name_or_path'.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    engram_dict = config.get("engram_config", {})
    lora_dict = config.get("lora_config")
    training_dict = config.get("training_args", {})
    data_dict = config.get("data_args", {})

    typer.echo(f"[*] Loading tokenizer & model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Build EngramConfig
    engram_config = EngramConfig(tokenizer_name_or_path=model_name, **engram_dict)

    # Explicitly type base_model to satisfy mypy
    base_model: nn.Module = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype="auto", device_map="auto"
    )

    # 1. Apply LoRA if configured
    train_mode = config.get("train_mode")
    if lora_dict:
        typer.echo("[*] Applying LoRA (PEFT) to backbone")
        lora_config = LoraConfig(**lora_dict)
        # get_peft_model expects PreTrainedModel, so we cast the AutoModel result
        # and then cast the PeftModel result to keep base_model consistently typed
        pm = cast("PreTrainedModel", base_model)
        base_model = cast("PreTrainedModel", get_peft_model(pm, lora_config))
        # Default to preserving LoRA trainable weights
        if train_mode is None:
            train_mode = "preserve_trainable"

    # 2. Inject Engram layers
    if train_mode is None:
        train_mode = "engram_only"

    # Ensure train_mode is valid TrainMode literal
    resolved_train_mode: TrainMode = "engram_only"
    if train_mode == "preserve_trainable":
        resolved_train_mode = "preserve_trainable"
    elif train_mode == "full_finetune":
        resolved_train_mode = "full_finetune"

    typer.echo(
        f"[*] Wrapping model with Engram layers (train_mode={resolved_train_mode})"
    )
    engram_model = get_engram_model(
        base_model,
        engram_config,
        tokenizer=tokenizer,
        train_mode=resolved_train_mode,
    )

    # Dataset Loading
    ds_name = data_dict.get("dataset_name")
    if dataset_path:
        typer.echo(f"[*] Loading local dataset: {dataset_path}")
        ext = Path(dataset_path).suffix[1:]
        if ext == "jsonl":
            ext = "json"
        dataset = load_dataset(ext, data_files=dataset_path, split="train")
    elif ds_name:
        typer.echo(f"[*] Loading Hugging Face dataset: {ds_name}")
        dataset = load_dataset(
            ds_name,
            data_dict.get("dataset_config_name"),
            split=data_dict.get("split", "train"),
        )
    else:
        typer.secho("Error: No dataset specified.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    def tokenize_function(examples: dict[str, Any]) -> dict[str, Any]:
        text_column = data_dict.get("text_column", "text")
        tokenized = tokenizer(
            examples[text_column],
            truncation=True,
            max_length=data_dict.get("max_length", 512),
            padding=False,
        )
        assert isinstance(tokenized, dict)
        return tokenized

    typer.echo("[*] Tokenizing dataset")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
    )

    data_collator = EngramDataCollator(
        tokenizer=tokenizer, config=engram_config, mlm=False
    )

    output_dir_default = (
        f"./outputs/{model_name.split('/')[-1]}-engram" if model_name else "./outputs"
    )
    output_dir = Path(training_dict.pop("output_dir", output_dir_default))

    # Apply demo defaults if not specified
    training_dict.setdefault("max_steps", 100)
    training_dict.setdefault("save_strategy", "no")
    training_dict.setdefault("logging_steps", 10)

    training_args = TrainingArguments(output_dir=str(output_dir), **training_dict)

    trainer = EngramTrainer(
        model=engram_model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    typer.echo("[*] Starting training...")
    trainer.train()

    typer.echo(f"[*] Saving adapter to {output_dir}")
    engram_model.save_pretrained(str(output_dir))

    typer.echo("[*] Generating inference script...")
    generate_inference_script(output_dir, model_name)

    typer.secho("Success: Training complete.", fg=typer.colors.GREEN, bold=True)


@app.command()
def evaluate() -> None:
    """[Placeholder] Evaluates an Engram-augmented model on a test set."""
    typer.echo("Evaluate functionality coming soon...")


if __name__ == "__main__":
    app()
