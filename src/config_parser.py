import yaml
import argparse
from typing import Dict, Any


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def parse_args() -> Dict[str, Any]:
    """
    Parses CLI arguments and merges them with the YAML config.
    CLI arguments (like --learning_rate) override YAML values.
    """
    parser = argparse.ArgumentParser(description="TinyReason Experiment Runner")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")

    # Project overrides
    parser.add_argument("--seed", type=int, help="Override random seed")
    parser.add_argument("--output_dir", type=str, help="Override output directory")
    parser.add_argument("--adapter_path", type=str, help="Override adapter path")

    # Training overrides
    parser.add_argument("--learning_rate", type=float, help="Override LR")
    parser.add_argument("--max_steps", type=int, help="Override max training steps")
    parser.add_argument("--batch_size", type=int, help="Override per-device batch size")
    parser.add_argument(
        "--gradient_accumulation_steps", type=int, help="Override gradient accumulation"
    )
    parser.add_argument("--warmup_steps", type=int, help="Override warmup steps")
    parser.add_argument("--eval_steps", type=int, help="Override eval steps")
    parser.add_argument("--save_steps", type=int, help="Override save steps")

    # Data overrides
    parser.add_argument(
        "--num_samples", type=int, help="Override training sample count"
    )
    parser.add_argument("--test_samples", type=int, help="Override test sample count")
    parser.add_argument(
        "--validation_split", type=float, help="Override validation split (0.0-1.0)"
    )

    # LoRA overrides
    parser.add_argument("--lora_r", type=int, help="Override LoRA rank")
    parser.add_argument("--lora_alpha", type=float, help="Override LoRA alpha")

    args = parser.parse_args()
    config = load_yaml(args.config)

    # Merge project overrides
    if args.seed is not None:
        config["project"]["seed"] = args.seed
    if args.output_dir is not None:
        config["project"]["output_dir"] = args.output_dir
    if args.adapter_path is not None:
        config["project"]["adapter_path"] = args.adapter_path

    # Merge training overrides
    if args.learning_rate is not None:
        config["training"]["learning_rate"] = args.learning_rate
    if args.max_steps is not None:
        config["training"]["max_steps"] = args.max_steps
    if args.batch_size is not None:
        config["training"]["per_device_train_batch_size"] = args.batch_size
    if args.gradient_accumulation_steps is not None:
        config["training"]["gradient_accumulation_steps"] = (
            args.gradient_accumulation_steps
        )
    if args.warmup_steps is not None:
        config["training"]["warmup_steps"] = args.warmup_steps
    if args.eval_steps is not None:
        config["training"]["eval_steps"] = args.eval_steps
    if args.save_steps is not None:
        config["training"]["save_steps"] = args.save_steps

    # Merge data overrides
    if args.num_samples is not None:
        config["data"]["num_samples"] = args.num_samples
    if args.test_samples is not None:
        config["data"]["test_samples"] = args.test_samples
    if args.validation_split is not None:
        config["data"]["validation_split"] = args.validation_split

    # Merge LoRA overrides
    if args.lora_r is not None:
        config["lora"]["r"] = args.lora_r
    if args.lora_alpha is not None:
        config["lora"]["lora_alpha"] = args.lora_alpha

    return config
