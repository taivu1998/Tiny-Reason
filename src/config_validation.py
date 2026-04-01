from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict


class ConfigValidationError(ValueError):
    """Raised when the merged experiment configuration is invalid."""


def _require_mapping(config: Dict[str, Any], key: str) -> Dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ConfigValidationError(f"Missing or invalid '{key}' configuration section.")
    return value


def _ensure_non_empty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigValidationError(f"'{name}' must be a non-empty string.")
    return value


def _ensure_int(value: Any, name: str, *, minimum: int | None = None) -> int:
    if not isinstance(value, int):
        raise ConfigValidationError(f"'{name}' must be an integer.")
    if minimum is not None and value < minimum:
        raise ConfigValidationError(f"'{name}' must be >= {minimum}.")
    return value


def _ensure_number(
    value: Any, name: str, *, minimum: float | None = None, allow_equal: bool = True
) -> float:
    if not isinstance(value, (int, float)):
        raise ConfigValidationError(f"'{name}' must be a number.")

    numeric_value = float(value)
    if minimum is not None:
        if allow_equal and numeric_value < minimum:
            raise ConfigValidationError(f"'{name}' must be >= {minimum}.")
        if not allow_equal and numeric_value <= minimum:
            raise ConfigValidationError(f"'{name}' must be > {minimum}.")
    return numeric_value


def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validates and normalizes the merged experiment config.

    The returned config is a deep copy so downstream code can rely on the
    normalized values without mutating the caller's dictionary.
    """

    validated = deepcopy(config)

    project_cfg = _require_mapping(validated, "project")
    model_cfg = _require_mapping(validated, "model")
    lora_cfg = _require_mapping(validated, "lora")
    data_cfg = _require_mapping(validated, "data")
    training_cfg = _require_mapping(validated, "training")

    seed = _ensure_int(project_cfg.get("seed"), "project.seed")
    _ensure_non_empty_string(project_cfg.get("name"), "project.name")
    _ensure_non_empty_string(project_cfg.get("output_dir"), "project.output_dir")
    _ensure_non_empty_string(project_cfg.get("adapter_path"), "project.adapter_path")

    _ensure_non_empty_string(model_cfg.get("name"), "model.name")
    _ensure_int(model_cfg.get("max_seq_length"), "model.max_seq_length", minimum=1)
    if not isinstance(model_cfg.get("load_in_4bit"), bool):
        raise ConfigValidationError("'model.load_in_4bit' must be a boolean.")

    _ensure_int(lora_cfg.get("r"), "lora.r", minimum=1)
    _ensure_number(lora_cfg.get("lora_alpha"), "lora.lora_alpha", minimum=0, allow_equal=False)
    _ensure_number(lora_cfg.get("lora_dropout"), "lora.lora_dropout", minimum=0)
    _ensure_non_empty_string(lora_cfg.get("bias"), "lora.bias")

    target_modules = lora_cfg.get("target_modules")
    if (
        not isinstance(target_modules, list)
        or not target_modules
        or not all(isinstance(module_name, str) and module_name.strip() for module_name in target_modules)
    ):
        raise ConfigValidationError("'lora.target_modules' must be a non-empty list of strings.")

    _ensure_non_empty_string(data_cfg.get("dataset_name"), "data.dataset_name")
    _ensure_non_empty_string(data_cfg.get("subset"), "data.subset")
    _ensure_non_empty_string(data_cfg.get("split"), "data.split")
    data_cfg["seed"] = seed

    if data_cfg.get("num_samples") is not None:
        _ensure_int(data_cfg["num_samples"], "data.num_samples", minimum=1)
    if data_cfg.get("test_samples") is not None:
        _ensure_int(data_cfg["test_samples"], "data.test_samples", minimum=1)
    if data_cfg.get("num_proc") is not None:
        _ensure_int(data_cfg["num_proc"], "data.num_proc", minimum=1)

    validation_split = _ensure_number(data_cfg.get("validation_split", 0), "data.validation_split", minimum=0)
    if validation_split >= 1:
        raise ConfigValidationError("'data.validation_split' must be < 1.")

    _ensure_int(
        training_cfg.get("per_device_train_batch_size"),
        "training.per_device_train_batch_size",
        minimum=1,
    )
    _ensure_int(
        training_cfg.get("gradient_accumulation_steps"),
        "training.gradient_accumulation_steps",
        minimum=1,
    )
    _ensure_int(training_cfg.get("warmup_steps"), "training.warmup_steps", minimum=0)
    _ensure_int(training_cfg.get("max_steps"), "training.max_steps", minimum=1)
    _ensure_number(
        training_cfg.get("learning_rate"),
        "training.learning_rate",
        minimum=0,
        allow_equal=False,
    )
    _ensure_int(training_cfg.get("logging_steps"), "training.logging_steps", minimum=1)
    _ensure_int(training_cfg.get("eval_steps"), "training.eval_steps", minimum=1)
    _ensure_int(training_cfg.get("save_steps"), "training.save_steps", minimum=1)
    _ensure_int(training_cfg.get("save_total_limit"), "training.save_total_limit", minimum=1)
    _ensure_non_empty_string(training_cfg.get("optim"), "training.optim")
    _ensure_non_empty_string(training_cfg.get("lr_scheduler_type"), "training.lr_scheduler_type")
    _ensure_number(training_cfg.get("weight_decay"), "training.weight_decay", minimum=0)

    has_validation = validation_split > 0
    training_cfg["load_best_model_at_end"] = has_validation

    if has_validation and training_cfg["save_steps"] % training_cfg["eval_steps"] != 0:
        raise ConfigValidationError(
            "Invalid training schedule: 'training.save_steps' must be a round "
            "multiple of 'training.eval_steps' when validation is enabled."
        )

    return validated
