from trl import SFTTrainer
from transformers import TrainingArguments
import torch
import os
import json
import yaml
import logging
import inspect
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class TinyTrainer:
    """Manages the SFTTrainer lifecycle and artifact saving."""

    def __init__(
        self, model, tokenizer, dataset, config: Dict[str, Any], eval_dataset=None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.eval_dataset = eval_dataset
        self.config = config

    def train(self):
        args = self.build_training_arguments(self.config, self.eval_dataset is not None)

        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=self.dataset,
            eval_dataset=self.eval_dataset,
            dataset_text_field="text",
            max_seq_length=self.config["model"]["max_seq_length"],
            dataset_num_proc=self.config["data"].get("num_proc", 1),
            packing=False,
            args=args,
        )

        logger.info("Starting model fine-tuning...")
        train_result = trainer.train()
        trainer.save_state()
        self._save_artifacts(train_result.metrics)

    @staticmethod
    def build_training_arguments(
        config: Dict[str, Any], has_eval_dataset: bool
    ) -> TrainingArguments:
        """Builds validated Hugging Face training arguments for the current run."""
        t_cfg = config["training"]
        eval_steps = t_cfg.get("eval_steps", 50)
        save_steps = t_cfg.get("save_steps", 50)

        if has_eval_dataset and save_steps % eval_steps != 0:
            raise ValueError(
                "Invalid training schedule: save_steps must be a round multiple of "
                "eval_steps when load_best_model_at_end is enabled."
            )

        cuda_available = torch.cuda.is_available()
        bf16_supported = cuda_available and torch.cuda.is_bf16_supported()
        evaluation_strategy_key = (
            "eval_strategy"
            if "eval_strategy" in inspect.signature(TrainingArguments.__init__).parameters
            else "evaluation_strategy"
        )

        training_kwargs = {
            "per_device_train_batch_size": t_cfg["per_device_train_batch_size"],
            "gradient_accumulation_steps": t_cfg["gradient_accumulation_steps"],
            "warmup_steps": t_cfg["warmup_steps"],
            "max_steps": t_cfg["max_steps"],
            "learning_rate": t_cfg["learning_rate"],
            "fp16": cuda_available and not bf16_supported,
            "bf16": bf16_supported,
            "logging_steps": t_cfg["logging_steps"],
            "eval_steps": eval_steps,
            "save_steps": save_steps,
            "save_strategy": "steps",
            "save_total_limit": t_cfg.get("save_total_limit", 3),
            "load_best_model_at_end": has_eval_dataset,
            "metric_for_best_model": "eval_loss" if has_eval_dataset else None,
            "greater_is_better": False if has_eval_dataset else None,
            "optim": t_cfg["optim"],
            "weight_decay": t_cfg["weight_decay"],
            "lr_scheduler_type": t_cfg["lr_scheduler_type"],
            "seed": config["project"]["seed"],
            "output_dir": config["project"]["output_dir"],
            "report_to": "none",
        }
        training_kwargs[evaluation_strategy_key] = "steps" if has_eval_dataset else "no"
        return TrainingArguments(**training_kwargs)

    def _save_artifacts(self, metrics: Optional[Dict[str, Any]] = None):
        """Saves the model, tokenizer, and configuration for reproducibility."""
        output_dir = self.config["project"]["output_dir"]
        save_path = self.config["project"].get(
            "adapter_path", os.path.join(output_dir, "lora_adapter")
        )

        # Ensure output directories exist
        os.makedirs(save_path, exist_ok=True)

        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

        # Save exact config used
        with open(os.path.join(save_path, "experiment_config.yaml"), "w", encoding="utf-8") as f:
            yaml.dump(self.config, f)

        if metrics is not None:
            with open(os.path.join(save_path, "training_metrics.json"), "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2, ensure_ascii=True)
