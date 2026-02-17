from trl import SFTTrainer
from transformers import TrainingArguments
import torch
import os
import yaml
from typing import Dict, Any, Optional


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
        t_cfg = self.config["training"]

        # Auto-detect BF16 support (Ampere+) vs FP16 (T4/Volta)
        bf16_supported = torch.cuda.is_bf16_supported()

        eval_steps = t_cfg.get("eval_steps", 50)
        save_steps = t_cfg.get("save_steps", 50)

        args = TrainingArguments(
            per_device_train_batch_size=t_cfg["per_device_train_batch_size"],
            gradient_accumulation_steps=t_cfg["gradient_accumulation_steps"],
            warmup_steps=t_cfg["warmup_steps"],
            max_steps=t_cfg["max_steps"],
            learning_rate=t_cfg["learning_rate"],
            fp16=not bf16_supported,
            bf16=bf16_supported,
            logging_steps=t_cfg["logging_steps"],
            eval_steps=eval_steps,
            evaluation_strategy="steps" if self.eval_dataset else "no",
            save_steps=save_steps,
            save_strategy="steps",
            save_total_limit=t_cfg.get("save_total_limit", 3),
            load_best_model_at_end=True if self.eval_dataset else False,
            optim=t_cfg["optim"],
            weight_decay=t_cfg["weight_decay"],
            lr_scheduler_type=t_cfg["lr_scheduler_type"],
            seed=self.config["project"]["seed"],
            output_dir=self.config["project"]["output_dir"],
            report_to="none",
        )

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

        trainer.train()
        self._save_artifacts()

    def _save_artifacts(self):
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
        with open(os.path.join(save_path, "experiment_config.yaml"), "w") as f:
            yaml.dump(self.config, f)
