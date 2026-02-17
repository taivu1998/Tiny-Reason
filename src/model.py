from unsloth import FastLanguageModel
from typing import Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)


class ModelLoader:
    """Wrapper for Unsloth FastLanguageModel loading and configuration."""

    @staticmethod
    def load(config: Dict[str, Any]) -> Tuple[Any, Any]:
        """Loads the PreTrained model and Tokenizer with QLoRA settings."""
        model_cfg = config["model"]
        lora_cfg = config["lora"]

        logger.info(f"Loading model: {model_cfg['name']}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_cfg["name"],
            max_seq_length=model_cfg["max_seq_length"],
            dtype=None,  # Auto-detect
            load_in_4bit=model_cfg["load_in_4bit"],
        )

        logger.info(f"Applying LoRA (Rank {lora_cfg['r']})...")
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_cfg["r"],
            target_modules=lora_cfg["target_modules"],
            lora_alpha=lora_cfg["lora_alpha"],
            lora_dropout=lora_cfg["lora_dropout"],
            bias=lora_cfg["bias"],
            use_gradient_checkpointing="unsloth",
            random_state=config["project"]["seed"],
        )
        return model, tokenizer

    @staticmethod
    def load_for_inference(
        checkpoint_path: str, max_seq_length: int = 2048
    ) -> Tuple[Any, Any]:
        """Loads a model in inference mode (optimized)."""
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=checkpoint_path,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(model)
        return model, tokenizer
