from typing import Tuple, Dict, Any
import logging
import torch

logger = logging.getLogger(__name__)


def _get_fast_language_model():
    try:
        from unsloth import FastLanguageModel
    except ImportError as exc:
        raise ImportError(
            "Unsloth is required for model loading. Install the project dependencies "
            "with `make install` or install `unsloth` manually."
        ) from exc
    return FastLanguageModel


class ModelLoader:
    """Wrapper for Unsloth FastLanguageModel loading and configuration."""

    @staticmethod
    def load(config: Dict[str, Any]) -> Tuple[Any, Any]:
        """Loads the PreTrained model and Tokenizer with QLoRA settings."""
        model_cfg = config["model"]
        lora_cfg = config["lora"]
        fast_language_model = _get_fast_language_model()

        logger.info(f"Loading model: {model_cfg['name']}")
        model, tokenizer = fast_language_model.from_pretrained(
            model_name=model_cfg["name"],
            max_seq_length=model_cfg["max_seq_length"],
            dtype=None,  # Auto-detect
            load_in_4bit=model_cfg["load_in_4bit"],
        )

        logger.info(f"Applying LoRA (Rank {lora_cfg['r']})...")
        model = fast_language_model.get_peft_model(
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
        checkpoint_path: str,
        max_seq_length: int = 2048,
        load_in_4bit: bool = True,
        device: torch.device | None = None,
    ) -> Tuple[Any, Any]:
        """Loads a model in inference mode (optimized)."""
        if load_in_4bit:
            fast_language_model = _get_fast_language_model()
            model, tokenizer = fast_language_model.from_pretrained(
                model_name=checkpoint_path,
                max_seq_length=max_seq_length,
                dtype=None,
                load_in_4bit=load_in_4bit,
            )
            fast_language_model.for_inference(model)
            return model, tokenizer

        from peft import PeftConfig, PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        peft_config = PeftConfig.from_pretrained(checkpoint_path)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

        torch_dtype = None
        if device is not None and device.type == "cuda":
            torch_dtype = torch.float16

        logger.info(
            "Loading non-quantized PEFT inference model from adapter '%s' with base '%s'.",
            checkpoint_path,
            peft_config.base_model_name_or_path,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path,
            torch_dtype=torch_dtype,
        )
        model = PeftModel.from_pretrained(base_model, checkpoint_path)
        if device is not None:
            model = model.to(device)
        model.eval()
        return model, tokenizer
