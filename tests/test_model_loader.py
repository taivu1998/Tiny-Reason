import types
import unittest
from unittest.mock import MagicMock, patch

import torch

from src.model import ModelLoader


class ModelLoaderInferenceTests(unittest.TestCase):
    @patch("transformers.AutoTokenizer.from_pretrained")
    @patch("transformers.AutoModelForCausalLM.from_pretrained")
    @patch("peft.PeftModel.from_pretrained")
    @patch("peft.PeftConfig.from_pretrained")
    def test_load_for_inference_uses_peft_fallback_when_not_quantized(
        self,
        peft_config_from_pretrained,
        peft_model_from_pretrained,
        auto_model_from_pretrained,
        auto_tokenizer_from_pretrained,
    ):
        peft_config_from_pretrained.return_value = types.SimpleNamespace(
            base_model_name_or_path="base-model"
        )
        tokenizer = MagicMock()
        auto_tokenizer_from_pretrained.return_value = tokenizer
        base_model = MagicMock()
        auto_model_from_pretrained.return_value = base_model
        peft_model = MagicMock()
        peft_model.to.return_value = peft_model
        peft_model_from_pretrained.return_value = peft_model

        model, loaded_tokenizer = ModelLoader.load_for_inference(
            "adapter-path",
            max_seq_length=2048,
            load_in_4bit=False,
            device=torch.device("cpu"),
        )

        peft_config_from_pretrained.assert_called_once_with("adapter-path")
        auto_tokenizer_from_pretrained.assert_called_once_with("adapter-path")
        auto_model_from_pretrained.assert_called_once()
        peft_model_from_pretrained.assert_called_once_with(base_model, "adapter-path")
        peft_model.to.assert_called_once_with(torch.device("cpu"))
        peft_model.eval.assert_called_once()
        self.assertIs(model, peft_model)
        self.assertIs(loaded_tokenizer, tokenizer)


if __name__ == "__main__":
    unittest.main()
