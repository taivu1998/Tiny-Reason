from datasets import load_dataset, Dataset
from typing import Dict, List, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class GSM8KProcessor:
    """Handles data loading and CoT formatting for the GSM8K dataset."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def _format_function(self, examples: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """Injects <think> tags into the ChatML format."""
        system_prompt = self.config.get("system_prompt", "You are a helpful assistant.")

        inputs = examples["question"]
        outputs = examples["answer"]
        texts = []

        for input_text, output_text in zip(inputs, outputs):
            # GSM8K delimiter: "Reasoning... #### Answer"
            if "####" in output_text:
                parts = output_text.split("####")
                reasoning = parts[0].strip()
                final_ans = parts[1].strip()
            else:
                # Skip malformed data but maintain batch alignment with empty string
                # These will be filtered out after mapping
                texts.append("")
                continue

            text = f"""<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{input_text}<|im_end|>
<|im_start|>assistant
<think>
{reasoning}

</think>

**Final Answer:** {final_ans}<|im_end|>"""
            texts.append(text)

        return {"text": texts}

    def _create_prompt(self, question: str, system_prompt: str = None) -> str:
        """Creates a prompt for inference/evaluation."""
        if system_prompt is None:
            system_prompt = self.config.get(
                "system_prompt", "You are a helpful assistant."
            )
        return f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n<think>\n"

    def load_and_process(self) -> Tuple[Dataset, Dataset]:
        """Loads and formats the training dataset with optional validation split."""
        logger.info(f"Loading {self.config['dataset_name']}...")
        ds = load_dataset(
            self.config["dataset_name"],
            self.config["subset"],
            split=self.config["split"],
        )

        if self.config.get("shuffle", True):
            logger.info("Shuffling dataset...")
            ds = ds.shuffle(seed=self.config.get("seed", 3407))

        if self.config.get("num_samples"):
            logger.info(f"Subsampling to {self.config['num_samples']} samples.")
            ds = ds.select(range(self.config["num_samples"]))

        ds = ds.map(
            self._format_function, batched=True, num_proc=self.config.get("num_proc", 1)
        )
        # Filter out empty strings from malformed data
        ds = ds.filter(lambda x: len(x["text"]) > 0)
        logger.info(f"Dataset size after filtering: {len(ds)}")

        # Create validation split if configured
        validation_split = self.config.get("validation_split", 0)
        if validation_split > 0:
            split = ds.train_test_split(
                test_size=validation_split, seed=self.config.get("seed", 3407)
            )
            train_ds = split["train"]
            val_ds = split["test"]
            logger.info(f"Train size: {len(train_ds)}, Validation size: {len(val_ds)}")
            return train_ds, val_ds

        return ds, None

    def load_train_and_validation(self) -> Tuple[Dataset, Dataset]:
        """Loads training and validation datasets separately."""
        train_ds, val_ds = self.load_and_process()
        return train_ds, val_ds

    def get_test_set(self) -> Dataset:
        """Loads the test split for evaluation."""
        ds = load_dataset(
            self.config["dataset_name"], self.config["subset"], split="test"
        )
        if self.config.get("test_samples"):
            ds = ds.select(range(self.config["test_samples"]))
        return ds
