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
        malformed = []

        for input_text, output_text in zip(inputs, outputs):
            # GSM8K delimiter: "Reasoning... #### Answer"
            if "####" in output_text:
                parts = output_text.split("####", 1)
                reasoning = parts[0].strip()
                final_ans = parts[1].strip()
            else:
                # Skip malformed data but maintain batch alignment with empty string
                # These will be filtered out after mapping
                texts.append("")
                malformed.append(True)
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
            malformed.append(False)

        return {"text": texts, "is_malformed": malformed}

    def create_prompt(self, question: str, system_prompt: str = None) -> str:
        """Creates a prompt for inference/evaluation."""
        if system_prompt is None:
            system_prompt = self.config.get(
                "system_prompt", "You are a helpful assistant."
            )
        return f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n<think>\n"

    def _create_prompt(self, question: str, system_prompt: str = None) -> str:
        """
        Backwards-compatible alias for older callers.

        Prefer `create_prompt()` in new code.
        """
        return self.create_prompt(question, system_prompt)

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
            requested_samples = self.config["num_samples"]
            actual_samples = min(requested_samples, len(ds))
            if actual_samples < requested_samples:
                logger.warning(
                    "Requested %s samples but dataset only has %s rows. Using all available rows.",
                    requested_samples,
                    actual_samples,
                )
            logger.info(f"Subsampling to {actual_samples} samples.")
            ds = ds.select(range(actual_samples))

        num_proc = self.config.get("num_proc", 1)
        map_kwargs = {"batched": True}
        if num_proc and num_proc > 1:
            map_kwargs["num_proc"] = num_proc
        ds = ds.map(self._format_function, **map_kwargs)
        malformed_count = sum(ds["is_malformed"]) if len(ds) > 0 else 0
        if malformed_count:
            logger.warning("Filtered %s malformed GSM8K rows without a '####' delimiter.", malformed_count)
        ds = ds.filter(lambda x: not x["is_malformed"])
        ds = ds.remove_columns("is_malformed")
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
            requested_samples = self.config["test_samples"]
            actual_samples = min(requested_samples, len(ds))
            if actual_samples < requested_samples:
                logger.warning(
                    "Requested %s test samples but dataset only has %s rows. Using all available rows.",
                    requested_samples,
                    actual_samples,
                )
            ds = ds.select(range(actual_samples))
        return ds
