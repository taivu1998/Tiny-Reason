import copy
import unittest
from unittest.mock import patch

from datasets import Dataset

from src.dataset import GSM8KProcessor


def make_dataset(num_rows=6):
    return Dataset.from_dict(
        {
            "question": [f"Question {idx}" for idx in range(num_rows)],
            "answer": [f"Reasoning step {idx} #### {idx}" for idx in range(num_rows)],
        }
    )


class DatasetProcessorTests(unittest.TestCase):
    def test_create_prompt_is_public_and_structured(self):
        processor = GSM8KProcessor({"system_prompt": "Solve carefully."})
        prompt = processor.create_prompt("What is 2 + 2?")

        self.assertIn("<|im_start|>system", prompt)
        self.assertIn("Solve carefully.", prompt)
        self.assertIn("What is 2 + 2?", prompt)
        self.assertTrue(prompt.endswith("<think>\n"))

    def test_load_and_process_filters_malformed_rows(self):
        dataset = Dataset.from_dict(
            {
                "question": ["good", "bad"],
                "answer": ["reasoning #### 42", "missing delimiter"],
            }
        )
        processor = GSM8KProcessor(
            {
                "dataset_name": "unused",
                "subset": "main",
                "split": "train",
                "seed": 7,
                "shuffle": False,
                "validation_split": 0.0,
                "num_proc": 1,
            }
        )

        with patch("src.dataset.load_dataset", return_value=dataset):
            train_ds, val_ds = processor.load_and_process()

        self.assertIsNone(val_ds)
        self.assertEqual(len(train_ds), 1)
        self.assertEqual(train_ds[0]["question"], "good")
        self.assertIn("**Final Answer:** 42", train_ds[0]["text"])

    def test_seed_changes_dataset_split_order(self):
        base_config = {
            "dataset_name": "unused",
            "subset": "main",
            "split": "train",
            "shuffle": True,
            "validation_split": 0.2,
            "num_proc": 1,
        }
        dataset = make_dataset(num_rows=12)

        config_a = copy.deepcopy(base_config)
        config_a["seed"] = 1
        config_b = copy.deepcopy(base_config)
        config_b["seed"] = 2

        with patch("src.dataset.load_dataset", return_value=dataset):
            train_a, _ = GSM8KProcessor(config_a).load_and_process()
        with patch("src.dataset.load_dataset", return_value=dataset):
            train_b, _ = GSM8KProcessor(config_b).load_and_process()

        self.assertNotEqual(train_a["question"], train_b["question"])


if __name__ == "__main__":
    unittest.main()
