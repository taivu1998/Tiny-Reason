import copy
import os
import unittest

from src.config_parser import load_yaml, parse_args
from src.config_validation import ConfigValidationError, validate_config
from src.trainer import TinyTrainer


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_CONFIG_PATH = os.path.join(REPO_ROOT, "configs", "default.yaml")


class ConfigValidationTests(unittest.TestCase):
    def test_parse_args_injects_project_seed_into_data_config(self):
        config = parse_args(
            [
                "--config",
                DEFAULT_CONFIG_PATH,
                "--seed",
                "123",
                "--eval_steps",
                "50",
                "--save_steps",
                "100",
            ]
        )

        self.assertEqual(config["project"]["seed"], 123)
        self.assertEqual(config["data"]["seed"], 123)
        self.assertTrue(config["training"]["load_best_model_at_end"])

    def test_invalid_eval_and_save_schedule_fails_validation(self):
        with self.assertRaises(ConfigValidationError):
            parse_args(
                [
                    "--config",
                    DEFAULT_CONFIG_PATH,
                    "--eval_steps",
                    "50",
                    "--save_steps",
                    "75",
                ]
            )

    def test_schedule_is_allowed_without_validation_split(self):
        config = load_yaml(DEFAULT_CONFIG_PATH)
        config["data"]["validation_split"] = 0.0
        config["training"]["eval_steps"] = 50
        config["training"]["save_steps"] = 75

        validated = validate_config(config)
        self.assertFalse(validated["training"]["load_best_model_at_end"])


class TrainerArgumentTests(unittest.TestCase):
    def test_training_argument_builder_rejects_invalid_schedule(self):
        config = load_yaml(DEFAULT_CONFIG_PATH)
        config["training"]["eval_steps"] = 50
        config["training"]["save_steps"] = 75

        with self.assertRaises(ValueError):
            TinyTrainer.build_training_arguments(config, has_eval_dataset=True)

    def test_training_argument_builder_uses_expected_intervals(self):
        config = copy.deepcopy(load_yaml(DEFAULT_CONFIG_PATH))
        validated = validate_config(config)

        args = TinyTrainer.build_training_arguments(validated, has_eval_dataset=True)

        self.assertEqual(args.eval_steps, 50)
        self.assertEqual(args.save_steps, 50)
        self.assertTrue(args.load_best_model_at_end)


if __name__ == "__main__":
    unittest.main()
