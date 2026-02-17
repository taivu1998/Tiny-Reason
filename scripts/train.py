import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config_parser import parse_args
from src.utils import setup_logging, seed_everything
from src.dataset import GSM8KProcessor
from src.model import ModelLoader
from src.trainer import TinyTrainer


def main():
    config = parse_args()
    logger = setup_logging()
    seed_everything(config["project"]["seed"])

    logger.info("Starting TinyReason Training Pipeline...")

    # Data
    logger.info("Initializing Data Pipeline...")
    processor = GSM8KProcessor(config["data"])
    train_dataset, val_dataset = processor.load_train_and_validation()

    # Model
    logger.info("Loading Unsloth Model...")
    model, tokenizer = ModelLoader.load(config)

    # Train
    logger.info("Starting Trainer...")
    trainer = TinyTrainer(
        model, tokenizer, train_dataset, config, eval_dataset=val_dataset
    )
    trainer.train()

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
