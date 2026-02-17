import os
import random
import numpy as np
import torch
import logging

def setup_logging(name: str = "TinyReason") -> logging.Logger:
    """Configures and returns a standardized logger."""
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    return logging.getLogger(name)

def seed_everything(seed: int):
    """Sets the seed for reproducibility across torch, numpy, and python."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)