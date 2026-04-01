"""TinyReason Source Package."""

from src.config_parser import parse_args, load_yaml
from src.utils import setup_logging, seed_everything

__all__ = [
    "parse_args",
    "load_yaml",
    "GSM8KProcessor",
    "ModelLoader",
    "TinyTrainer",
    "setup_logging",
    "seed_everything",
]

_LAZY_IMPORTS = {
    "GSM8KProcessor": "src.dataset",
    "ModelLoader": "src.model",
    "TinyTrainer": "src.trainer",
}


def __getattr__(name):
    if name in _LAZY_IMPORTS:
        module = __import__(_LAZY_IMPORTS[name], fromlist=[name])
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'src' has no attribute {name!r}")
