# src/dingus/__init__.py
__version__ = "0.0.1"

from .config import CaseCnfg
from .cli import app

__all__ = ["__version__", "CaseCfg", "app"]