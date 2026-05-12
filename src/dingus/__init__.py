# src/dingus/__init__.py
__version__ = "0.0.1"

from dingus.config import CaseCfg
from .cli import app

__all__ = ["__version__", "CaseCfg", "app"]