# src/dingus/config.py
from pydantic import BaseModel
class CaseCfg(BaseModel):
    meta: dict = {}