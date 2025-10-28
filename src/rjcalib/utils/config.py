from __future__ import annotations
import json, pathlib, os

def load_config(path: str) -> dict:
    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(f"config file not found: {path}")
    txt = p.read_text().strip()
    # JSON first; YAML optional if you want later
    return json.loads(txt)
