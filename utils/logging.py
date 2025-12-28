from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


class JsonlLogger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, obj: Dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
