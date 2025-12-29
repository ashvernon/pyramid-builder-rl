from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


_default_logger: Optional["JsonlLogger"] = None


class JsonlLogger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("a", encoding="utf-8")

        global _default_logger
        _default_logger = self

    def log(self, obj: Dict[str, Any]) -> None:
        self._file.write(json.dumps(obj, ensure_ascii=False) + "\n")

    def flush(self) -> None:
        self._file.flush()

    def close(self) -> None:
        self._file.close()


def log_event(event_type: str, payload: Dict[str, Any]) -> None:
    if _default_logger is None:
        raise RuntimeError("No default logger configured; instantiate JsonlLogger first")

    _default_logger.log({"type": event_type, **payload})


def flush() -> None:
    if _default_logger is not None:
        _default_logger.flush()
