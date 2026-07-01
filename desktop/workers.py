from __future__ import annotations

import logging
import time
from dataclasses import dataclass
import traceback
from collections.abc import Callable
from typing import Any

try:
    from PySide6.QtCore import QObject, QRunnable, Signal, Slot
except ImportError as exc:  # pragma: no cover - exercised only when launching the UI.
    raise RuntimeError("Install desktop dependencies with `py -m pip install -e .[desktop]`.") from exc


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WorkerResult:
    label: str
    result: object
    elapsed_seconds: float


class WorkerSignals(QObject):
    started = Signal(str)
    finished = Signal(object)
    error = Signal(str)


class FunctionWorker(QRunnable):
    def __init__(self, label: str, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.label = label
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    @Slot()
    def run(self) -> None:
        started_at = time.perf_counter()
        self.signals.started.emit(self.label)
        logger.info("worker started: %s", self.label)
        try:
            result = self.fn(*self.args, **self.kwargs)
        except Exception:
            logger.exception("worker failed: %s", self.label)
            self.signals.error.emit(traceback.format_exc())
            return
        elapsed = time.perf_counter() - started_at
        logger.info("worker finished: %s in %.2fs", self.label, elapsed)
        self.signals.finished.emit(WorkerResult(self.label, result, elapsed))
