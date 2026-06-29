from __future__ import annotations

import traceback
from collections.abc import Callable
from typing import Any

try:
    from PySide6.QtCore import QObject, QRunnable, Signal, Slot
except ImportError as exc:  # pragma: no cover - exercised only when launching the UI.
    raise RuntimeError("Install desktop dependencies with `py -m pip install -e .[desktop]`.") from exc


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
        self.signals.started.emit(self.label)
        try:
            result = self.fn(*self.args, **self.kwargs)
        except Exception:
            self.signals.error.emit(traceback.format_exc())
            return
        self.signals.finished.emit(result)
