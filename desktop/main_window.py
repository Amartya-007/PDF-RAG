from __future__ import annotations

from pathlib import Path

try:
    from PySide6.QtCore import QThreadPool, Qt
    from PySide6.QtWidgets import (
        QApplication,
        QDialog,
        QFileDialog,
        QFormLayout,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QListWidget,
        QMainWindow,
        QMessageBox,
        QPlainTextEdit,
        QProgressBar,
        QPushButton,
        QSplitter,
        QTextBrowser,
        QVBoxLayout,
        QWidget,
    )
except ImportError as exc:  # pragma: no cover - exercised only when launching the UI.
    raise RuntimeError("Install desktop dependencies with `py -m pip install -e .[desktop]`.") from exc

from backend.app.models import Answer, Document
from desktop.controller import DesktopController
from desktop.workers import FunctionWorker


class MainWindow(QMainWindow):
    def __init__(self, controller: DesktopController) -> None:
        super().__init__()
        self.controller = controller
        self.thread_pool = QThreadPool.globalInstance()
        self.setWindowTitle("Local PDF RAG")
        self._build_ui()
        self.refresh_documents()
        self.refresh_status()

    def _build_ui(self) -> None:
        root = QWidget()
        root_layout = QVBoxLayout(root)

        toolbar = QHBoxLayout()
        self.import_button = QPushButton("Import PDFs / Text")
        self.import_button.clicked.connect(self.import_documents)
        self.settings_button = QPushButton("Settings")
        self.settings_button.clicked.connect(self.show_settings)
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.refresh_documents)
        toolbar.addWidget(self.import_button)
        toolbar.addWidget(self.refresh_button)
        toolbar.addStretch(1)
        toolbar.addWidget(self.settings_button)
        root_layout.addLayout(toolbar)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.addWidget(QLabel("Documents"))
        self.document_list = QListWidget()
        left_layout.addWidget(self.document_list)
        splitter.addWidget(left_panel)

        center_panel = QWidget()
        center_layout = QVBoxLayout(center_panel)
        center_layout.addWidget(QLabel("Chat"))
        self.chat_view = QTextBrowser()
        self.chat_view.setOpenExternalLinks(False)
        center_layout.addWidget(self.chat_view)
        question_row = QHBoxLayout()
        self.question_input = QLineEdit()
        self.question_input.setPlaceholderText("Ask a question about your local documents...")
        self.question_input.returnPressed.connect(self.ask_question)
        self.ask_button = QPushButton("Ask")
        self.ask_button.clicked.connect(self.ask_question)
        question_row.addWidget(self.question_input)
        question_row.addWidget(self.ask_button)
        center_layout.addLayout(question_row)
        splitter.addWidget(center_panel)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.addWidget(QLabel("Citations and Source Excerpts"))
        self.citation_view = QPlainTextEdit()
        self.citation_view.setReadOnly(True)
        right_layout.addWidget(self.citation_view)
        splitter.addWidget(right_panel)

        splitter.setSizes([260, 620, 400])
        root_layout.addWidget(splitter)

        status_row = QHBoxLayout()
        self.status_label = QLabel("Ready")
        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(1)
        status_row.addWidget(self.status_label, 1)
        status_row.addWidget(self.progress)
        root_layout.addLayout(status_row)

        self.setCentralWidget(root)

    def refresh_documents(self) -> None:
        self.document_list.clear()
        for document in self.controller.list_documents():
            self.document_list.addItem(f"{document.filename}  [{document.status}]")
        self.refresh_status()

    def refresh_status(self) -> None:
        status = self.controller.status()
        self.status_label.setText(
            f"Documents: {status['documents']} | Chunks: {status['chunks']} | "
            f"Concepts: {status['concepts']} | {status['ollama_message']}"
        )

    def import_documents(self) -> None:
        paths, _filter = QFileDialog.getOpenFileNames(
            self,
            "Import PDFs or text files",
            str(Path.home()),
            "Documents (*.pdf *.txt *.md);;All files (*.*)",
        )
        for path in paths:
            self._run_worker(f"Ingesting {Path(path).name}...", self.controller.ingest, path)

    def ask_question(self) -> None:
        question = self.question_input.text().strip()
        if not question:
            return
        self.question_input.clear()
        self.chat_view.append(f"<b>You:</b> {question}")
        self._run_worker("Answering question...", self.controller.ask, question, False)

    def show_settings(self) -> None:
        readiness = self.controller.model_readiness()
        dialog = QDialog(self)
        dialog.setWindowTitle("Offline Settings")
        layout = QVBoxLayout(dialog)
        form = QFormLayout()
        form.addRow("Data directory", QLabel(str(self.controller.settings.data_dir)))
        form.addRow("Ollama URL", QLabel(self.controller.settings.ollama_base_url))
        form.addRow("Active model", QLabel(self.controller.settings.active_model))
        form.addRow("Embedding model", QLabel(self.controller.settings.embedding_model))
        form.addRow("Ollama required", QLabel("Yes" if readiness.ollama_required else "No"))
        form.addRow("Ollama reachable", QLabel("Yes" if readiness.ollama_reachable else "No"))
        form.addRow("Offline ready", QLabel("Yes" if readiness.ready else "No"))
        layout.addLayout(form)
        commands = QPlainTextEdit()
        commands.setReadOnly(True)
        commands.setPlainText("\n".join(readiness.setup_commands) or readiness.message)
        layout.addWidget(QLabel("Setup commands / status"))
        layout.addWidget(commands)
        close = QPushButton("Close")
        close.clicked.connect(dialog.accept)
        layout.addWidget(close)
        dialog.resize(620, 420)
        dialog.exec()

    def _run_worker(self, label: str, fn: object, *args: object) -> None:
        worker = FunctionWorker(label, fn, *args)
        worker.signals.started.connect(self._worker_started)
        worker.signals.finished.connect(self._worker_finished)
        worker.signals.error.connect(self._worker_error)
        self.thread_pool.start(worker)

    def _worker_started(self, label: str) -> None:
        self.status_label.setText(label)
        self.progress.setRange(0, 0)
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)

    def _worker_finished(self, result: object) -> None:
        self.progress.setRange(0, 1)
        self.progress.setValue(1)
        QApplication.restoreOverrideCursor()
        if isinstance(result, Answer):
            self._show_answer(result)
        elif isinstance(result, Document):
            self.chat_view.append(f"<i>Imported {result.filename}</i>")
            self.refresh_documents()
        else:
            self.refresh_status()

    def _worker_error(self, message: str) -> None:
        self.progress.setRange(0, 1)
        self.progress.setValue(1)
        QApplication.restoreOverrideCursor()
        self.status_label.setText("Error")
        QMessageBox.critical(self, "Local PDF RAG Error", message)

    def _show_answer(self, answer: Answer) -> None:
        self.chat_view.append(f"<b>Assistant:</b> {answer.answer}")
        citation_lines: list[str] = []
        for citation in answer.citations:
            pages = (
                str(citation.page_start)
                if citation.page_start == citation.page_end
                else f"{citation.page_start}-{citation.page_end}"
            )
            citation_lines.extend(
                [
                    f"{citation.source_id} | {citation.filename} | page {pages}",
                    f"Chunk: {citation.chunk_id}",
                    citation.excerpt,
                    "",
                ]
            )
        self.citation_view.setPlainText("\n".join(citation_lines))
        self.refresh_status()
