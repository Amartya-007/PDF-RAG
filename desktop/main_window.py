from __future__ import annotations

from pathlib import Path

try:
    from PySide6.QtCore import QTimer, QThreadPool, Qt
    from PySide6.QtWidgets import (
        QApplication,
        QDialog,
        QCheckBox,
        QComboBox,
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
        QScrollArea,
        QSplitter,
        QVBoxLayout,
        QWidget,
    )
except ImportError as exc:  # pragma: no cover - exercised only when launching the UI.
    raise RuntimeError("Install desktop dependencies with `py -m pip install -e .[desktop]`.") from exc

from backend.app.models import Answer, Document
from desktop.controller import DesktopController
from desktop.workers import FunctionWorker, WorkerResult


class MainWindow(QMainWindow):
    def __init__(self, controller: DesktopController) -> None:
        super().__init__()
        self.controller = controller
        self.thread_pool = QThreadPool.globalInstance()
        self._refreshing_sessions = False
        self.setWindowTitle("Local PDF RAG")
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background: #1f1f1f;
                color: #f5f5f5;
                font-size: 10pt;
            }
            QPushButton {
                background: #2d2d2d;
                border: 1px solid #4a4a4a;
                border-radius: 6px;
                padding: 6px 10px;
            }
            QPushButton:hover { background: #3a3a3a; }
            QLineEdit, QListWidget, QPlainTextEdit {
                background: #292929;
                border: 1px solid #4a4a4a;
                border-radius: 6px;
                padding: 6px;
                selection-background-color: #0f6cbd;
            }
            QLabel { color: #f5f5f5; }
            QScrollArea {
                background: #232323;
                border: 1px solid #4a4a4a;
                border-radius: 8px;
            }
            QProgressBar {
                border: 0;
                background: transparent;
                max-height: 6px;
            }
            QProgressBar::chunk {
                background: #60cdff;
                border-radius: 3px;
            }
            """
        )
        self._build_ui()
        self.refresh_sessions()
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
        self.repair_button = QPushButton("Repair Stuck Imports")
        self.repair_button.clicked.connect(self.repair_stuck_imports)
        toolbar.addWidget(self.import_button)
        toolbar.addWidget(self.refresh_button)
        toolbar.addWidget(self.repair_button)
        toolbar.addStretch(1)
        toolbar.addWidget(self.settings_button)
        root_layout.addLayout(toolbar)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.addWidget(QLabel("Chats"))
        session_row = QHBoxLayout()
        self.session_combo = QComboBox()
        self.session_combo.currentIndexChanged.connect(self.change_session)
        self.new_session_button = QPushButton("New Chat")
        self.new_session_button.clicked.connect(self.new_chat)
        session_row.addWidget(self.session_combo, 1)
        session_row.addWidget(self.new_session_button)
        left_layout.addLayout(session_row)
        left_layout.addWidget(QLabel("Documents"))
        self.document_list = QListWidget()
        left_layout.addWidget(self.document_list)
        splitter.addWidget(left_panel)

        center_panel = QWidget()
        center_layout = QVBoxLayout(center_panel)
        center_layout.addWidget(QLabel("Chat"))
        self.chat_scroll = QScrollArea()
        self.chat_scroll.setWidgetResizable(True)
        self.chat_content = QWidget()
        self.chat_layout = QVBoxLayout(self.chat_content)
        self.chat_layout.setContentsMargins(12, 12, 12, 12)
        self.chat_layout.setSpacing(10)
        self.chat_layout.addStretch(1)
        self.chat_scroll.setWidget(self.chat_content)
        center_layout.addWidget(self.chat_scroll)
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

    def refresh_sessions(self) -> None:
        self._refreshing_sessions = True
        active_session_id = self.controller.active_session_id()
        self.session_combo.clear()
        for session in self.controller.list_sessions():
            self.session_combo.addItem(session.title, session.session_id)
        index = self.session_combo.findData(active_session_id)
        if index >= 0:
            self.session_combo.setCurrentIndex(index)
        elif self.session_combo.count():
            self.session_combo.setCurrentIndex(0)
            session_id = self.session_combo.currentData()
            if session_id:
                self.controller.set_active_session(str(session_id))
        self._refreshing_sessions = False

    def change_session(self) -> None:
        if self._refreshing_sessions:
            return
        session_id = self.session_combo.currentData()
        if not session_id:
            return
        self.controller.set_active_session(str(session_id))
        self._clear_chat()
        self.refresh_documents()
        self._append_system_message(f"Switched to {self.session_combo.currentText()}")

    def new_chat(self) -> None:
        session = self.controller.create_session()
        self.refresh_sessions()
        index = self.session_combo.findData(session.session_id)
        if index >= 0:
            self.session_combo.setCurrentIndex(index)
        self._clear_chat()
        self.refresh_documents()
        self._append_system_message(f"Started {session.title}")

    def refresh_documents(self) -> None:
        self.document_list.clear()
        for document in self.controller.list_documents():
            self.document_list.addItem(f"{document.filename}  [{document.status}]")
        self.refresh_status()

    def refresh_status(self) -> None:
        status = self.controller.status()
        chat_title = self.session_combo.currentText() if hasattr(self, "session_combo") else "Chat"
        self.status_label.setText(
            f"{chat_title} | Documents: {status['documents']} | Chunks: {status['chunks']} | "
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

    def repair_stuck_imports(self) -> None:
        self._run_worker("Repairing stuck imports...", self.controller.repair_unready_documents)

    def ask_question(self) -> None:
        question = self.question_input.text().strip()
        if not question:
            return
        status = self.controller.status()
        if status["documents"] and not status["chunks"]:
            QMessageBox.warning(
                self,
                "No Searchable Text Yet",
                "Your documents are listed, but none have searchable chunks yet.\n\n"
                "Click 'Repair Stuck Imports' or reimport the PDFs. Documents should show "
                "'ready' and the status bar should show Chunks greater than 0 before asking.",
            )
            return
        self.question_input.clear()
        self._append_chat_bubble("You", question, align="right")
        self._run_worker("Answering question...", self.controller.ask, question, False)

    def show_settings(self) -> None:
        readiness = self.controller.model_readiness()
        dialog = QDialog(self)
        dialog.setWindowTitle("Offline Settings")
        layout = QVBoxLayout(dialog)
        form = QFormLayout()
        use_ollama = QCheckBox("Use Ollama for local models")
        use_ollama.setChecked(self.controller.settings.use_ollama)
        ollama_url = QLineEdit(self.controller.settings.ollama_base_url)
        active_model = QComboBox()
        active_model.setEditable(True)
        embedding_model = QComboBox()
        embedding_model.setEditable(True)
        available_models = readiness.available_models
        if not available_models and readiness.ollama_reachable:
            try:
                available_models = self.controller.available_ollama_models()
            except OSError:
                available_models = []
        for model in available_models:
            active_model.addItem(model)
            embedding_model.addItem(model)
        self._set_combo_text(active_model, self.controller.settings.active_model)
        self._set_combo_text(embedding_model, self.controller.settings.embedding_model)
        form.addRow("Data directory", QLabel(str(self.controller.settings.data_dir)))
        form.addRow("Ollama enabled", use_ollama)
        form.addRow("Ollama URL", ollama_url)
        form.addRow("Active model", active_model)
        form.addRow("Embedding model", embedding_model)
        form.addRow("Ollama required", QLabel("Yes" if readiness.ollama_required else "No"))
        form.addRow("Ollama reachable", QLabel("Yes" if readiness.ollama_reachable else "No"))
        form.addRow("Offline ready", QLabel("Yes" if readiness.ready else "No"))
        layout.addLayout(form)
        commands = QPlainTextEdit()
        commands.setReadOnly(True)
        model_list = "\n".join(f"- {model}" for model in available_models)
        commands.setPlainText(
            "\n".join(readiness.setup_commands)
            or readiness.message
            + ("\n\nInstalled Ollama models:\n" + model_list if model_list else "")
        )
        layout.addWidget(QLabel("Setup commands / status"))
        layout.addWidget(commands)
        buttons = QHBoxLayout()
        refresh_models = QPushButton("Refresh Models")
        refresh_models.clicked.connect(
            lambda: self._refresh_model_combos(ollama_url.text().strip(), active_model, embedding_model)
        )
        save = QPushButton("Save Settings")
        save.clicked.connect(
            lambda: self._save_settings(
                dialog,
                use_ollama.isChecked(),
                ollama_url.text().strip(),
                active_model.currentText().strip(),
                embedding_model.currentText().strip(),
            )
        )
        close = QPushButton("Close")
        close.clicked.connect(dialog.reject)
        buttons.addWidget(refresh_models)
        buttons.addStretch(1)
        buttons.addWidget(save)
        buttons.addWidget(close)
        layout.addLayout(buttons)
        dialog.resize(620, 420)
        dialog.exec()

    def _set_combo_text(self, combo: QComboBox, text: str) -> None:
        index = combo.findText(text)
        if index >= 0:
            combo.setCurrentIndex(index)
        else:
            combo.addItem(text)
            combo.setCurrentText(text)

    def _refresh_model_combos(
        self,
        ollama_url: str,
        active_model: QComboBox,
        embedding_model: QComboBox,
    ) -> None:
        try:
            models = self.controller.available_ollama_models(ollama_url)
        except OSError as exc:
            QMessageBox.warning(self, "Ollama Models", f"Could not reach Ollama:\n{exc}")
            return
        current_active = active_model.currentText()
        current_embedding = embedding_model.currentText()
        active_model.clear()
        embedding_model.clear()
        for model in models:
            active_model.addItem(model)
            embedding_model.addItem(model)
        self._set_combo_text(active_model, current_active)
        self._set_combo_text(embedding_model, current_embedding)

    def _save_settings(
        self,
        dialog: QDialog,
        use_ollama: bool,
        ollama_url: str,
        active_model: str,
        embedding_model: str,
    ) -> None:
        if not active_model or not embedding_model:
            QMessageBox.warning(self, "Settings", "Choose both an active model and an embedding model.")
            return
        self.controller.update_preferences(
            use_ollama=use_ollama,
            ollama_base_url=ollama_url or "http://localhost:11434",
            active_model=active_model,
            embedding_model=embedding_model,
        )
        self.refresh_documents()
        dialog.accept()

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
        elapsed_seconds: float | None = None
        if isinstance(result, WorkerResult):
            elapsed_seconds = result.elapsed_seconds
            result = result.result
        self.progress.setRange(0, 1)
        self.progress.setValue(1)
        QApplication.restoreOverrideCursor()
        if isinstance(result, Answer):
            self._show_answer(result, elapsed_seconds)
            if elapsed_seconds is not None:
                self.status_label.setText(f"Answered in {self._format_elapsed(elapsed_seconds)}")
        elif isinstance(result, Document):
            suffix = f" in {self._format_elapsed(elapsed_seconds)}" if elapsed_seconds is not None else ""
            self._append_system_message(f"Imported {result.filename}{suffix}")
            self.refresh_documents()
        elif isinstance(result, list):
            suffix = f" in {self._format_elapsed(elapsed_seconds)}" if elapsed_seconds is not None else ""
            self._append_system_message(f"Repaired {len(result)} document(s){suffix}.")
            self.refresh_documents()
        else:
            self.refresh_status()

    def _clear_chat(self) -> None:
        while self.chat_layout.count() > 1:
            item = self.chat_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _worker_error(self, message: str) -> None:
        self.progress.setRange(0, 1)
        self.progress.setValue(1)
        QApplication.restoreOverrideCursor()
        self.status_label.setText("Error")
        QMessageBox.critical(self, "Local PDF RAG Error", self._friendly_error(message))

    def _friendly_error(self, message: str) -> str:
        if "PDF parsing requires PyMuPDF" in message:
            return (
                "PDF parsing is not available in this Python environment.\n\n"
                "Fix:\n"
                "  py -m pip install -e .[desktop]\n\n"
                "The desktop extra now includes PyMuPDF for local PDF parsing. "
                "Restart the app after installing.\n\n"
                f"{message}"
            )
        if "no searchable text was extracted" in message:
            return (
                "The PDF parser opened this file, but did not find searchable text.\n\n"
                "If this PDF is scanned or image-only, install OCR support:\n"
                "  py -m pip install -e .[pdf]\n\n"
                "Then restart the app and click Repair Stuck Imports."
            )
        return message

    def _show_answer(self, answer: Answer, elapsed_seconds: float | None = None) -> None:
        self._append_chat_bubble("Assistant", answer.answer, align="left", elapsed_seconds=elapsed_seconds)
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

    def _append_chat_bubble(
        self,
        role: str,
        text: str,
        *,
        align: str,
        elapsed_seconds: float | None = None,
    ) -> None:
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(8)

        bubble = QWidget()
        bubble_layout = QVBoxLayout(bubble)
        bubble_layout.setContentsMargins(12, 9, 12, 9)
        bubble_layout.setSpacing(5)

        label = QLabel(role)
        label.setStyleSheet("color: #d7ecff; font-size: 8pt; font-weight: 700;")
        message = QLabel(text)
        message.setWordWrap(True)
        message.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        message.setMinimumWidth(260)
        message.setMaximumWidth(560)
        message.setStyleSheet("color: #ffffff; line-height: 1.35;")
        bubble_layout.addWidget(label)
        bubble_layout.addWidget(message)

        if elapsed_seconds is not None:
            timer = QLabel(f"Time: {self._format_elapsed(elapsed_seconds)}")
            timer.setStyleSheet("color: #c9c9c9; font-size: 8pt;")
            bubble_layout.addWidget(timer)

        if align == "right":
            bubble.setStyleSheet("background: #0f6cbd; border-radius: 12px;")
            row_layout.addStretch(1)
            row_layout.addWidget(bubble, 0)
        else:
            bubble.setStyleSheet("background: #343434; border-radius: 12px;")
            row_layout.addWidget(bubble, 0)
            row_layout.addStretch(1)

        self.chat_layout.insertWidget(self.chat_layout.count() - 1, row)
        self._scroll_chat_to_bottom()

    def _append_system_message(self, text: str) -> None:
        label = QLabel(text)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setWordWrap(True)
        label.setStyleSheet("color: #bdbdbd; font-style: italic; padding: 4px;")
        self.chat_layout.insertWidget(self.chat_layout.count() - 1, label)
        self._scroll_chat_to_bottom()

    def _scroll_chat_to_bottom(self) -> None:
        def scroll() -> None:
            bar = self.chat_scroll.verticalScrollBar()
            bar.setValue(bar.maximum())

        QTimer.singleShot(0, scroll)

    def _format_elapsed(self, elapsed_seconds: float | None) -> str:
        if elapsed_seconds is None:
            return "unknown"
        if elapsed_seconds < 60:
            return f"{elapsed_seconds:.2f}s"
        minutes = int(elapsed_seconds // 60)
        seconds = elapsed_seconds - minutes * 60
        return f"{minutes}m {seconds:.1f}s"
