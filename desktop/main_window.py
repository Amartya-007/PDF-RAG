from __future__ import annotations

import html
from pathlib import Path

try:
    from PySide6.QtCore import QSize, QThreadPool, Qt
    from PySide6.QtWidgets import (
        QApplication,
        QDialog,
        QCheckBox,
        QComboBox,
        QFileDialog,
        QFormLayout,
        QFrame,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QListWidget,
        QListWidgetItem,
        QMainWindow,
        QMessageBox,
        QPlainTextEdit,
        QProgressBar,
        QPushButton,
        QSizePolicy,
        QSplitter,
        QTextBrowser,
        QVBoxLayout,
        QWidget,
    )
except ImportError as exc:  # pragma: no cover - exercised only when launching the UI.
    raise RuntimeError("Install desktop dependencies with `py -m pip install -e .[desktop]`.") from exc

from backend.app.models import Answer, Document
from desktop.controller import DesktopController
from desktop.theme import Colors, status_badge_style
from desktop.workers import FunctionWorker


class MainWindow(QMainWindow):
    def __init__(self, controller: DesktopController) -> None:
        super().__init__()
        self.controller = controller
        self.thread_pool = QThreadPool.globalInstance()
        self.setWindowTitle("Local PDF RAG")
        self._chat_started = False
        self._build_ui()
        self.refresh_documents()
        self.refresh_status()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QWidget()
        root.setObjectName("root")
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        root_layout.addWidget(self._build_header())

        body = QWidget()
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(16, 16, 16, 12)
        body_layout.setSpacing(10)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self._build_documents_panel())
        splitter.addWidget(self._build_chat_panel())
        splitter.addWidget(self._build_citations_panel())
        splitter.setSizes([260, 640, 380])
        body_layout.addWidget(splitter, 1)

        body_layout.addLayout(self._build_status_row())
        root_layout.addWidget(body, 1)

        self.setCentralWidget(root)

    def _build_header(self) -> QFrame:
        header = QFrame()
        header.setObjectName("headerBar")
        layout = QHBoxLayout(header)
        layout.setContentsMargins(20, 14, 20, 14)
        layout.setSpacing(16)

        title_box = QVBoxLayout()
        title_box.setSpacing(2)
        title = QLabel("Local PDF RAG")
        title.setObjectName("appTitle")
        subtitle = QLabel("Offline document Q&A with source citations")
        subtitle.setObjectName("appSubtitle")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        layout.addLayout(title_box)
        layout.addStretch(1)

        self.connection_pill = QLabel()
        self.connection_pill.setMinimumHeight(28)
        layout.addWidget(self.connection_pill)

        self.import_button = QPushButton("Import PDFs / Text")
        self.import_button.setObjectName("primaryButton")
        self.import_button.clicked.connect(self.import_documents)
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.refresh_documents)
        self.repair_button = QPushButton("Repair Stuck Imports")
        self.repair_button.clicked.connect(self.repair_stuck_imports)
        self.settings_button = QPushButton("Settings")
        self.settings_button.clicked.connect(self.show_settings)

        for button in (self.repair_button, self.refresh_button, self.settings_button, self.import_button):
            layout.addWidget(button)

        return header

    def _build_documents_panel(self) -> QFrame:
        panel = QFrame()
        panel.setObjectName("panel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(14, 12, 14, 14)
        layout.setSpacing(8)

        header_row = QHBoxLayout()
        title = QLabel("DOCUMENTS")
        title.setObjectName("sectionTitle")
        header_row.addWidget(title)
        header_row.addStretch(1)
        self.document_count_label = QLabel("")
        self.document_count_label.setStyleSheet(f"color: {Colors.text_faint}; font-size: 11px;")
        header_row.addWidget(self.document_count_label)
        layout.addLayout(header_row)

        self.document_list = QListWidget()
        self.document_list.setSpacing(2)
        self.document_list.setUniformItemSizes(False)
        layout.addWidget(self.document_list, 1)
        return panel

    def _build_chat_panel(self) -> QFrame:
        panel = QFrame()
        panel.setObjectName("panel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(14, 12, 14, 14)
        layout.setSpacing(8)

        title = QLabel("CHAT")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)

        self.chat_view = QTextBrowser()
        self.chat_view.setOpenExternalLinks(False)
        self.chat_view.setHtml(self._empty_chat_html())
        layout.addWidget(self.chat_view, 1)

        question_row = QHBoxLayout()
        question_row.setSpacing(8)
        self.question_input = QLineEdit()
        self.question_input.setPlaceholderText("Ask a question about your local documents...")
        self.question_input.returnPressed.connect(self.ask_question)
        self.ask_button = QPushButton("Ask")
        self.ask_button.setObjectName("primaryButton")
        self.ask_button.clicked.connect(self.ask_question)
        question_row.addWidget(self.question_input, 1)
        question_row.addWidget(self.ask_button)
        layout.addLayout(question_row)
        return panel

    def _build_citations_panel(self) -> QFrame:
        panel = QFrame()
        panel.setObjectName("panel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(14, 12, 14, 14)
        layout.setSpacing(8)

        title = QLabel("CITATIONS & SOURCE EXCERPTS")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)

        self.citation_view = QTextBrowser()
        self.citation_view.setOpenExternalLinks(False)
        self.citation_view.setHtml(self._empty_citations_html())
        layout.addWidget(self.citation_view, 1)
        return panel

    def _build_status_row(self) -> QHBoxLayout:
        status_row = QHBoxLayout()
        status_row.setSpacing(10)
        self.status_dot = QLabel("\u25cf")
        self.status_dot.setStyleSheet(f"color: {Colors.success}; font-size: 10px;")
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet(f"color: {Colors.text_muted}; font-size: 12px;")
        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(1)
        self.progress.setFixedWidth(160)
        self.progress.setTextVisible(False)
        status_row.addWidget(self.status_dot)
        status_row.addWidget(self.status_label, 1)
        status_row.addWidget(self.progress)
        return status_row

    # ------------------------------------------------------------------
    # Data refresh
    # ------------------------------------------------------------------

    def refresh_documents(self) -> None:
        self.document_list.clear()
        documents = self.controller.list_documents()
        if not documents:
            self.document_count_label.setText("0")
            placeholder = QListWidgetItem()
            placeholder.setFlags(Qt.ItemFlag.NoItemFlags)
            self.document_list.addItem(placeholder)
            self.document_list.setItemWidget(placeholder, self._empty_state_widget(
                "No documents yet.\nImport a PDF to get started."
            ))
        else:
            self.document_count_label.setText(str(len(documents)))
            for document in documents:
                self._add_document_row(document)
        self.refresh_status()

    def _add_document_row(self, document: Document) -> None:
        item = QListWidgetItem()
        item.setSizeHint(QSize(0, 56))
        self.document_list.addItem(item)
        self.document_list.setItemWidget(item, self._document_row_widget(document))

    def _document_row_widget(self, document: Document) -> QWidget:
        row = QWidget()
        layout = QVBoxLayout(row)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(4)

        name_label = QLabel(document.filename)
        name_label.setStyleSheet("font-size: 12px; font-weight: 600;")
        name_label.setWordWrap(True)
        layout.addWidget(name_label)

        bg, fg, label = status_badge_style(document.status)
        badge_row = QHBoxLayout()
        badge = QLabel(label)
        badge.setStyleSheet(
            f"background-color: {bg}; color: {fg}; font-size: 10px; font-weight: 700;"
            "border-radius: 8px; padding: 2px 8px;"
        )
        badge.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        badge_row.addWidget(badge)
        badge_row.addStretch(1)
        layout.addLayout(badge_row)
        return row

    def refresh_status(self) -> None:
        status = self.controller.status()
        ready = bool(status["ollama_ready"])
        dot_color = Colors.success if ready else Colors.warning
        self.status_dot.setStyleSheet(f"color: {dot_color}; font-size: 10px;")
        self.status_label.setText(
            f"Documents: {status['documents']}  \u00b7  Chunks: {status['chunks']}  \u00b7  "
            f"Concepts: {status['concepts']}  \u00b7  {status['ollama_message']}"
        )
        pill_bg = Colors.success_soft if ready else Colors.warning_soft
        pill_fg = Colors.success if ready else Colors.warning
        pill_text = "Models ready" if ready else "Setup needed"
        self.connection_pill.setText(pill_text)
        self.connection_pill.setStyleSheet(
            f"background-color: {pill_bg}; color: {pill_fg}; font-size: 11px; font-weight: 700;"
            "border-radius: 14px; padding: 5px 14px;"
        )

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

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
        self._clear_empty_chat_if_needed()
        self._append_chat_bubble("user", question)
        self._run_worker("Answering question...", self.controller.ask, question, False)

    def show_settings(self) -> None:
        readiness = self.controller.model_readiness()
        dialog = QDialog(self)
        dialog.setWindowTitle("Offline Settings")
        layout = QVBoxLayout(dialog)
        layout.setSpacing(12)
        form = QFormLayout()
        form.setSpacing(10)
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
        form.addRow("Ollama required", self._status_value_label(readiness.ollama_required))
        form.addRow("Ollama reachable", self._status_value_label(readiness.ollama_reachable))
        form.addRow("Offline ready", self._status_value_label(readiness.ready))
        layout.addLayout(form)
        commands_title = QLabel("SETUP COMMANDS / STATUS")
        commands_title.setObjectName("sectionTitle")
        layout.addWidget(commands_title)
        commands = QPlainTextEdit()
        commands.setReadOnly(True)
        model_list = "\n".join(f"- {model}" for model in available_models)
        commands.setPlainText(
            "\n".join(readiness.setup_commands)
            or readiness.message
            + ("\n\nInstalled Ollama models:\n" + model_list if model_list else "")
        )
        layout.addWidget(commands)
        buttons = QHBoxLayout()
        refresh_models = QPushButton("Refresh Models")
        refresh_models.clicked.connect(
            lambda: self._refresh_model_combos(ollama_url.text().strip(), active_model, embedding_model)
        )
        save = QPushButton("Save Settings")
        save.setObjectName("primaryButton")
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
        dialog.resize(640, 460)
        dialog.exec()

    @staticmethod
    def _status_value_label(value: bool) -> QLabel:
        label = QLabel("Yes" if value else "No")
        color = Colors.success if value else Colors.danger
        label.setStyleSheet(f"color: {color}; font-weight: 700;")
        return label

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

    # ------------------------------------------------------------------
    # Background work
    # ------------------------------------------------------------------

    def _run_worker(self, label: str, fn: object, *args: object) -> None:
        worker = FunctionWorker(label, fn, *args)
        worker.signals.started.connect(self._worker_started)
        worker.signals.finished.connect(self._worker_finished)
        worker.signals.error.connect(self._worker_error)
        self.thread_pool.start(worker)

    def _worker_started(self, label: str) -> None:
        self.status_label.setText(label)
        self.status_dot.setStyleSheet(f"color: {Colors.accent}; font-size: 10px;")
        self.progress.setRange(0, 0)
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)

    def _worker_finished(self, result: object) -> None:
        self.progress.setRange(0, 1)
        self.progress.setValue(1)
        QApplication.restoreOverrideCursor()
        if isinstance(result, Answer):
            self._show_answer(result)
        elif isinstance(result, Document):
            self._append_chat_note(f"Imported {result.filename}")
            self.refresh_documents()
        elif isinstance(result, list):
            self._append_chat_note(f"Repaired {len(result)} document(s).")
            self.refresh_documents()
        else:
            self.refresh_status()

    def _worker_error(self, message: str) -> None:
        self.progress.setRange(0, 1)
        self.progress.setValue(1)
        QApplication.restoreOverrideCursor()
        self.status_dot.setStyleSheet(f"color: {Colors.danger}; font-size: 10px;")
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

    # ------------------------------------------------------------------
    # Chat & citation rendering
    # ------------------------------------------------------------------

    def _empty_chat_html(self) -> str:
        return (
            f"<div style='color:{Colors.text_faint}; padding: 24px; text-align:center;'>"
            "No messages yet.<br>Import a document and ask a question to get started."
            "</div>"
        )

    def _empty_citations_html(self) -> str:
        return (
            f"<div style='color:{Colors.text_faint}; padding: 24px; text-align:center;'>"
            "Citations for your next answer will appear here."
            "</div>"
        )

    def _clear_empty_chat_if_needed(self) -> None:
        if not self._chat_started:
            self.chat_view.setHtml("")
            self._chat_started = True

    def _append_chat_bubble(self, role: str, text: str) -> None:
        escaped = html.escape(text).replace("\n", "<br>")
        if role == "user":
            bubble = (
                "<div style='margin: 6px 0; text-align: right;'>"
                f"<span style='background-color:{Colors.user_bubble}; color:{Colors.user_bubble_text}; "
                "border-radius: 12px; padding: 8px 14px; display: inline-block; max-width: 80%; "
                f"font-size: 13px;'>{escaped}</span></div>"
            )
        else:
            bubble = (
                "<div style='margin: 6px 0; text-align: left;'>"
                f"<span style='background-color:{Colors.assistant_bubble}; color:{Colors.assistant_bubble_text}; "
                "border-radius: 12px; padding: 8px 14px; display: inline-block; max-width: 80%; "
                f"font-size: 13px;'>{escaped}</span></div>"
            )
        self.chat_view.append(bubble)
        self._scroll_chat_to_bottom()

    def _append_chat_note(self, text: str) -> None:
        self._clear_empty_chat_if_needed()
        note = (
            f"<div style='margin: 8px 0; text-align: center; color:{Colors.text_faint}; font-size: 11px;'>"
            f"{html.escape(text)}</div>"
        )
        self.chat_view.append(note)
        self._scroll_chat_to_bottom()

    def _scroll_chat_to_bottom(self) -> None:
        scrollbar = self.chat_view.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _empty_state_widget(self, text: str) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(8, 24, 8, 24)
        label = QLabel(text)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setWordWrap(True)
        label.setStyleSheet(f"color: {Colors.text_faint}; font-size: 12px;")
        layout.addWidget(label)
        return widget

    def _show_answer(self, answer: Answer) -> None:
        self._clear_empty_chat_if_needed()
        self._append_chat_bubble("assistant", answer.answer)
        if not answer.citations:
            self.citation_view.setHtml(
                f"<div style='color:{Colors.text_faint}; padding: 24px; text-align:center;'>"
                "No citations for this answer.</div>"
            )
        else:
            cards = []
            for citation in answer.citations:
                pages = (
                    str(citation.page_start)
                    if citation.page_start == citation.page_end
                    else f"{citation.page_start}-{citation.page_end}"
                )
                cards.append(
                    "<div style='border:1px solid {border}; border-radius:10px; padding:10px 12px; "
                    "margin-bottom:10px; background-color:{surface};'>"
                    "<div style='font-size:12px; font-weight:700; color:{accent};'>{source_id}</div>"
                    "<div style='font-size:11px; color:{muted}; margin:2px 0 6px 0;'>"
                    "{filename} &middot; page {pages} &middot; chunk <code>{chunk_id}</code></div>"
                    "<div style='font-size:12px; color:{text}; line-height:1.4;'>{excerpt}</div>"
                    "</div>".format(
                        border=Colors.border,
                        surface=Colors.surface_alt,
                        accent=Colors.accent,
                        muted=Colors.text_muted,
                        text=Colors.text,
                        source_id=html.escape(citation.source_id),
                        filename=html.escape(citation.filename),
                        pages=html.escape(pages),
                        chunk_id=html.escape(citation.chunk_id),
                        excerpt=html.escape(citation.excerpt).replace("\n", "<br>"),
                    )
                )
            self.citation_view.setHtml("".join(cards))
        self.refresh_status()
