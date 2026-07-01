from __future__ import annotations

import html
from pathlib import Path

try:
    from PySide6.QtCore import QSize, QThreadPool, Qt
    from PySide6.QtGui import QColor
    from PySide6.QtWidgets import (
        QApplication,
        QDialog,
        QCheckBox,
        QComboBox,
        QFileDialog,
        QFormLayout,
        QFrame,
        QHBoxLayout,
        QInputDialog,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMenu,
        QMessageBox,
        QPlainTextEdit,
        QProgressBar,
        QPushButton,
        QSplitter,
        QTextBrowser,
        QTreeWidget,
        QTreeWidgetItem,
        QVBoxLayout,
        QWidget,
    )
except ImportError as exc:
    raise RuntimeError("Install desktop dependencies with `py -m pip install -e .[desktop]`.") from exc

from backend.app.models import Answer, Document
from desktop.controller import DesktopController
from desktop.theme import Colors, status_badge_style
from desktop.workers import FunctionWorker, WorkerResult


_TREE_SESSION_ROLE = Qt.ItemDataRole.UserRole
_TREE_TYPE_ROLE = Qt.ItemDataRole.UserRole + 1


class MainWindow(QMainWindow):
    def __init__(self, controller: DesktopController) -> None:
        super().__init__()
        self.controller = controller
        self.thread_pool = QThreadPool.globalInstance()
        self._chat_started = False
        self._chat_fragments: list[str] = []
        self.setWindowTitle("Local PDF RAG")
        self._build_ui()
        self._populate_sessions()
        self.refresh_status()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QWidget()
        root.setObjectName("root")
        rl = QVBoxLayout(root)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(0)
        rl.addWidget(self._build_header())

        body = QWidget()
        body.setStyleSheet(f"background-color: {Colors.sidebar_bg};")
        bl = QHBoxLayout(body)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.setSpacing(0)

        # Sidebar — full height, dark, no padding on left
        bl.addWidget(self._build_sidebar())

        # Right content area — panels with padding
        content = QWidget()
        content.setStyleSheet(f"background-color: {Colors.sidebar_bg};")
        cl = QVBoxLayout(content)
        cl.setContentsMargins(10, 10, 10, 8)
        cl.setSpacing(8)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self._build_chat_panel())
        splitter.addWidget(self._build_citations_panel())
        splitter.setSizes([680, 380])
        cl.addWidget(splitter, 1)
        cl.addLayout(self._build_status_row())
        bl.addWidget(content, 1)

        rl.addWidget(body, 1)
        self.setCentralWidget(root)

    def _build_header(self) -> QFrame:
        header = QFrame()
        header.setObjectName("headerBar")
        layout = QHBoxLayout(header)
        layout.setContentsMargins(16, 10, 16, 10)
        layout.setSpacing(12)

        vbox = QVBoxLayout()
        vbox.setSpacing(1)
        t = QLabel("Local PDF RAG")
        t.setObjectName("appTitle")
        s = QLabel("Offline document Q&A · fully local · no cloud")
        s.setObjectName("appSubtitle")
        vbox.addWidget(t)
        vbox.addWidget(s)
        layout.addLayout(vbox)
        layout.addStretch(1)

        self.connection_pill = QLabel()
        self.connection_pill.setMinimumHeight(26)
        layout.addWidget(self.connection_pill)
        self.settings_button = QPushButton("Settings")
        self.settings_button.setStyleSheet(
            f"background-color: #2A2A2E; color: {Colors.sidebar_text};"
            "border: 1px solid #3A3A3F; border-radius: 8px; padding: 6px 14px;"
            "font-size: 12px; font-weight: 600;"
        )
        self.settings_button.clicked.connect(self.show_settings)
        layout.addWidget(self.settings_button)
        return header

    def _build_sidebar(self) -> QFrame:
        panel = QFrame()
        panel.setObjectName("sidebarPanel")
        panel.setMinimumWidth(220)
        panel.setMaximumWidth(300)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)

        # New Chat button at top
        self.new_chat_btn = QPushButton("✦  New Chat")
        self.new_chat_btn.setObjectName("primaryButton")
        self.new_chat_btn.setFixedHeight(36)
        self.new_chat_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.new_chat_btn.clicked.connect(self._on_new_chat)
        layout.addWidget(self.new_chat_btn)

        # Divider label
        lbl = QLabel("CHATS")
        lbl.setObjectName("sectionTitle")
        lbl.setStyleSheet(
            f"color:{Colors.sidebar_text_muted};font-size:10px;font-weight:700;"
            "letter-spacing:1px;padding:8px 4px 4px 4px;"
        )
        layout.addWidget(lbl)

        self.session_tree = QTreeWidget()
        self.session_tree.setHeaderHidden(True)
        self.session_tree.setIndentation(14)
        self.session_tree.setAnimated(True)
        self.session_tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.session_tree.customContextMenuRequested.connect(self._on_tree_context_menu)
        self.session_tree.itemClicked.connect(self._on_tree_item_clicked)
        layout.addWidget(self.session_tree, 1)

        # Spacer then actions at bottom
        layout.addSpacing(4)

        self.import_btn = QPushButton("⬆  Import PDFs / Text")
        self.import_btn.setObjectName("sidebarButton")
        self.import_btn.setFixedHeight(34)
        self.import_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.import_btn.clicked.connect(self.import_documents)
        layout.addWidget(self.import_btn)

        self.repair_btn = QPushButton("🔧  Repair Stuck Imports")
        self.repair_btn.setObjectName("sidebarButton")
        self.repair_btn.setFixedHeight(34)
        self.repair_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.repair_btn.clicked.connect(self.repair_stuck_imports)
        layout.addWidget(self.repair_btn)
        return panel

    def _build_chat_panel(self) -> QFrame:
        panel = QFrame()
        panel.setObjectName("panel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(8)

        self.chat_title_label = QLabel("SELECT A CHAT")
        self.chat_title_label.setObjectName("sectionTitle")
        layout.addWidget(self.chat_title_label)

        self.chat_view = QTextBrowser()
        self.chat_view.setOpenExternalLinks(False)
        self.chat_view.setHtml(self._empty_chat_html())
        self.chat_view.setStyleSheet(
            "background-color:#F2F2F7;border-radius:10px;border:none;padding:2px;"
        )
        layout.addWidget(self.chat_view, 1)

        qrow = QHBoxLayout()
        qrow.setSpacing(8)
        self.question_input = QLineEdit()
        self.question_input.setPlaceholderText("Ask a question about this chat's documents…")
        self.question_input.returnPressed.connect(self.ask_question)
        self.question_input.setFixedHeight(42)
        # Dark input to match the dark-themed app; white text for visibility
        self.question_input.setStyleSheet(
            "QLineEdit {"
            "  background-color: #1C1C1E;"
            "  border: 1.5px solid #3A3A3F;"
            "  border-radius: 10px;"
            "  padding: 10px 14px;"
            "  font-size: 13px;"
            "  color: #FFFFFF;"
            "}"
            "QLineEdit:focus {"
            "  border: 1.5px solid #5E5CE6;"
            "}"
        )
        self.ask_button = QPushButton("Ask")
        self.ask_button.setObjectName("primaryButton")
        self.ask_button.setFixedHeight(42)
        self.ask_button.setFixedWidth(72)
        self.ask_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.ask_button.clicked.connect(self.ask_question)
        qrow.addWidget(self.question_input, 1)
        qrow.addWidget(self.ask_button)
        layout.addLayout(qrow)
        return panel

    def _build_citations_panel(self) -> QFrame:
        panel = QFrame()
        panel.setObjectName("panel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(8)
        lbl = QLabel("SOURCES & CITATIONS")
        lbl.setObjectName("sectionTitle")
        layout.addWidget(lbl)
        self.citation_view = QTextBrowser()
        self.citation_view.setOpenExternalLinks(False)
        self.citation_view.setHtml(self._empty_citations_html())
        self.citation_view.setStyleSheet(
            f"background-color:{Colors.surface};border:none;font-size:12px;"
        )
        layout.addWidget(self.citation_view, 1)
        return panel

    def _build_status_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(8)
        self.status_dot = QLabel("●")
        self.status_dot.setStyleSheet(f"color:{Colors.success};font-size:9px;")
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet(
            f"color:{Colors.sidebar_text_muted};font-size:11px;"
        )
        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(1)
        self.progress.setFixedWidth(120)
        self.progress.setFixedHeight(4)
        self.progress.setTextVisible(False)
        row.addWidget(self.status_dot)
        row.addWidget(self.status_label, 1)
        row.addWidget(self.progress)
        return row

    # ------------------------------------------------------------------
    # Session tree population + interaction
    # ------------------------------------------------------------------

    def _populate_sessions(self) -> None:
        self.session_tree.blockSignals(True)
        self.session_tree.clear()
        sessions = self.controller.list_sessions()
        active_id = self.controller.active_session_id()

        for session in sessions:
            doc_count = self.controller.session_document_count(session.session_id)
            suffix = f"  {doc_count} file{'s' if doc_count != 1 else ''}"
            top = QTreeWidgetItem([f"💬  {session.title}"])
            top.setData(0, _TREE_SESSION_ROLE, session.session_id)
            top.setData(0, _TREE_TYPE_ROLE, "session")
            top.setToolTip(0, session.title)
            top.setForeground(0, QColor(Colors.sidebar_text))

            if session.session_id == active_id:
                f = top.font(0)
                f.setBold(True)
                top.setFont(0, f)
                top.setForeground(0, QColor("#FFFFFF"))

            docs = self.controller.service.store.list_documents(session.session_id)
            for doc in docs:
                _bg, _fg, slabel = status_badge_style(doc.status)
                child = QTreeWidgetItem([f"    📄  {doc.filename}"])
                child.setData(0, _TREE_SESSION_ROLE, doc.document_id)
                child.setData(0, _TREE_TYPE_ROLE, "document")
                child.setToolTip(0, f"{doc.filename} — {slabel}")
                if doc.status == "failed":
                    child.setForeground(0, QColor(Colors.danger))
                elif doc.status == "ready":
                    child.setForeground(0, QColor(Colors.success))
                else:
                    child.setForeground(0, QColor(Colors.warning))
                top.addChild(child)

            self.session_tree.addTopLevelItem(top)
            if session.session_id == active_id:
                top.setExpanded(True)
                self.session_tree.setCurrentItem(top)
                # Update chat title to active session
                self.chat_title_label.setText(
                    f"CHAT  ·  {session.title}  ·  {doc_count} doc{'s' if doc_count != 1 else ''}"
                )

        self.session_tree.blockSignals(False)
        self.refresh_status()

    def _on_tree_item_clicked(self, item: QTreeWidgetItem, _col: int) -> None:
        kind = item.data(0, _TREE_TYPE_ROLE)
        if kind != "session":
            return
        session_id = item.data(0, _TREE_SESSION_ROLE)
        if session_id == self.controller.active_session_id():
            return
        self.controller.set_active_session(session_id)
        self._chat_started = False
        self._chat_fragments = []
        self.chat_view.setHtml(self._empty_chat_html())
        self.citation_view.setHtml(self._empty_citations_html())
        self._populate_sessions()

    def _on_tree_context_menu(self, pos: object) -> None:
        item = self.session_tree.itemAt(pos)
        if not item:
            return
        kind = item.data(0, _TREE_TYPE_ROLE)
        menu = QMenu(self)

        if kind == "session":
            session_id = item.data(0, _TREE_SESSION_ROLE)
            title = item.toolTip(0)
            rename_act = menu.addAction("✏️  Rename Chat")
            menu.addSeparator()
            delete_act = menu.addAction("🗑️  Delete Chat")
            chosen = menu.exec(self.session_tree.viewport().mapToGlobal(pos))
            if chosen == rename_act:
                self._rename_session(session_id, title)
            elif chosen == delete_act:
                self._delete_session(session_id, title)

        elif kind == "document":
            document_id = item.data(0, _TREE_SESSION_ROLE)  # stores doc_id for doc items
            filename = item.toolTip(0).split(" — ")[0]
            remove_act = menu.addAction("🗑️  Remove Document")
            chosen = menu.exec(self.session_tree.viewport().mapToGlobal(pos))
            if chosen == remove_act:
                self._delete_document(document_id, filename)

    def _rename_session(self, session_id: str, current_title: str) -> None:
        new_title, ok = QInputDialog.getText(
            self, "Rename Chat", "Chat name:", QLineEdit.EchoMode.Normal, current_title
        )
        if ok and new_title.strip():
            self.controller.rename_session(session_id, new_title.strip())
            if session_id == self.controller.active_session_id():
                self.chat_title_label.setText(f"CHAT  —  {new_title.strip()}")
            self._populate_sessions()

    def _delete_session(self, session_id: str, title: str) -> None:
        reply = QMessageBox.question(
            self, "Delete Chat",
            f'Delete "{title}" and all its documents?\n\nThis cannot be undone.',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        self.controller.delete_session(session_id)
        self._chat_started = False
        self._chat_fragments = []
        self.chat_view.setHtml(self._empty_chat_html())
        self.citation_view.setHtml(self._empty_citations_html())
        self.chat_title_label.setText("SELECT A CHAT")
        self._populate_sessions()

    def _delete_document(self, document_id: str, filename: str) -> None:
        reply = QMessageBox.question(
            self, "Remove Document",
            f'Remove "{filename}" from this chat?\n\n'
            "It will be removed from the search index. This cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        try:
            self.controller.delete_document(document_id)
        except Exception as exc:
            QMessageBox.critical(self, "Remove Document", f"Failed to remove:\n{exc}")
            return
        self._append_system_note(f"Removed: {filename}")
        self._populate_sessions()

    # ------------------------------------------------------------------
    # Status bar
    # ------------------------------------------------------------------

    def refresh_status(self) -> None:
        status = self.controller.status()
        ready = bool(status["ollama_ready"])
        dot_color = Colors.success if ready else Colors.warning
        self.status_dot.setStyleSheet(f"color:{dot_color};font-size:9px;")
        self.status_label.setText(
            f"Session: {status['session_id']}  ·  "
            f"Docs: {status['documents']}  ·  "
            f"Chunks: {status['chunks']}  ·  "
            f"{status['ollama_message']}"
        )
        pill_text = "● Models ready" if ready else "● Setup needed"
        pill_bg = "#1A3A2A" if ready else "#3A2A0A"
        pill_fg = Colors.success if ready else Colors.warning
        self.connection_pill.setText(pill_text)
        self.connection_pill.setStyleSheet(
            f"background-color:{pill_bg};color:{pill_fg};font-size:11px;font-weight:700;"
            "border-radius:12px;padding:4px 12px;"
        )

    # ------------------------------------------------------------------
    # Chat actions
    # ------------------------------------------------------------------

    def _on_new_chat(self) -> None:
        session = self.controller.create_session()
        self._chat_started = False
        self._chat_fragments = []
        self.chat_view.setHtml(self._empty_chat_html())
        self.citation_view.setHtml(self._empty_citations_html())
        self.chat_title_label.setText(f"CHAT  —  {session.title}")
        self._populate_sessions()

        # Immediately prompt for files — if none selected, remove the empty chat
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            f'Import PDFs into "{session.title}"',
            str(Path.home()),
            "Documents (*.pdf *.txt *.md);;All files (*.*)",
        )

        if not paths:
            # User cancelled — clean up the empty session silently
            self.controller.delete_session(session.session_id)
            self._chat_started = False
            self._chat_fragments = []
            self.chat_view.setHtml(self._empty_chat_html())
            self.citation_view.setHtml(self._empty_citations_html())
            self.chat_title_label.setText("SELECT A CHAT")
            self._populate_sessions()
            return

        # Ingest each selected file into the new session
        for path in paths:
            name = Path(path).name

            def _make_cb(doc_name: str):
                def _cb(done: int, total: int, msg: str) -> None:
                    pct = int(100 * done / total) if total else 0
                    self.status_label.setText(f"{doc_name}: {msg}  ({pct}%)")
                return _cb

            self._run_worker(
                f"Ingesting {name}…",
                self.controller.ingest,
                path,
                False,
                _make_cb(name),
            )

    def import_documents(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Import PDFs or text files",
            str(Path.home()),
            "Documents (*.pdf *.txt *.md);;All files (*.*)",
        )
        for path in paths:
            name = Path(path).name

            # Build a progress callback that posts to the UI thread safely.
            # We capture `name` by closure; Qt signals are thread-safe.
            def _make_cb(doc_name: str):
                def _cb(done: int, total: int, msg: str) -> None:
                    # Update the status label from the worker thread.
                    # QLabel.setText is thread-safe in PySide6.
                    pct = int(100 * done / total) if total else 0
                    self.status_label.setText(
                        f"{doc_name}: {msg}  ({pct}%)"
                    )
                return _cb

            cb = _make_cb(name)
            self._run_worker(
                f"Ingesting {name}…",
                self.controller.ingest,
                path,
                False,   # build_okf
                cb,      # progress_callback
            )

    def repair_stuck_imports(self) -> None:
        self._run_worker("Repairing stuck imports...", self.controller.repair_unready_documents)

    def ask_question(self) -> None:
        question = self.question_input.text().strip()
        if not question:
            return
        status = self.controller.status()
        if status["documents"] and not status["chunks"]:
            QMessageBox.warning(
                self, "No Searchable Text Yet",
                "Documents are listed but none have searchable chunks yet.\n\n"
                "Click 'Repair Stuck Imports' or reimport the PDFs.",
            )
            return
        self.question_input.clear()
        self._clear_empty_chat_if_needed()
        self._append_bubble("user", question)
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
        for m in available_models:
            active_model.addItem(m)
            embedding_model.addItem(m)
        self._set_combo(active_model, self.controller.settings.active_model)
        self._set_combo(embedding_model, self.controller.settings.embedding_model)
        form.addRow("Data directory", QLabel(str(self.controller.settings.data_dir)))
        form.addRow("Ollama enabled", use_ollama)
        form.addRow("Ollama URL", ollama_url)
        form.addRow("Active model", active_model)
        form.addRow("Embedding model", embedding_model)
        form.addRow("Ollama required", self._bool_label(readiness.ollama_required))
        form.addRow("Ollama reachable", self._bool_label(readiness.ollama_reachable))
        form.addRow("Offline ready", self._bool_label(readiness.ready))
        layout.addLayout(form)
        lbl = QLabel("SETUP COMMANDS / STATUS")
        lbl.setObjectName("sectionTitle")
        layout.addWidget(lbl)
        commands = QPlainTextEdit()
        commands.setReadOnly(True)
        model_list = "\n".join(f"- {m}" for m in available_models)
        commands.setPlainText(
            "\n".join(readiness.setup_commands) or readiness.message
            + ("\n\nInstalled Ollama models:\n" + model_list if model_list else "")
        )
        layout.addWidget(commands)
        btns = QHBoxLayout()
        refresh_btn = QPushButton("Refresh Models")
        refresh_btn.clicked.connect(
            lambda: self._refresh_model_combos(ollama_url.text().strip(), active_model, embedding_model)
        )
        save_btn = QPushButton("Save Settings")
        save_btn.setObjectName("primaryButton")
        save_btn.clicked.connect(
            lambda: self._save_settings(
                dialog, use_ollama.isChecked(), ollama_url.text().strip(),
                active_model.currentText().strip(), embedding_model.currentText().strip(),
            )
        )
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.reject)
        btns.addWidget(refresh_btn)
        btns.addStretch(1)
        btns.addWidget(save_btn)
        btns.addWidget(close_btn)
        layout.addLayout(btns)
        dialog.resize(640, 460)
        dialog.exec()

    @staticmethod
    def _bool_label(value: bool) -> QLabel:
        lbl = QLabel("Yes" if value else "No")
        lbl.setStyleSheet(f"color:{Colors.success if value else Colors.danger};font-weight:700;")
        return lbl

    def _set_combo(self, combo: QComboBox, text: str) -> None:
        idx = combo.findText(text)
        if idx >= 0:
            combo.setCurrentIndex(idx)
        else:
            combo.addItem(text)
            combo.setCurrentText(text)

    def _refresh_model_combos(self, url: str, ac: QComboBox, em: QComboBox) -> None:
        try:
            models = self.controller.available_ollama_models(url)
        except OSError as exc:
            QMessageBox.warning(self, "Ollama Models", f"Could not reach Ollama:\n{exc}")
            return
        cur_ac, cur_em = ac.currentText(), em.currentText()
        ac.clear(); em.clear()
        for m in models:
            ac.addItem(m); em.addItem(m)
        self._set_combo(ac, cur_ac); self._set_combo(em, cur_em)

    def _save_settings(self, dialog: QDialog, use_ollama: bool, url: str, am: str, em: str) -> None:
        if not am or not em:
            QMessageBox.warning(self, "Settings", "Choose both an active model and an embedding model.")
            return
        self.controller.update_preferences(
            use_ollama=use_ollama, ollama_base_url=url or "http://localhost:11434",
            active_model=am, embedding_model=em,
        )
        self._populate_sessions()
        dialog.accept()

    # ------------------------------------------------------------------
    # Background workers
    # ------------------------------------------------------------------

    def _run_worker(self, label: str, fn: object, *args: object) -> None:
        worker = FunctionWorker(label, fn, *args)
        worker.signals.started.connect(self._on_worker_started)
        worker.signals.finished.connect(self._on_worker_finished)
        worker.signals.error.connect(self._on_worker_error)
        self.thread_pool.start(worker)

    def _on_worker_started(self, label: str) -> None:
        self.status_label.setText(label)
        self.status_dot.setStyleSheet(f"color:{Colors.accent};font-size:10px;")
        self.progress.setRange(0, 0)
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)

    def _on_worker_finished(self, result: object) -> None:
        elapsed: float | None = None
        if isinstance(result, WorkerResult):
            elapsed = result.elapsed_seconds
            result = result.result
        self.progress.setRange(0, 1)
        self.progress.setValue(1)
        QApplication.restoreOverrideCursor()
        if isinstance(result, Answer):
            self._show_answer(result, elapsed)
            if elapsed is not None:
                self.status_label.setText(f"Answered in {self._fmt_elapsed(elapsed)}")
        elif isinstance(result, Document):
            self._append_system_note(f"Imported: {result.filename}")
            self._populate_sessions()
        elif isinstance(result, list):
            self._append_system_note(f"Repaired {len(result)} document(s).")
            self._populate_sessions()
        else:
            self.refresh_status()

    def _on_worker_error(self, message: str) -> None:
        self.progress.setRange(0, 1)
        self.progress.setValue(1)
        QApplication.restoreOverrideCursor()
        self.status_dot.setStyleSheet(f"color:{Colors.danger};font-size:10px;")
        self.status_label.setText("Error")
        QMessageBox.critical(self, "Local PDF RAG Error", self._friendly_error(message))

    def _friendly_error(self, message: str) -> str:
        if "PDF parsing requires PyMuPDF" in message:
            return (
                "PDF parsing is not available in this Python environment.\n\n"
                "Fix:\n  py -m pip install -e .[desktop]\n\n"
                f"{message}"
            )
        if "no searchable text was extracted" in message:
            return (
                "The PDF parser opened this file but found no searchable text.\n\n"
                "If this PDF is scanned or image-only, install OCR support:\n"
                "  py -m pip install -e .[pdf]\n\n"
                "Then restart and click Repair Stuck Imports."
            )
        return message

    # ------------------------------------------------------------------
    # Chat / citation rendering  — Qt-compatible bubble layout
    # ------------------------------------------------------------------
    # Qt's QTextBrowser HTML engine is a subset of HTML4. It does NOT support:
    #   - display:inline-block on span/div
    #   - float:right
    #   - flexbox
    # The ONLY reliable way to right-align a block is <table align="right">.

    _CHAT_CSS = (
        "<style>"
        "body{margin:0;padding:6px 8px;background:#F2F2F7;font-family:'Segoe UI',Arial,sans-serif;}"
        "table.bubble{width:100%;border-collapse:collapse;margin:3px 0;}"
        "td.user-cell{"
            "background:#5E5CE6;color:#ffffff;"
            "border-radius:16px 16px 2px 16px;"
            "padding:9px 14px;font-size:13px;line-height:1.5;"
            "text-align:left;"
        "}"
        "td.bot-cell{"
            "background:#FFFFFF;color:#1C1C1E;"
            "border-radius:2px 16px 16px 16px;"
            "padding:9px 14px;font-size:13px;line-height:1.5;"
            "border:1px solid #E5E5EA;"
        "}"
        "td.spacer{background:transparent;}"
        ".timer{font-size:10px;color:#8E8E93;margin-top:4px;display:block;}"
        ".note{color:#8E8E93;font-size:11px;font-style:italic;text-align:center;margin:4px 0;}"
        "b,strong{font-weight:700;}"
        "ul,ol{margin:4px 0 4px 16px;padding:0;}"
        "li{margin:2px 0;}"
        "</style>"
    )

    def _flush_chat(self) -> None:
        body = "".join(self._chat_fragments)
        self.chat_view.setHtml(self._CHAT_CSS + "<body>" + body + "</body>")
        sb = self.chat_view.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _empty_chat_html(self) -> str:
        return (
            self._CHAT_CSS
            + "<body><p class='note' style='margin-top:60px;font-size:13px;'>"
            "Select a chat from the sidebar or create a new one.<br/>"
            "<span style='font-size:11px;color:#AEAEB2;'>"
            "Import PDFs, then ask a question.</span></p></body>"
        )

    def _empty_citations_html(self) -> str:
        return (
            "<p style='color:#AEAEB2;padding:30px 16px;text-align:center;"
            "font-size:12px;'>Source excerpts will appear here after each answer.</p>"
        )

    def _clear_empty_chat_if_needed(self) -> None:
        if not self._chat_started:
            self._chat_fragments = []
            self._chat_started = True

    def _bubble_html(self, role: str, text: str, timer_text: str = "") -> str:
        """Build a Qt-compatible table-based chat bubble."""
        import re as _re
        # Convert minimal markdown for assistant messages
        if role == "assistant":
            body_html = self._md_to_html(text)
        else:
            body_html = html.escape(text).replace("\n", "<br/>")

        timer_span = (
            f"<span class='timer'>{html.escape(timer_text)}</span>"
            if timer_text else ""
        )

        if role == "user":
            # User: right-aligned — spacer(30%) | bubble(70%)
            return (
                "<table class='bubble'><tr>"
                "<td class='spacer' width='28%'></td>"
                f"<td class='user-cell' width='72%'>{body_html}{timer_span}</td>"
                "</tr></table>"
            )
        else:
            # Assistant: left-aligned — bubble(72%) | spacer(28%)
            return (
                "<table class='bubble'><tr>"
                f"<td class='bot-cell' width='72%'>{body_html}{timer_span}</td>"
                "<td class='spacer' width='28%'></td>"
                "</tr></table>"
            )

    def _append_bubble(self, role: str, text: str, timer_text: str = "") -> None:
        self._chat_fragments.append(self._bubble_html(role, text, timer_text))
        self._flush_chat()

    def _md_to_html(self, text: str) -> str:
        """Minimal markdown → Qt-compatible HTML."""
        import re as _re
        escaped = html.escape(text)
        # **bold**
        escaped = _re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escaped)
        lines = escaped.split("\n")
        out: list[str] = []
        in_list = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("* ") or stripped.startswith("- "):
                if not in_list:
                    out.append("<ul>")
                    in_list = True
                out.append(f"<li>{stripped[2:]}</li>")
            elif stripped.startswith("## "):
                if in_list:
                    out.append("</ul>")
                    in_list = False
                out.append(f"<b>{stripped[3:]}</b><br/>")
            elif stripped.startswith("# "):
                if in_list:
                    out.append("</ul>")
                    in_list = False
                out.append(f"<b>{stripped[2:]}</b><br/>")
            else:
                if in_list:
                    out.append("</ul>")
                    in_list = False
                if stripped:
                    out.append(stripped + "<br/>")
                else:
                    out.append("<br/>")
        if in_list:
            out.append("</ul>")
        # Strip trailing <br/>
        result = "".join(out)
        while result.endswith("<br/>"):
            result = result[:-5]
        return result

    def _append_system_note(self, text: str) -> None:
        self._clear_empty_chat_if_needed()
        self._chat_fragments.append(
            f"<p class='note'>{html.escape(text)}</p>"
        )
        self._flush_chat()

    def _show_answer(self, answer: Answer, elapsed: float | None = None) -> None:
        self._clear_empty_chat_if_needed()
        timer_text = f"⏱ {self._fmt_elapsed(elapsed)}" if elapsed is not None else ""
        self._append_bubble("assistant", answer.answer, timer_text)

        if not answer.citations:
            self.citation_view.setHtml(
                "<p style='color:#AEAEB2;padding:24px;text-align:center;'>"
                "No citations for this answer.</p>"
            )
        else:
            rows: list[str] = []
            for cit in answer.citations:
                pages = (
                    str(cit.page_start)
                    if cit.page_start == cit.page_end
                    else f"{cit.page_start}–{cit.page_end}"
                )
                ex = html.escape(cit.excerpt).replace("\n", "<br/>")
                fn = html.escape(cit.filename)
                sid = html.escape(cit.source_id)
                rows.append(
                    "<table width='100%' cellpadding='0' cellspacing='0' "
                    "style='border:1px solid #D1D1D6;border-radius:8px;"
                    "margin-bottom:10px;background:#FAFAFA;'>"
                    "<tr><td style='padding:10px 13px;'>"
                    f"<b style='color:#5E5CE6;font-size:12px;'>{sid}</b><br/>"
                    f"<span style='font-size:11px;color:#636366;'>📄 {fn} · page {pages}</span><br/>"
                    f"<span style='font-size:12px;color:#1C1C1E;line-height:1.45;'>{ex}</span>"
                    "</td></tr></table>"
                )
            self.citation_view.setHtml(
                "<body style='padding:8px;background:#FFFFFF;'>"
                + "".join(rows)
                + "</body>"
            )
        self.refresh_status()

    def _fmt_elapsed(self, secs: float | None) -> str:
        if secs is None:
            return "unknown"
        if secs < 60:
            return f"{secs:.2f}s"
        m = int(secs // 60)
        return f"{m}m {secs - m * 60:.1f}s"