from __future__ import annotations

import html
from pathlib import Path

try:
    from PySide6.QtCore import QSize, QThreadPool, Qt, QTimer
    from PySide6.QtGui import QColor, QFont, QFontDatabase
    from PySide6.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QDialog,
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
        QScrollArea,
        QSizePolicy,
        QSplitter,
        QTextBrowser,
        QTreeWidget,
        QTreeWidgetItem,
        QVBoxLayout,
        QWidget,
    )
except ImportError as exc:
    raise RuntimeError(
        "Install desktop dependencies with `py -m pip install -e .[desktop]`."
    ) from exc

from backend.app.models import Answer, Citation, Document
from desktop.controller import DesktopController
from desktop.theme import Colors, relevance_color, status_badge
from desktop.workers import FunctionWorker, WorkerResult

_TREE_SESSION_ROLE = Qt.ItemDataRole.UserRole
_TREE_TYPE_ROLE = Qt.ItemDataRole.UserRole + 1


# ══════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════

def _label(text: str, size: int = 12, bold: bool = False,
           color: str = Colors.text, wrap: bool = False) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet(
        f"color:{color}; font-size:{size}px;"
        + (" font-weight:700;" if bold else "")
    )
    if wrap:
        lbl.setWordWrap(True)
    return lbl


def _icon_btn(icon: str, tip: str = "") -> QPushButton:
    btn = QPushButton(icon)
    btn.setObjectName("iconBtn")
    btn.setFixedSize(28, 28)
    btn.setToolTip(tip)
    return btn


def _divider(horizontal: bool = True) -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.Shape.HLine if horizontal else QFrame.Shape.VLine)
    line.setStyleSheet(f"color: {Colors.border}; background: {Colors.border};")
    line.setFixedHeight(1) if horizontal else line.setFixedWidth(1)
    return line


def _format_elapsed(secs: float | None) -> str:
    if secs is None:
        return "unknown"
    if secs < 60:
        return f"{secs:.2f}s"
    m = int(secs // 60)
    return f"{m}m {secs - m * 60:.1f}s"


# ══════════════════════════════════════════════════════════════════════
#  Chat message renderer
# ══════════════════════════════════════════════════════════════════════

def _render_user_msg(text: str) -> str:
    escaped = html.escape(text).replace("\n", "<br>")
    return f"""
<div style="margin:12px 0; display:flex; justify-content:flex-end;">
  <table width="100%" cellpadding="0" cellspacing="0"><tr>
    <td width="60%">&nbsp;</td>
    <td>
      <div style="background:{Colors.user_bubble}; border-radius:14px 14px 4px 14px;
                  padding:12px 16px; font-size:13px; color:{Colors.user_text};
                  line-height:1.55;">
        {escaped}
      </div>
    </td>
    <td width="12" style="padding-left:10px; vertical-align:top;">
      <div style="width:32px; height:32px; border-radius:50%;
                  background:{Colors.accent}; text-align:center;
                  line-height:32px; color:white; font-size:12px; font-weight:700;">
        You
      </div>
    </td>
  </tr></table>
</div>"""


def _render_assist_msg(text: str, citations: list[Citation], elapsed_seconds: float | None = None) -> str:
    escaped = html.escape(text).replace("\n", "<br>")
    cite_refs = ""
    if citations:
        refs = []
        for i, c in enumerate(citations, 1):
            refs.append(
                f'<span style="color:{Colors.text_link}; font-size:11px; '
                f'cursor:pointer;" title="{html.escape(c.filename)} p.{c.page_start}">'
                f'[{c.filename} &bull; p.{c.page_start} &bull; chunk {i}]</span>'
            )
        cite_refs = (
            f'<div style="margin-top:10px; padding-top:8px; '
            f'border-top:1px solid {Colors.border}; font-size:11px; '
            f'color:{Colors.text_muted}; line-height:2.0;">'
            + "  ".join(refs) + "</div>"
        )
    
    # Timing label at the bottom of assistant message
    time_label = ""
    if elapsed_seconds is not None:
        time_label = f'<div style="font-size:10px; color:{Colors.text_faint}; margin-top:4px;">⏱ {_format_elapsed(elapsed_seconds)}</div>'
        
    return f"""
<div style="margin:12px 0;">
  <table width="100%" cellpadding="0" cellspacing="0"><tr>
    <td width="12" style="padding-right:10px; vertical-align:top;">
      <div style="width:32px; height:32px; border-radius:50%;
                  background:{Colors.bg_card}; border:1px solid {Colors.border_light};
                  text-align:center; line-height:32px; font-size:14px;">🤖</div>
    </td>
    <td>
      <div style="background:{Colors.assist_bubble}; border-radius:4px 14px 14px 14px;
                  border:1px solid {Colors.border}; padding:14px 16px;
                  font-size:13px; color:{Colors.assist_text}; line-height:1.6;">
        {escaped}
        {cite_refs}
        {time_label}
      </div>
    </td>
    <td width="42">&nbsp;</td>
  </tr></table>
</div>"""


def _render_system_note(text: str) -> str:
    return (
        f'<div style="text-align:center; color:{Colors.text_faint}; '
        f'font-size:11px; margin:8px 0;">{html.escape(text)}</div>'
    )


# ══════════════════════════════════════════════════════════════════════
#  Citation card widget
# ══════════════════════════════════════════════════════════════════════

class CitationCard(QWidget):
    def __init__(self, citation: Citation, index: int, is_primary: bool = False) -> None:
        super().__init__()
        self.setObjectName("primaryCitationCard" if is_primary else "citationCard")
        self._build(citation, index, is_primary)

    def _build(self, c: Citation, idx: int, primary: bool) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(6)

        # ── top row: filename + primary badge ────────────────────────
        top = QHBoxLayout()
        top.setSpacing(8)
        icon = _label("📄", 12)
        top.addWidget(icon)
        name = _label(c.filename, 12, bold=True, color=Colors.text)
        name.setMaximumWidth(160)
        top.addWidget(name, 1)
        if primary:
            badge = _label("★  Primary source", 10, bold=True, color=Colors.accent)
            badge.setStyleSheet(
                f"color:{Colors.accent}; background:{Colors.accent_soft};"
                "border-radius:6px; padding:2px 8px; font-size:10px; font-weight:700;"
            )
            top.addWidget(badge)
        layout.addLayout(top)

        # ── page · chunk · relevance ─────────────────────────────────
        pages = (str(c.page_start) if c.page_start == c.page_end
                 else f"{c.page_start}–{c.page_end}")
        score = _guess_relevance(idx)
        rc = relevance_color(score)
        meta_row = QHBoxLayout()
        meta_row.setSpacing(6)
        meta_row.addWidget(_label(f"Page {pages}", 10, color=Colors.text_muted))
        meta_row.addWidget(_label("·", 10, color=Colors.text_faint))
        meta_row.addWidget(_label(f"Chunk {idx}", 10, color=Colors.text_muted))
        meta_row.addStretch(1)
        rel = _label(f"Relevance {score:.2f}", 10, bold=True, color=rc)
        rel.setStyleSheet(
            f"color:{rc}; background:{rc}22; border-radius:5px;"
            f"padding:1px 7px; font-size:10px; font-weight:700;"
        )
        meta_row.addWidget(rel)
        layout.addLayout(meta_row)

        # ── excerpt ───────────────────────────────────────────────────
        excerpt_short = c.excerpt[:180].replace("\n", " ")
        if len(c.excerpt) > 180:
            excerpt_short += "…"
        excerpt_lbl = _label(excerpt_short, 11, color=Colors.text_muted, wrap=True)
        layout.addWidget(excerpt_lbl)


def _guess_relevance(index: int) -> float:
    """Approximate relevance decreasing by rank when not available."""
    return max(0.50, 0.97 - (index - 1) * 0.08)


# ══════════════════════════════════════════════════════════════════════
#  Main window
# ══════════════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):
    def __init__(self, controller: DesktopController) -> None:
        super().__init__()
        self.controller = controller
        self.thread_pool = QThreadPool.globalInstance()
        self._chat_started = False
        self.setWindowTitle("Local PDF RAG")
        self.setMinimumSize(1100, 680)
        self._build_ui()
        self.refresh_documents()

    # ──────────────────────────────────────────────────────────────────
    # Top-level layout
    # ──────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QWidget()
        root_vbox = QVBoxLayout(root)
        root_vbox.setContentsMargins(0, 0, 0, 0)
        root_vbox.setSpacing(0)

        root_vbox.addWidget(self._build_header())

        body = QWidget()
        body_hbox = QHBoxLayout(body)
        body_hbox.setContentsMargins(0, 0, 0, 0)
        body_hbox.setSpacing(0)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(1)
        splitter.setChildrenCollapsible(False)

        sidebar = self._build_sidebar()
        sidebar.setObjectName("sidebar")
        sidebar.setMinimumWidth(220)
        sidebar.setMaximumWidth(400)

        chat = self._build_chat_area()
        chat.setObjectName("chatArea")

        sources = self._build_sources_panel()
        sources.setObjectName("sourcesPanel")
        sources.setMinimumWidth(300)
        sources.setMaximumWidth(480)

        splitter.addWidget(sidebar)
        splitter.addWidget(chat)
        splitter.addWidget(sources)
        splitter.setSizes([300, 720, 400])

        body_hbox.addWidget(splitter)
        root_vbox.addWidget(body, 1)

        self.setCentralWidget(root)

    # ──────────────────────────────────────────────────────────────────
    # Header bar
    # ──────────────────────────────────────────────────────────────────

    def _build_header(self) -> QWidget:
        bar = QWidget()
        bar.setObjectName("headerBar")
        bar.setFixedHeight(52)
        hbox = QHBoxLayout(bar)
        hbox.setContentsMargins(16, 0, 16, 0)
        hbox.setSpacing(10)

        # Logo + name
        logo = _label("📄", 18)
        app_name = _label("Local PDF RAG", 14, bold=True)
        app_name.setObjectName("appName")
        version = _label("v1.2.0", 10, color=Colors.text_muted)
        version.setObjectName("appVersion")

        hbox.addWidget(logo)
        hbox.addWidget(app_name)
        hbox.addWidget(version)
        hbox.addSpacing(12)

        # Status pill
        self._header_status = QLabel("● All good")
        self._header_status.setObjectName("statusPill")
        hbox.addWidget(self._header_status)
        hbox.addStretch(1)

        # Search bar
        search = QLineEdit()
        search.setObjectName("searchBar")
        search.setPlaceholderText("Search chats or documents…")
        search.setFixedWidth(260)
        hbox.addWidget(search)
        shortcut = _label("Ctrl+K", 10, color=Colors.text_faint)
        shortcut.setStyleSheet(
            f"color:{Colors.text_faint}; background:{Colors.bg_card};"
            f"border:1px solid {Colors.border}; border-radius:5px; padding:2px 6px;"
        )
        hbox.addWidget(shortcut)
        hbox.addSpacing(8)

        settings_btn = QPushButton("⚙  Settings")
        settings_btn.setObjectName("ghostBtn")
        settings_btn.clicked.connect(self.show_settings)
        hbox.addWidget(settings_btn)

        theme_btn = QPushButton("🌙  Theme")
        theme_btn.setObjectName("ghostBtn")
        hbox.addWidget(theme_btn)

        return bar

    # ──────────────────────────────────────────────────────────────────
    # Left sidebar
    # ──────────────────────────────────────────────────────────────────

    def _build_sidebar(self) -> QWidget:
        sidebar = QWidget()
        vbox = QVBoxLayout(sidebar)
        vbox.setContentsMargins(12, 14, 12, 14)
        vbox.setSpacing(10)

        # New Chat + Import PDFs
        top_btns = QHBoxLayout()
        new_chat = QPushButton("＋  New Chat")
        new_chat.setObjectName("primaryBtn")
        new_chat.clicked.connect(self._new_chat)
        import_btn = QPushButton("⬆  Import PDFs")
        import_btn.setObjectName("ghostBtn")
        import_btn.clicked.connect(self.import_documents)
        top_btns.addWidget(new_chat, 2)
        top_btns.addWidget(import_btn, 1)
        vbox.addLayout(top_btns)
        vbox.addSpacing(4)

        # Workspaces heading
        ws_row = QHBoxLayout()
        ws_lbl = _label("WORKSPACES", 10, color=Colors.text_faint)
        ws_lbl.setObjectName("sectionLabel")
        ws_row.addWidget(ws_lbl)
        ws_row.addStretch(1)
        add_ws = _icon_btn("＋", "New workspace")
        add_ws.clicked.connect(self._new_chat)
        ws_row.addWidget(add_ws)
        vbox.addLayout(ws_row)

        # Document tree
        self.doc_tree = QTreeWidget()
        self.doc_tree.setHeaderHidden(True)
        self.doc_tree.setIndentation(16)
        self.doc_tree.setAnimated(True)
        self.doc_tree.setColumnCount(1)
        self.doc_tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.doc_tree.customContextMenuRequested.connect(self._on_tree_context_menu)
        self.doc_tree.itemClicked.connect(self._on_tree_item_clicked)
        vbox.addWidget(self.doc_tree, 1)

        vbox.addWidget(_divider())

        # Local Data Status
        status_lbl = _label("Local Data Status", 11, bold=True, color=Colors.text_muted)
        vbox.addWidget(status_lbl)
        info = _label("All data is stored locally on this device.", 10, color=Colors.text_faint, wrap=True)
        vbox.addWidget(info)

        self._stat_docs   = self._stat_card("Documents", "—")
        self._stat_chunks = self._stat_card("Indexed Chunks", "—")
        self._stat_size   = self._stat_card("Storage Used", "—")
        stat_row = QHBoxLayout()
        stat_row.setSpacing(6)
        for card in (self._stat_docs, self._stat_chunks, self._stat_size):
            stat_row.addWidget(card)
        vbox.addLayout(stat_row)

        vbox.addWidget(_divider())

        # Bottom: Ollama + model
        bottom = QHBoxLayout()
        bottom.setSpacing(8)
        self._ollama_dot = _label("●", 12, color=Colors.success)
        self._ollama_lbl = _label("Ollama connected", 11, color=Colors.text_muted)
        bottom.addWidget(self._ollama_dot)
        bottom.addWidget(self._ollama_lbl, 1)
        model_col = QVBoxLayout()
        model_col.setSpacing(0)
        model_col.addWidget(_label("Model", 9, color=Colors.text_faint))
        self._model_lbl = _label("—", 11, bold=True, color=Colors.text)
        model_col.addWidget(self._model_lbl)
        bottom.addLayout(model_col)
        gear = _icon_btn("⚙", "Settings")
        gear.clicked.connect(self.show_settings)
        bottom.addWidget(gear)
        vbox.addLayout(bottom)

        return sidebar

    def _stat_card(self, label: str, value: str) -> QWidget:
        card = QWidget()
        card.setObjectName("statCard")
        col = QVBoxLayout(card)
        col.setContentsMargins(8, 8, 8, 8)
        col.setSpacing(2)
        col.addWidget(_label(label, 9, color=Colors.text_faint, wrap=True))
        val_lbl = _label(value, 16, bold=True, color=Colors.text)
        col.addWidget(val_lbl)
        card._value_label = val_lbl   # type: ignore[attr-defined]
        return card

    # ──────────────────────────────────────────────────────────────────
    # Chat area
    # ──────────────────────────────────────────────────────────────────

    def _build_chat_area(self) -> QWidget:
        area = QWidget()
        vbox = QVBoxLayout(area)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(0)

        vbox.addWidget(self._build_chat_titlebar())
        vbox.addWidget(_divider())
        vbox.addWidget(self._build_chat_messages(), 1)
        vbox.addWidget(_divider())
        vbox.addWidget(self._build_chat_input())

        return area

    def _build_chat_titlebar(self) -> QWidget:
        bar = QWidget()
        bar.setFixedHeight(52)
        hbox = QHBoxLayout(bar)
        hbox.setContentsMargins(20, 0, 16, 0)
        hbox.setSpacing(10)

        self._chat_title_lbl = _label("New Chat", 16, bold=True)
        hbox.addWidget(self._chat_title_lbl)
        edit = _icon_btn("✏", "Rename")
        edit.clicked.connect(self._rename_active_session)
        hbox.addWidget(edit)
        hbox.addSpacing(16)

        # Status pills
        self._pill_offline   = self._status_pill("Offline",         Colors.success, Colors.success_soft)
        self._pill_ollama    = self._status_pill("Ollama connected", Colors.info,    Colors.info_soft)
        self._pill_model     = self._status_pill("Model: —",        Colors.text_muted, Colors.bg_card)
        for p in (self._pill_offline, self._pill_ollama, self._pill_model):
            hbox.addWidget(p)
        hbox.addStretch(1)

        more = _icon_btn("⋯", "More options")
        hbox.addWidget(more)
        return bar

    @staticmethod
    def _status_pill(text: str, fg: str, bg: str) -> QLabel:
        pill = QLabel(text)
        pill.setStyleSheet(
            f"color:{fg}; background:{bg}; border-radius:9px;"
            f"padding:3px 10px; font-size:11px; font-weight:600;"
        )
        return pill

    def _build_chat_messages(self) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        container = QWidget()
        container.setStyleSheet(f"background:{Colors.bg_panel};")
        vbox = QVBoxLayout(container)
        vbox.setContentsMargins(24, 16, 24, 8)
        vbox.setSpacing(0)
        vbox.setAlignment(Qt.AlignmentFlag.AlignTop)

        self._messages_view = QTextBrowser()
        self._messages_view.setOpenExternalLinks(False)
        self._messages_view.setStyleSheet(
            f"background:transparent; border:none; font-size:13px; color:{Colors.text};"
        )
        self._messages_view.setHtml(self._empty_chat_html())
        vbox.addWidget(self._messages_view, 1)

        scroll.setWidget(container)
        self._chat_scroll = scroll
        return scroll

    def _build_chat_input(self) -> QWidget:
        outer = QWidget()
        outer.setStyleSheet(f"background:{Colors.bg_panel};")
        outer_vbox = QVBoxLayout(outer)
        outer_vbox.setContentsMargins(24, 12, 24, 8)
        outer_vbox.setSpacing(6)

        # Input box
        box = QWidget()
        box.setObjectName("chatInputBox")
        box_hbox = QHBoxLayout(box)
        box_hbox.setContentsMargins(14, 10, 10, 10)
        box_hbox.setSpacing(8)

        attach = _icon_btn("📄", "Attach file")
        attach.clicked.connect(self.import_documents)
        box_hbox.addWidget(attach)

        self.question_input = QLineEdit()
        self.question_input.setObjectName("chatInput")
        self.question_input.setPlaceholderText("Ask a question about the selected documents…")
        self.question_input.returnPressed.connect(self.ask_question)
        box_hbox.addWidget(self.question_input, 1)

        self.send_btn = QPushButton("➔")
        self.send_btn.setObjectName("sendBtn")
        self.send_btn.clicked.connect(self.ask_question)
        box_hbox.addWidget(self.send_btn)

        expand = _icon_btn("⤢", "Expand input")
        box_hbox.addWidget(expand)

        outer_vbox.addWidget(box)

        hint = _label(
            "Enter to send  •  Shift+Enter for new line",
            10, color=Colors.text_faint
        )
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        outer_vbox.addWidget(hint)

        footer = _label(
            "Responses are generated by AI and may contain inaccuracies.",
            10, color=Colors.text_faint
        )
        footer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        outer_vbox.addWidget(footer)

        return outer

    # ──────────────────────────────────────────────────────────────────
    # Sources / citations panel
    # ──────────────────────────────────────────────────────────────────

    def _build_sources_panel(self) -> QWidget:
        panel = QWidget()
        vbox = QVBoxLayout(panel)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(0)

        # Header
        hdr = QWidget()
        hdr.setFixedHeight(52)
        hbox = QHBoxLayout(hdr)
        hbox.setContentsMargins(16, 0, 16, 0)
        hbox.setSpacing(10)
        src_title = _label("Sources", 15, bold=True)
        hbox.addWidget(src_title)
        hbox.addStretch(1)
        filter_btn = _icon_btn("🔍", "Filter")
        grid_btn   = _icon_btn("⊞", "Grid view")
        hbox.addWidget(filter_btn)
        hbox.addWidget(grid_btn)
        vbox.addWidget(hdr)
        vbox.addWidget(_divider())

        # Count + sort row
        meta = QWidget()
        meta.setFixedHeight(40)
        mhbox = QHBoxLayout(meta)
        mhbox.setContentsMargins(16, 0, 16, 0)
        self._cite_count_lbl = _label("0 citations", 11, color=Colors.text_muted)
        mhbox.addWidget(self._cite_count_lbl)
        mhbox.addStretch(1)
        sort_box = QComboBox()
        sort_box.addItems(["Sort by relevance", "Sort by page", "Sort by file"])
        mhbox.addWidget(sort_box)
        vbox.addWidget(meta)
        vbox.addWidget(_divider())

        # Scroll area for cards
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet(f"background:{Colors.bg_sidebar};")

        self._sources_container = QWidget()
        self._sources_container.setStyleSheet(f"background:{Colors.bg_sidebar};")
        self._sources_vbox = QVBoxLayout(self._sources_container)
        self._sources_vbox.setContentsMargins(12, 12, 12, 12)
        self._sources_vbox.setSpacing(8)
        self._sources_vbox.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._sources_vbox.addWidget(self._empty_sources_widget())
        scroll.setWidget(self._sources_container)
        vbox.addWidget(scroll, 1)

        vbox.addWidget(_divider())

        # Export button
        export_row = QWidget()
        export_row.setFixedHeight(52)
        ehbox = QHBoxLayout(export_row)
        ehbox.setContentsMargins(12, 8, 12, 8)
        export_btn = QPushButton("⬆  Export citations")
        export_btn.setObjectName("ghostBtn")
        export_btn.setFixedHeight(34)
        ehbox.addWidget(export_btn)
        vbox.addWidget(export_row)

        return panel

    # ──────────────────────────────────────────────────────────────────
    # Data refresh helpers
    # ──────────────────────────────────────────────────────────────────

    def refresh_documents(self) -> None:
        self.doc_tree.blockSignals(True)
        self.doc_tree.clear()
        sessions = self.controller.list_sessions()
        active_id = self.controller.active_session_id()

        for session in sessions:
            doc_count = self.controller.session_document_count(session.session_id)
            ws_item = QTreeWidgetItem(self.doc_tree)
            ws_item.setText(0, f"💬  {session.title}  ({doc_count})")
            ws_item.setData(0, _TREE_SESSION_ROLE, session.session_id)
            ws_item.setData(0, _TREE_TYPE_ROLE, "session")
            ws_item.setToolTip(0, session.title)

            if session.session_id == active_id:
                ws_item.setFont(0, self._bold_font(12))
                ws_item.setForeground(0, self._qcolor(Colors.text))
                ws_item.setExpanded(True)
                self.doc_tree.setCurrentItem(ws_item)
                self._chat_title_lbl.setText(session.title)
            else:
                ws_item.setFont(0, self._regular_font(12))
                ws_item.setForeground(0, self._qcolor(Colors.text_muted))

            docs = self.controller.service.store.list_documents(session.session_id)
            for doc in docs:
                _bg, fg, status_text = status_badge(doc.status)
                doc_item = QTreeWidgetItem(ws_item)
                icon = "✅" if doc.status == "ready" else "⏳" if doc.status == "processing" else "❌"
                doc_item.setText(0, f"  📄  {doc.filename}  {icon}")
                doc_item.setFont(0, self._regular_font(11))
                doc_item.setToolTip(0, f"Status: {status_text}  |  {doc.path}")
                doc_item.setData(0, _TREE_SESSION_ROLE, doc.document_id)
                doc_item.setData(0, _TREE_TYPE_ROLE, "document")
                
                if doc.status == "failed":
                    doc_item.setForeground(0, self._qcolor(Colors.danger))
                elif doc.status == "ready":
                    doc_item.setForeground(0, self._qcolor(Colors.success))
                else:
                    doc_item.setForeground(0, self._qcolor(Colors.warning))

        self.doc_tree.blockSignals(False)
        self._update_status()

    def _on_tree_item_clicked(self, item: QTreeWidgetItem, _col: int) -> None:
        kind = item.data(0, _TREE_TYPE_ROLE)
        if kind != "session":
            return
        session_id = item.data(0, _TREE_SESSION_ROLE)
        if session_id == self.controller.active_session_id():
            return
        self.controller.set_active_session(session_id)
        self._chat_started = False
        self._messages_view.setHtml(self._empty_chat_html())
        self._clear_sources()
        self.refresh_documents()

    def _on_tree_context_menu(self, pos: object) -> None:
        item = self.doc_tree.itemAt(pos)
        if not item:
            return
        kind = item.data(0, _TREE_TYPE_ROLE)
        menu = QMenu(self)
        menu.setStyleSheet(
            f"QMenu {{ background-color: {Colors.bg_card}; color: {Colors.text}; border: 1px solid {Colors.border}; }}"
            f"QMenu::item:selected {{ background-color: {Colors.bg_selected}; }}"
        )

        if kind == "session":
            session_id = item.data(0, _TREE_SESSION_ROLE)
            title = item.toolTip(0)
            rename_act = menu.addAction("✏️  Rename Chat")
            menu.addSeparator()
            delete_act = menu.addAction("🗑️  Delete Chat")
            chosen = menu.exec(self.doc_tree.viewport().mapToGlobal(pos))
            if chosen == rename_act:
                self._rename_session(session_id, title)
            elif chosen == delete_act:
                self._delete_session(session_id, title)

        elif kind == "document":
            document_id = item.data(0, _TREE_SESSION_ROLE)
            filename = item.toolTip(0).split(" — ")[0]
            remove_act = menu.addAction("🗑️  Remove Document")
            chosen = menu.exec(self.doc_tree.viewport().mapToGlobal(pos))
            if chosen == remove_act:
                self._delete_document(document_id, filename)

    def _rename_session(self, session_id: str, current_title: str) -> None:
        new_title, ok = QInputDialog.getText(
            self, "Rename Chat", "Chat name:", QLineEdit.EchoMode.Normal, current_title
        )
        if ok and new_title.strip():
            self.controller.rename_session(session_id, new_title.strip())
            if session_id == self.controller.active_session_id():
                self._chat_title_lbl.setText(new_title.strip())
            self.refresh_documents()

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
        self._messages_view.setHtml(self._empty_chat_html())
        self._clear_sources()
        self.refresh_documents()

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
        self._ensure_chat_started()
        self._append_html(_render_system_note(f"Removed: {filename}"))
        self.refresh_documents()

    def _rename_active_session(self) -> None:
        session_id = self.controller.active_session_id()
        current_title = self._chat_title_lbl.text()
        self._rename_session(session_id, current_title)

    def _update_status(self) -> None:
        status = self.controller.status()
        ready = bool(status["ollama_ready"])

        # Header pill
        if ready:
            self._header_status.setText("● All good")
            self._header_status.setStyleSheet(
                f"color:{Colors.success}; background:{Colors.success_soft};"
                "border-radius:8px; padding:2px 10px; font-size:11px; font-weight:600;"
            )
        else:
            self._header_status.setText("● Setup needed")
            self._header_status.setStyleSheet(
                f"color:{Colors.warning}; background:{Colors.warning_soft};"
                "border-radius:8px; padding:2px 10px; font-size:11px; font-weight:600;"
            )

        # Stat cards
        n_docs   = status["documents"]
        n_chunks = status["chunks"]
        self._stat_docs._value_label.setText(str(n_docs))       # type: ignore[attr-defined]
        self._stat_chunks._value_label.setText(f"{n_chunks:,}") # type: ignore[attr-defined]
        
        # Calculate storage used
        try:
            import os
            data_dir = status["data_dir"]
            total_bytes = 0
            for root_dir, _, files in os.walk(data_dir):
                for f in files:
                    fp = os.path.join(root_dir, f)
                    if os.path.exists(fp):
                        total_bytes += os.path.getsize(fp)
            if total_bytes < 1024:
                size_str = f"{total_bytes} B"
            elif total_bytes < 1024 * 1024:
                size_str = f"{total_bytes / 1024:.1f} KB"
            else:
                size_str = f"{total_bytes / (1024 * 1024):.1f} MB"
        except Exception:
            size_str = "—"
        self._stat_size._value_label.setText(size_str) # type: ignore[attr-defined]

        # Bottom Ollama row
        dot_color = Colors.success if ready else Colors.warning
        self._ollama_dot.setStyleSheet(f"color:{dot_color}; font-size:12px;")
        self._ollama_lbl.setText("Ollama connected" if ready else "Ollama offline")
        model_name = self.controller.settings.active_model
        self._model_lbl.setText(model_name)

        # Chat title-bar pills
        self._pill_ollama.setText("Ollama connected" if ready else "Ollama offline")
        self._pill_ollama.setStyleSheet(
            f"color:{Colors.info if ready else Colors.warning};"
            f"background:{Colors.info_soft if ready else Colors.warning_soft};"
            "border-radius:9px; padding:3px 10px; font-size:11px; font-weight:600;"
        )
        self._pill_model.setText(f"Model: {model_name}")

    # ──────────────────────────────────────────────────────────────────
    # Actions
    # ──────────────────────────────────────────────────────────────────

    def _new_chat(self) -> None:
        session = self.controller.create_session()
        self._chat_started = False
        self._messages_view.setHtml(self._empty_chat_html())
        self._clear_sources()
        self.refresh_documents()
        self.question_input.setFocus()

        # Prompt for files immediately
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
            self._messages_view.setHtml(self._empty_chat_html())
            self._clear_sources()
            self.refresh_documents()
            return

        for path in paths:
            name = Path(path).name
            
            def _make_cb(doc_name: str):
                def _cb(done: int, total: int, msg: str) -> None:
                    pct = int(100 * done / total) if total else 0
                    self._pill_offline.setText(f"⏳ Ingesting {doc_name} ({pct}%)")
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
            self, "Import PDFs or text files", str(Path.home()),
            "Documents (*.pdf *.txt *.md);;All files (*.*)",
        )
        for path in paths:
            name = Path(path).name
            
            def _make_cb(doc_name: str):
                def _cb(done: int, total: int, msg: str) -> None:
                    pct = int(100 * done / total) if total else 0
                    self._pill_offline.setText(f"⏳ Ingesting {doc_name} ({pct}%)")
                return _cb
                
            cb = _make_cb(name)
            self._run_worker(
                f"Ingesting {name}…",
                self.controller.ingest,
                path,
                False,
                cb,
            )

    def repair_stuck_imports(self) -> None:
        self._run_worker("Repairing stuck imports…", self.controller.repair_unready_documents)

    def ask_question(self) -> None:
        question = self.question_input.text().strip()
        if not question:
            return
        status = self.controller.status()
        if status["documents"] and not status["chunks"]:
            QMessageBox.warning(
                self, "No Searchable Text Yet",
                "Documents are listed but have no indexed chunks yet.\n\n"
                "Use Settings → Repair Stuck Imports or re-import the PDF.",
            )
            return
        self.question_input.clear()
        self._ensure_chat_started()
        self._append_html(_render_user_msg(question))
        self._run_worker("Answering…", self.controller.ask, question, False)

    def show_settings(self) -> None:
        readiness = self.controller.model_readiness()
        dialog = QDialog(self)
        dialog.setWindowTitle("Settings")
        dialog.setMinimumWidth(560)
        vbox = QVBoxLayout(dialog)
        vbox.setSpacing(14)

        form = QFormLayout()
        form.setSpacing(10)
        use_ollama = QCheckBox("Enable Ollama for local generation and embeddings")
        use_ollama.setChecked(self.controller.settings.use_ollama)
        ollama_url = QLineEdit(self.controller.settings.ollama_base_url)
        ollama_url.setStyleSheet(
            f"background:{Colors.bg_input}; border:1px solid {Colors.border};"
            f"border-radius:8px; padding:6px 10px; color:{Colors.text};"
        )
        active_model = QComboBox()
        active_model.setEditable(True)
        embedding_model = QComboBox()
        embedding_model.setEditable(True)

        available = readiness.available_models
        if not available and readiness.ollama_reachable:
            try:
                available = self.controller.available_ollama_models()
            except OSError:
                available = []
        for m in available:
            active_model.addItem(m)
            embedding_model.addItem(m)
        self._set_combo(active_model,    self.controller.settings.active_model)
        self._set_combo(embedding_model, self.controller.settings.embedding_model)

        form.addRow("Ollama", use_ollama)
        form.addRow("Ollama URL", ollama_url)
        form.addRow("Generation model", active_model)
        form.addRow("Embedding model", embedding_model)
        form.addRow("Ollama ready",
                    _label("Yes" if readiness.ready else "No", 12, bold=True,
                           color=Colors.success if readiness.ready else Colors.danger))
        vbox.addLayout(form)

        msg_lbl = _label(readiness.message, 11, color=Colors.text_muted, wrap=True)
        vbox.addWidget(msg_lbl)

        if readiness.setup_commands:
            cmd = QPlainTextEdit("\n".join(readiness.setup_commands))
            cmd.setReadOnly(True)
            cmd.setFixedHeight(90)
            vbox.addWidget(cmd)

        btns = QHBoxLayout()
        refresh_btn = QPushButton("Refresh models")
        refresh_btn.clicked.connect(
            lambda: self._refresh_combos(ollama_url.text().strip(), active_model, embedding_model)
        )
        repair_btn = QPushButton("Repair stuck imports")
        repair_btn.clicked.connect(lambda: (dialog.accept(), self.repair_stuck_imports()))
        save_btn = QPushButton("Save")
        save_btn.setObjectName("primaryBtn")
        save_btn.clicked.connect(
            lambda: self._save_settings(
                dialog, use_ollama.isChecked(), ollama_url.text().strip(),
                active_model.currentText().strip(), embedding_model.currentText().strip(),
            )
        )
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(dialog.reject)
        btns.addWidget(refresh_btn)
        btns.addWidget(repair_btn)
        btns.addStretch(1)
        btns.addWidget(save_btn)
        btns.addWidget(cancel_btn)
        vbox.addLayout(btns)
        dialog.exec()

    def _save_settings(self, dialog, use_ollama, url, model, emb_model):
        if not model or not emb_model:
            QMessageBox.warning(self, "Settings", "Select both a generation model and an embedding model.")
            return
        self.controller.update_preferences(
            use_ollama=use_ollama,
            ollama_base_url=url or "http://localhost:11434",
            active_model=model,
            embedding_model=emb_model,
        )
        self.refresh_documents()
        dialog.accept()

    def _refresh_combos(self, url, active_combo, emb_combo):
        try:
            models = self.controller.available_ollama_models(url)
        except OSError as e:
            QMessageBox.warning(self, "Ollama", f"Could not reach Ollama:\n{e}")
            return
        cur_a, cur_e = active_combo.currentText(), emb_combo.currentText()
        active_combo.clear(); emb_combo.clear()
        for m in models:
            active_combo.addItem(m); emb_combo.addItem(m)
        self._set_combo(active_combo, cur_a)
        self._set_combo(emb_combo,    cur_e)

    # ──────────────────────────────────────────────────────────────────
    # Background worker wiring
    # ──────────────────────────────────────────────────────────────────

    def _run_worker(self, label, fn, *args):
        worker = FunctionWorker(label, fn, *args)
        worker.signals.started.connect(self._on_worker_start)
        worker.signals.finished.connect(self._on_worker_done)
        worker.signals.error.connect(self._on_worker_error)
        self.thread_pool.start(worker)

    def _on_worker_start(self, label: str) -> None:
        self._pill_offline.setText(f"⏳ {label}")
        self._pill_offline.setStyleSheet(
            f"color:{Colors.warning}; background:{Colors.warning_soft};"
            "border-radius:9px; padding:3px 10px; font-size:11px; font-weight:600;"
        )
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)

    def _on_worker_done(self, result: object) -> None:
        elapsed: float | None = None
        if isinstance(result, WorkerResult):
            elapsed = result.elapsed_seconds
            result = result.result

        QApplication.restoreOverrideCursor()
        self._pill_offline.setText("Offline")
        self._pill_offline.setStyleSheet(
            f"color:{Colors.success}; background:{Colors.success_soft};"
            "border-radius:9px; padding:3px 10px; font-size:11px; font-weight:600;"
        )
        if isinstance(result, Answer):
            self._show_answer(result, elapsed)
        elif isinstance(result, Document):
            self._ensure_chat_started()
            self._append_html(_render_system_note(f"Imported {result.filename}"))
            self.refresh_documents()
        elif isinstance(result, list):
            self._ensure_chat_started()
            self._append_html(_render_system_note(f"Repaired {len(result)} document(s)."))
            self.refresh_documents()
        else:
            self._update_status()

    def _on_worker_error(self, message: str) -> None:
        QApplication.restoreOverrideCursor()
        self._pill_offline.setText("Error")
        self._pill_offline.setStyleSheet(
            f"color:{Colors.danger}; background:{Colors.danger_soft};"
            "border-radius:9px; padding:3px 10px; font-size:11px; font-weight:600;"
        )
        QMessageBox.critical(self, "Error", self._friendly_error(message))

    @staticmethod
    def _friendly_error(msg: str) -> str:
        if "PyMuPDF" in msg:
            return "PDF parsing unavailable.\n\nFix: py -m pip install -e .[desktop]\n\n" + msg
        if "no searchable text" in msg:
            return ("No text found in this PDF.\n\nFor scanned PDFs:\n"
                     "  py -m pip install -e .[pdf]\n  RAG_FORCE_OCR=1\n\n" + msg)
        return msg

    # ──────────────────────────────────────────────────────────────────
    # Chat rendering
    # ──────────────────────────────────────────────────────────────────

    def _empty_chat_html(self) -> str:
        return (
            f"<div style='text-align:center; padding:60px 20px; color:{Colors.text_faint};'>"
            "<div style='font-size:40px;'>📄</div>"
            f"<div style='font-size:16px; font-weight:700; color:{Colors.text_muted}; margin:12px 0 6px;'>"
            "Ask anything about your documents</div>"
            f"<div style='font-size:12px;'>Import a PDF and start asking questions.</div>"
            "</div>"
        )

    def _ensure_chat_started(self) -> None:
        if not self._chat_started:
            self._messages_view.setHtml("")
            self._chat_started = True

    def _append_html(self, fragment: str) -> None:
        self._messages_view.append(fragment)
        sb = self._messages_view.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _show_answer(self, answer: Answer, elapsed: float | None = None) -> None:
        self._ensure_chat_started()
        self._append_html(_render_assist_msg(answer.answer, answer.citations, elapsed_seconds=elapsed))
        self._render_citations(answer.citations)
        self._update_status()

    # ──────────────────────────────────────────────────────────────────
    # Sources panel rendering
    # ──────────────────────────────────────────────────────────────────

    def _clear_sources(self) -> None:
        while self._sources_vbox.count():
            item = self._sources_vbox.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._sources_vbox.addWidget(self._empty_sources_widget())
        self._cite_count_lbl.setText("0 citations")

    def _empty_sources_widget(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(8, 32, 8, 32)
        lbl = _label("Citations for your next\nanswer will appear here.",
                     12, color=Colors.text_faint, wrap=True)
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        v.addWidget(lbl)
        return w

    def _render_citations(self, citations: list[Citation]) -> None:
        while self._sources_vbox.count():
            item = self._sources_vbox.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if not citations:
            self._sources_vbox.addWidget(self._empty_sources_widget())
            self._cite_count_lbl.setText("0 citations")
            return

        self._cite_count_lbl.setText(
            f"{len(citations)} citation{'s' if len(citations) != 1 else ''}"
        )
        for i, citation in enumerate(citations, 1):
            card = CitationCard(citation, i, is_primary=(i == 1))
            self._sources_vbox.addWidget(card)
        self._sources_vbox.addStretch(1)

    # ──────────────────────────────────────────────────────────────────
    # Tiny helpers
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _set_combo(combo: QComboBox, text: str) -> None:
        idx = combo.findText(text)
        if idx >= 0:
            combo.setCurrentIndex(idx)
        else:
            combo.addItem(text)
            combo.setCurrentText(text)

    @staticmethod
    def _bold_font(size: int) -> QFont:
        f = QFont()
        f.setPointSize(size)
        f.setBold(True)
        return f

    @staticmethod
    def _regular_font(size: int) -> QFont:
        f = QFont()
        f.setPointSize(size)
        return f

    @staticmethod
    def _qcolor(hex_color: str):
        from PySide6.QtGui import QColor
        return QColor(hex_color)
