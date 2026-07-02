"""Dark design system — mirrors the reference UI screenshot.

Single source of truth for every color token and the global QSS stylesheet.
Import Colors and STYLESHEET; never hardcode hex values elsewhere.
"""
from __future__ import annotations


class Colors:
    # ── Backgrounds ───────────────────────────────────────────────────
    bg_base      = "#0f1117"   # outermost window
    bg_sidebar   = "#161b27"   # left sidebar
    bg_panel     = "#1a1f2e"   # main panels / cards
    bg_card      = "#1e2437"   # citation cards, message bubbles
    bg_input     = "#252d3d"   # text inputs
    bg_hover     = "#242b3d"   # hover state
    bg_selected  = "#2a3350"   # selected list item

    # ── Borders ───────────────────────────────────────────────────────
    border       = "#2a3350"
    border_light = "#323d58"

    # ── Text ──────────────────────────────────────────────────────────
    text         = "#e2e8f0"
    text_muted   = "#8892a4"
    text_faint   = "#4a5568"
    text_link    = "#7c9ef8"

    # ── Accent (indigo) ───────────────────────────────────────────────
    accent       = "#4f6ef7"
    accent_hover = "#3b5bdb"
    accent_soft  = "#1e2d5c"

    # ── Semantic ──────────────────────────────────────────────────────
    success      = "#34d399"
    success_soft = "#0d2e22"
    warning      = "#fbbf24"
    warning_soft = "#2e2200"
    danger       = "#f87171"
    danger_soft  = "#2e0d0d"
    info         = "#60a5fa"
    info_soft    = "#0d1f3c"

    # ── Relevance score chips ─────────────────────────────────────────
    rel_high     = "#34d399"   # ≥ 0.90
    rel_med      = "#fbbf24"   # ≥ 0.75
    rel_low      = "#f87171"   # < 0.75

    # ── Chat bubbles ──────────────────────────────────────────────────
    user_bubble    = "#1e2d5c"
    user_text      = "#c7d7fd"
    assist_bubble  = "#1e2437"
    assist_text    = "#e2e8f0"


STYLESHEET = f"""
/* ── Global ─────────────────────────────────────────────────────── */
* {{
    font-family: "Segoe UI", "Inter", "Helvetica Neue", Arial, sans-serif;
    color: {Colors.text};
    outline: none;
}}
QMainWindow, QWidget {{
    background-color: {Colors.bg_base};
}}
QWidget#sidebar {{
    background-color: {Colors.bg_sidebar};
    border-right: 1px solid {Colors.border};
}}
QWidget#chatArea {{
    background-color: {Colors.bg_panel};
}}
QWidget#sourcesPanel {{
    background-color: {Colors.bg_sidebar};
    border-left: 1px solid {Colors.border};
}}

/* ── Header bar ─────────────────────────────────────────────────── */
QWidget#headerBar {{
    background-color: {Colors.bg_sidebar};
    border-bottom: 1px solid {Colors.border};
}}
QLabel#appName {{
    font-size: 15px;
    font-weight: 700;
    color: {Colors.text};
}}
QLabel#appVersion {{
    font-size: 10px;
    color: {Colors.text_muted};
    padding: 3px 8px;
    background: {Colors.bg_card};
    border-radius: 8px;
    border: 1px solid {Colors.border_light};
}}
QLabel#statusPill {{
    font-size: 11px;
    font-weight: 600;
    color: {Colors.success};
    padding: 3px 10px;
    background: {Colors.success_soft};
    border-radius: 10px;
}}

/* ── Buttons ────────────────────────────────────────────────────── */
QPushButton {{
    background-color: {Colors.bg_card};
    border: 1px solid {Colors.border_light};
    border-radius: 8px;
    padding: 7px 14px;
    font-size: 12px;
    font-weight: 600;
    color: {Colors.text};
    min-height: 18px;
}}
QPushButton:hover {{
    background-color: {Colors.bg_hover};
    border-color: {Colors.accent};
    color: {Colors.text};
}}
QPushButton:pressed {{
    background-color: {Colors.accent_soft};
}}
QPushButton:disabled {{
    color: {Colors.text_faint};
}}
QPushButton#primaryBtn {{
    background-color: {Colors.accent};
    border: 1px solid {Colors.accent};
    color: white;
    font-size: 13px;
    font-weight: 700;
    border-radius: 9px;
    padding: 8px 16px;
    min-height: 20px;
}}
QPushButton#primaryBtn:hover {{
    background-color: {Colors.accent_hover};
    border-color: {Colors.accent_hover};
}}
QPushButton#ghostBtn {{
    background-color: transparent;
    border: 1px solid {Colors.border_light};
    color: {Colors.text_muted};
    padding: 7px 14px;
    font-size: 12px;
    min-height: 18px;
}}
QPushButton#ghostBtn:hover {{
    background-color: {Colors.bg_hover};
    color: {Colors.text};
    border-color: {Colors.border_light};
}}
QPushButton#headerBtn {{
    background-color: transparent;
    border: 1px solid {Colors.border_light};
    color: {Colors.text_muted};
    padding: 6px 14px;
    font-size: 12px;
    font-weight: 600;
    border-radius: 8px;
    min-height: 16px;
}}
QPushButton#headerBtn:hover {{
    background-color: {Colors.bg_hover};
    color: {Colors.text};
}}
QPushButton#iconBtn {{
    background-color: transparent;
    border: none;
    color: {Colors.text_muted};
    padding: 4px;
    font-size: 13px;
    min-height: 0px;
}}
QPushButton#iconBtn:hover {{
    color: {Colors.text};
    background-color: {Colors.bg_hover};
    border-radius: 6px;
}}
QPushButton#sendBtn {{
    background-color: {Colors.accent};
    border: none;
    border-radius: 12px;
    padding: 0px;
    color: white;
    font-size: 16px;
    font-weight: 700;
    min-width: 48px;
    min-height: 48px;
    max-width: 48px;
    max-height: 48px;
}}
QPushButton#sendBtn:hover {{
    background-color: {Colors.accent_hover};
}}
QPushButton#expandBtn {{
    background-color: transparent;
    border: 1px solid {Colors.border_light};
    border-radius: 10px;
    padding: 0px;
    color: {Colors.text_muted};
    font-size: 14px;
    min-width: 32px;
    min-height: 32px;
    max-width: 32px;
    max-height: 32px;
}}
QPushButton#expandBtn:hover {{
    color: {Colors.text};
    background-color: {Colors.bg_hover};
}}

/* ── Search bar ─────────────────────────────────────────────────── */
QLineEdit#searchBar {{
    background-color: {Colors.bg_input};
    border: 1px solid {Colors.border};
    border-radius: 9px;
    padding: 8px 14px;
    font-size: 12px;
    color: {Colors.text};
    min-width: 240px;
}}
QLineEdit#searchBar:focus {{
    border-color: {Colors.accent};
}}
QLineEdit#chatInput {{
    background-color: transparent;
    border: none;
    font-size: 13px;
    color: {Colors.text};
    padding: 6px 0px;
}}
QLineEdit#chatInput::placeholder {{
    color: {Colors.text_faint};
}}

/* ── Sidebar labels ─────────────────────────────────────────────── */
QLabel#sectionLabel {{
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1px;
    color: {Colors.text_faint};
    padding: 6px 0px 4px 0px;
}}

/* ── Workspace / document tree ──────────────────────────────────── */
QTreeWidget {{
    background-color: transparent;
    border: none;
    outline: none;
    font-size: 12px;
}}
QTreeWidget::item {{
    padding: 5px 6px;
    border-radius: 6px;
    color: {Colors.text_muted};
    min-height: 20px;
}}
QTreeWidget::item:hover {{
    background-color: {Colors.bg_hover};
    color: {Colors.text};
}}
QTreeWidget::item:selected {{
    background-color: {Colors.bg_selected};
    color: {Colors.text};
}}
QTreeWidget::branch {{
    background-color: transparent;
}}

/* ── Stat cards ─────────────────────────────────────────────────── */
QWidget#statCard {{
    background-color: {Colors.bg_card};
    border: 1px solid {Colors.border};
    border-radius: 8px;
    min-width: 70px;
}}

/* ── Chat view ──────────────────────────────────────────────────── */
QTextBrowser {{
    background-color: transparent;
    border: none;
    font-size: 13px;
    line-height: 1.6;
}}
QWidget#chatInputBox {{
    background-color: {Colors.bg_input};
    border: 1px solid {Colors.border_light};
    border-radius: 16px;
}}

/* ── Sources panel ──────────────────────────────────────────────── */
QWidget#citationCard {{
    background-color: {Colors.bg_card};
    border: 1px solid {Colors.border};
    border-radius: 10px;
}}
QWidget#citationCard:hover {{
    border-color: {Colors.border_light};
}}
QWidget#primaryCitationCard {{
    background-color: {Colors.bg_card};
    border: 2px solid {Colors.accent};
    border-radius: 10px;
}}

/* ── Combobox / Dropdown ────────────────────────────────────────── */
QComboBox {{
    background-color: {Colors.bg_card};
    border: 1px solid {Colors.border_light};
    border-radius: 8px;
    padding: 5px 10px;
    font-size: 11px;
    color: {Colors.text_muted};
    min-width: 130px;
}}
QComboBox::drop-down {{
    border: none;
    width: 20px;
}}
QComboBox QAbstractItemView {{
    background-color: {Colors.bg_card};
    border: 1px solid {Colors.border_light};
    color: {Colors.text};
    selection-background-color: {Colors.bg_selected};
}}

/* ── Scroll bars ────────────────────────────────────────────────── */
QScrollBar:vertical {{
    background: transparent;
    width: 6px;
    margin: 0;
}}
QScrollBar::handle:vertical {{
    background: {Colors.border_light};
    border-radius: 3px;
    min-height: 20px;
}}
QScrollBar::handle:vertical:hover {{
    background: {Colors.text_faint};
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0px;
}}
QScrollBar:horizontal {{
    height: 0px;
}}

/* ── Splitter ───────────────────────────────────────────────────── */
QSplitter::handle {{
    background-color: {Colors.border};
    width: 1px;
}}

/* ── Tooltips ───────────────────────────────────────────────────── */
QToolTip {{
    background-color: {Colors.bg_card};
    color: {Colors.text};
    border: 1px solid {Colors.border_light};
    border-radius: 6px;
    padding: 4px 8px;
    font-size: 11px;
}}

/* ── Dialog ─────────────────────────────────────────────────────── */
QDialog {{
    background-color: {Colors.bg_panel};
}}
QFormLayout QLabel {{
    color: {Colors.text_muted};
    font-size: 12px;
}}
QCheckBox {{
    color: {Colors.text};
    font-size: 12px;
    spacing: 8px;
}}
QCheckBox::indicator {{
    width: 16px;
    height: 16px;
    border: 1px solid {Colors.border_light};
    border-radius: 4px;
    background: {Colors.bg_input};
}}
QCheckBox::indicator:checked {{
    background: {Colors.accent};
    border-color: {Colors.accent};
}}
QPlainTextEdit {{
    background-color: {Colors.bg_input};
    border: 1px solid {Colors.border};
    border-radius: 8px;
    color: {Colors.text};
    font-family: "Consolas", "Courier New", monospace;
    font-size: 11px;
    padding: 8px;
}}
QProgressBar {{
    background-color: {Colors.bg_card};
    border: none;
    border-radius: 3px;
    height: 4px;
}}
QProgressBar::chunk {{
    background-color: {Colors.accent};
    border-radius: 3px;
}}

/* ── Local Data Status section ──────────────────────────────────── */
QWidget#localDataSection {{
    background-color: transparent;
}}
"""


def relevance_color(score: float) -> str:
    if score >= 0.90:
        return Colors.rel_high
    if score >= 0.75:
        return Colors.rel_med
    return Colors.rel_low


def status_badge(status: str) -> tuple[str, str, str]:
    """Returns (bg, fg, label) for document status."""
    return {
        "ready":      (Colors.success_soft, Colors.success, "●  Ready"),
        "processing": (Colors.warning_soft, Colors.warning, "●  Processing"),
        "failed":     (Colors.danger_soft,  Colors.danger,  "●  Failed"),
    }.get(status, (Colors.bg_card, Colors.text_muted, status.title()))
