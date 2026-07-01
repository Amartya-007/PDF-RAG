"""Visual theme for the desktop app — dark sidebar, light chat panels."""

from __future__ import annotations


class Colors:
    # App background
    bg = "#1C1C1E"
    bg_light = "#F5F5F7"

    # Sidebar (dark)
    sidebar_bg = "#111111"
    sidebar_item_hover = "#2A2A2E"
    sidebar_item_active = "#3A3A3F"
    sidebar_text = "#E5E5EA"
    sidebar_text_muted = "#8E8E93"

    # Panels (white)
    surface = "#FFFFFF"
    surface_alt = "#F2F2F7"
    border = "#D1D1D6"

    # Text
    text = "#1C1C1E"
    text_muted = "#636366"
    text_faint = "#AEAEB2"

    # Accent — indigo
    accent = "#5E5CE6"
    accent_hover = "#4845C8"
    accent_pressed = "#3634AA"
    accent_soft = "#EEEEFF"

    # Status
    success = "#30D158"
    success_soft = "#E3F9EC"
    warning = "#FF9F0A"
    warning_soft = "#FFF4E0"
    danger = "#FF453A"
    danger_soft = "#FFEEED"

    # Chat bubbles
    user_bubble = "#5E5CE6"
    user_bubble_text = "#FFFFFF"
    assistant_bubble = "#F2F2F7"
    assistant_bubble_text = "#1C1C1E"


STYLESHEET = f"""
* {{
    font-family: "Segoe UI", "SF Pro Text", "Inter", Arial, sans-serif;
}}

QMainWindow {{
    background-color: {Colors.sidebar_bg};
}}

QWidget#root {{
    background-color: {Colors.sidebar_bg};
}}

QWidget {{
    background-color: transparent;
    color: {Colors.text};
}}

/* ---- Header ---- */
QFrame#headerBar {{
    background-color: {Colors.sidebar_bg};
    border-bottom: 1px solid #2A2A2E;
    min-height: 52px;
}}

QLabel#appTitle {{
    font-size: 16px;
    font-weight: 700;
    color: {Colors.sidebar_text};
}}

QLabel#appSubtitle {{
    font-size: 11px;
    color: {Colors.sidebar_text_muted};
}}

/* ---- Sidebar panel ---- */
QFrame#sidebarPanel {{
    background-color: {Colors.sidebar_bg};
    border: none;
    border-radius: 0px;
}}

/* ---- Content panels (chat + citations) ---- */
QFrame#panel {{
    background-color: {Colors.surface};
    border: 1px solid {Colors.border};
    border-radius: 12px;
}}

QLabel#sectionTitle {{
    font-size: 11px;
    font-weight: 700;
    color: {Colors.text_muted};
    letter-spacing: 0.8px;
    text-transform: uppercase;
    padding: 2px 2px 6px 2px;
}}

/* ---- Buttons ---- */
QPushButton {{
    background-color: {Colors.surface};
    border: 1px solid {Colors.border};
    border-radius: 8px;
    padding: 7px 14px;
    font-size: 12px;
    font-weight: 600;
    color: {Colors.text};
}}

QPushButton:hover {{
    background-color: {Colors.surface_alt};
    border-color: {Colors.accent};
}}

QPushButton:pressed {{
    background-color: {Colors.accent_soft};
}}

QPushButton:disabled {{
    color: {Colors.text_faint};
    background-color: {Colors.surface_alt};
}}

QPushButton#primaryButton {{
    background-color: {Colors.accent};
    border: 1px solid {Colors.accent};
    color: white;
}}

QPushButton#primaryButton:hover {{
    background-color: {Colors.accent_hover};
    border-color: {Colors.accent_hover};
}}

QPushButton#primaryButton:pressed {{
    background-color: {Colors.accent_pressed};
}}

QPushButton#sidebarButton {{
    background-color: transparent;
    border: 1px solid #3A3A3F;
    border-radius: 8px;
    padding: 7px 12px;
    font-size: 12px;
    font-weight: 600;
    color: {Colors.sidebar_text};
    text-align: left;
}}

QPushButton#sidebarButton:hover {{
    background-color: {Colors.sidebar_item_hover};
}}

/* ---- Tree widget (sidebar sessions) ---- */
QTreeWidget {{
    background: transparent;
    border: none;
    outline: none;
    font-size: 13px;
    color: {Colors.sidebar_text};
}}

QTreeWidget::item {{
    padding: 6px 8px;
    border-radius: 8px;
    margin: 1px 2px;
    color: {Colors.sidebar_text};
}}

QTreeWidget::item:selected {{
    background-color: {Colors.sidebar_item_active};
    color: {Colors.sidebar_text};
}}

QTreeWidget::item:hover:!selected {{
    background-color: {Colors.sidebar_item_hover};
}}

QTreeWidget::branch {{
    background: transparent;
    image: none;
}}

/* ---- Input (inside dark sidebar / dark panels) ---- */
QLineEdit {{
    background-color: #2A2A2E;
    border: 1.5px solid #3A3A3F;
    border-radius: 10px;
    padding: 10px 14px;
    font-size: 13px;
    color: #FFFFFF;
    selection-background-color: {Colors.accent};
    selection-color: #FFFFFF;
}}

QLineEdit:focus {{
    border: 1.5px solid {Colors.accent};
    background-color: #2A2A2E;
}}

QLineEdit:hover {{
    border: 1.5px solid #4A4A4F;
}}

/* ---- Text views ---- */
QTextBrowser {{
    background-color: {Colors.surface};
    border: none;
    font-size: 13px;
    color: {Colors.text};
}}

QPlainTextEdit {{
    background-color: #2A2A2E;
    border: 1px solid #3A3A3F;
    border-radius: 8px;
    font-size: 12px;
    color: #E5E5EA;
    padding: 6px;
}}

/* ---- Splitter ---- */
QSplitter::handle {{
    background-color: transparent;
    width: 6px;
}}

/* ---- Progress bar ---- */
QProgressBar {{
    background-color: {Colors.surface_alt};
    border: none;
    border-radius: 4px;
    height: 6px;
}}

QProgressBar::chunk {{
    background-color: {Colors.accent};
    border-radius: 4px;
}}

/* ---- Combo + dialog ---- */
QComboBox {{
    background-color: #2A2A2E;
    border: 1px solid #3A3A3F;
    border-radius: 8px;
    padding: 6px 10px;
    color: #FFFFFF;
    font-size: 12px;
}}

QComboBox:hover {{
    border-color: {Colors.accent};
}}

QComboBox QAbstractItemView {{
    background-color: #2A2A2E;
    color: #FFFFFF;
    selection-background-color: {Colors.accent};
    selection-color: #FFFFFF;
    border: 1px solid #3A3A3F;
}}

QComboBox::drop-down {{
    border: none;
}}

QDialog {{
    background-color: #1C1C1E;
}}

QDialog QWidget {{
    color: #FFFFFF;
    background-color: transparent;
}}

QDialog QLabel {{
    color: #E5E5EA;
}}

QDialog QLineEdit {{
    background-color: #2A2A2E;
    border: 1px solid #3A3A3F;
    border-radius: 8px;
    padding: 6px 10px;
    color: #FFFFFF;
    font-size: 12px;
}}

QDialog QLineEdit:focus {{
    border-color: {Colors.accent};
}}

QDialog QCheckBox {{
    color: #E5E5EA;
    spacing: 8px;
}}

QDialog QPlainTextEdit {{
    background-color: #2A2A2E;
    border: 1px solid #3A3A3F;
    border-radius: 8px;
    color: #E5E5EA;
    font-size: 12px;
    padding: 6px;
}}

/* ---- Scrollbar ---- */
QScrollBar:vertical {{
    background: transparent;
    width: 8px;
}}

QScrollBar::handle:vertical {{
    background: {Colors.border};
    border-radius: 4px;
    min-height: 20px;
}}

QScrollBar::handle:vertical:hover {{
    background: {Colors.text_faint};
}}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0px;
}}
"""


def status_badge_style(status: str) -> tuple[str, str, str]:
    """Returns (background, text_color, label) for a document status."""
    mapping = {
        "ready": (Colors.success_soft, Colors.success, "Ready"),
        "processing": (Colors.warning_soft, Colors.warning, "Processing"),
        "failed": (Colors.danger_soft, Colors.danger, "Failed"),
    }
    return mapping.get(status, (Colors.surface_alt, Colors.text_muted, status.title()))
