"""Visual theme for the desktop app.

A single source of truth for colors and the QSS stylesheet, so the rest of
the UI code never hardcodes hex values. Keeping this separate also makes it
straightforward to add a dark theme later without touching main_window.py.
"""

from __future__ import annotations


class Colors:
    bg = "#F6F7FB"
    surface = "#FFFFFF"
    surface_alt = "#F0F1F6"
    border = "#E2E4EC"
    text = "#1B1D29"
    text_muted = "#6B7080"
    text_faint = "#9498A6"

    accent = "#4F46E5"
    accent_hover = "#4338CA"
    accent_pressed = "#3730A3"
    accent_soft = "#EEF0FE"

    success = "#16A34A"
    success_soft = "#E7F8EC"
    warning = "#D97706"
    warning_soft = "#FEF3E2"
    danger = "#DC2626"
    danger_soft = "#FDECEC"

    user_bubble = "#4F46E5"
    user_bubble_text = "#FFFFFF"
    assistant_bubble = "#F0F1F6"
    assistant_bubble_text = "#1B1D29"


STYLESHEET = f"""
* {{
    font-family: "Segoe UI", "Inter", "Helvetica Neue", Arial, sans-serif;
    color: {Colors.text};
}}

QMainWindow, QWidget#root {{
    background-color: {Colors.bg};
}}

QWidget {{
    background-color: transparent;
}}

QLabel#appTitle {{
    font-size: 17px;
    font-weight: 700;
    color: {Colors.text};
}}

QLabel#appSubtitle {{
    font-size: 11px;
    color: {Colors.text_muted};
}}

QFrame#headerBar {{
    background-color: {Colors.surface};
    border-bottom: 1px solid {Colors.border};
}}

QFrame#panel {{
    background-color: {Colors.surface};
    border: 1px solid {Colors.border};
    border-radius: 10px;
}}

QLabel#sectionTitle {{
    font-size: 12px;
    font-weight: 700;
    color: {Colors.text_muted};
    letter-spacing: 0.5px;
    padding: 2px 2px 6px 2px;
}}

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

QListWidget {{
    background-color: {Colors.surface};
    border: none;
    outline: none;
}}

QListWidget::item {{
    border: none;
    padding: 0px;
    margin: 2px 0px;
}}

QListWidget::item:selected {{
    background-color: {Colors.accent_soft};
    border-radius: 8px;
}}

QLineEdit {{
    background-color: {Colors.surface};
    border: 1px solid {Colors.border};
    border-radius: 8px;
    padding: 9px 12px;
    font-size: 13px;
}}

QLineEdit:focus {{
    border: 1px solid {Colors.accent};
}}

QTextBrowser, QPlainTextEdit {{
    background-color: {Colors.surface};
    border: none;
    font-size: 13px;
}}

QSplitter::handle {{
    background-color: {Colors.bg};
    width: 10px;
}}

QProgressBar {{
    background-color: {Colors.surface_alt};
    border: none;
    border-radius: 4px;
    height: 8px;
}}

QProgressBar::chunk {{
    background-color: {Colors.accent};
    border-radius: 4px;
}}

QComboBox, QFormLayout QLineEdit {{
    background-color: {Colors.surface};
    border: 1px solid {Colors.border};
    border-radius: 8px;
    padding: 6px 10px;
}}

QDialog {{
    background-color: {Colors.bg};
}}

QScrollBar:vertical {{
    background: transparent;
    width: 10px;
}}

QScrollBar::handle:vertical {{
    background: {Colors.border};
    border-radius: 5px;
    min-height: 24px;
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
