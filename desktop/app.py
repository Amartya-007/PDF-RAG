from __future__ import annotations

import sys

from desktop.runtime import configure_desktop_environment


def main() -> int:
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError as exc:
        raise RuntimeError(
            "PySide6 is not installed. Install desktop dependencies with "
            "`py -m pip install -e .[desktop]`."
        ) from exc

    from desktop.controller import DesktopController
    from desktop.main_window import MainWindow
    from desktop.preferences import apply_preferences, load_preferences
    from desktop.theme import STYLESHEET

    configure_desktop_environment()
    apply_preferences(load_preferences())
    app = QApplication(sys.argv)
    app.setApplicationName("Local PDF RAG")
    app.setStyleSheet(STYLESHEET)
    controller = DesktopController()
    window = MainWindow(controller)
    window.resize(1280, 780)
    window.show()
    exit_code = app.exec()
    controller.close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
