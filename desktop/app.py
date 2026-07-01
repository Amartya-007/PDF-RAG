from __future__ import annotations

import logging
import sys

from desktop.runtime import configure_desktop_environment


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )
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

    data_dir = configure_desktop_environment()
    apply_preferences(load_preferences())
    logging.getLogger(__name__).info("starting Local PDF RAG desktop app")
    logging.getLogger(__name__).info("data dir: %s", data_dir)
    logging.getLogger(__name__).info("documents dir: %s", data_dir / "documents")
    logging.getLogger(__name__).info("indexes dir: %s", data_dir / "indexes")
    logging.getLogger(__name__).info("OKF markdown dir: %s", data_dir / "knowledge")
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
