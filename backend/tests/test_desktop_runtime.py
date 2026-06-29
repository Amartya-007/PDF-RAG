from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from desktop.runtime import configure_desktop_environment, default_app_data_dir


class DesktopRuntimeTests(unittest.TestCase):
    def test_default_app_data_uses_local_app_data_on_windows(self) -> None:
        with patch.dict(os.environ, {"LOCALAPPDATA": "C:\\Users\\Test\\AppData\\Local"}, clear=False):
            path = default_app_data_dir()

        self.assertEqual(path, Path("C:\\Users\\Test\\AppData\\Local") / "Local PDF RAG" / "data")

    def test_configure_does_not_override_existing_data_dir(self) -> None:
        with patch.dict(os.environ, {"RAG_DATA_DIR": "D:\\Existing"}, clear=False):
            path = configure_desktop_environment()

        self.assertEqual(path, Path("D:\\Existing"))


if __name__ == "__main__":
    unittest.main()
