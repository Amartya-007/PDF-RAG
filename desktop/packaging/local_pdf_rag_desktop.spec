# PyInstaller spec for the Windows-first offline desktop build.
# Build from the repository root after installing desktop dependencies:
#   pyinstaller desktop/packaging/local_pdf_rag_desktop.spec

from PyInstaller.utils.hooks import collect_submodules
from pathlib import Path


block_cipher = None
project_root = Path(SPECPATH).parents[1]

hiddenimports = collect_submodules("backend") + collect_submodules("desktop")

a = Analysis(
    [str(project_root / "desktop" / "app.py")],
    pathex=[str(project_root)],
    binaries=[],
    datas=[
        (str(project_root / "README.md"), "."),
        (str(project_root / "Model-to-use-instructions.md"), "."),
    ],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Local PDF RAG",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="Local PDF RAG",
)
