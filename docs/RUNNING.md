# Running Local PDF RAG

This app is a Windows-first offline desktop PDF RAG product. The normal user interface is the PySide6 desktop app. No Next.js, React, hosted server, or internet connection is required during normal use after setup.

## 1. Recommended: Run the Built Windows App

From the project root:

```powershell
cd D:\PDF-RAG
& "D:\PDF-RAG\dist\Local PDF RAG\Local PDF RAG.exe"
```

Or open this file directly in Explorer:

```text
D:\PDF-RAG\dist\Local PDF RAG\Local PDF RAG.exe
```

Do not run the executable under `D:\PDF-RAG\build\...`. The `build` folder is only a temporary PyInstaller work directory and can fail with Python DLL errors.

Use the app like this:

1. Click `Settings`.
2. Confirm Ollama status and selected models.
3. Click `Import PDFs / Text`.
4. Select one or more PDF or text files.
5. Wait until the documents show `[ready]`.
6. Ask a question in the chat box.
7. Check the right-side citations panel for source filename, page, chunk ID, and excerpt.

If a previous import is stuck or failed, click `Repair Stuck Imports`.

## 2. Run From Source

Use `py`, not plain `python`, because this machine has Python 3.13 available through the Python launcher.

```powershell
cd D:\PDF-RAG
py --version
py -m pip install -e .[desktop]
py -m desktop.app
```

If PowerShell complains about the `[desktop]` extra, wrap it in quotes:

```powershell
py -m pip install -e ".[desktop]"
```

## 3. Ollama Setup

Ollama is optional for fallback testing, but recommended for real local AI answers and embeddings.

Install Ollama once, then pull models while online:

```powershell
ollama pull qwen3.5:4b
ollama pull qwen3-embedding:4b
```

Optional larger model:

```powershell
ollama pull qwen3.5:9b
```

Start or confirm Ollama:

```powershell
ollama list
```

Default desktop settings:

```text
Ollama URL: http://localhost:11434
Active model: qwen3.5:4b
Embedding model: qwen3-embedding:4b
```

Inside the app, open `Settings` to select any installed Ollama model.

## 4. Offline Use

After Ollama and the models are installed, the app can run offline:

```powershell
cd D:\PDF-RAG
.\dist\"Local PDF RAG"\"Local PDF RAG.exe"
```

The app keeps user data locally under:

```text
C:\Users\Windows 11\AppData\Local\Local PDF RAG\data
```

This includes imported PDFs, SQLite metadata, local indexes, generated OKF Markdown, settings, and logs.

## 5. Developer Commands

Run tests:

```powershell
cd D:\PDF-RAG
py -m unittest discover backend/tests
```

Run the CLI:

```powershell
py -m backend.app.cli init
py -m backend.app.cli ingest .\some-document.pdf
py -m backend.app.cli ask "What is this document about?"
```

Run the FastAPI developer server:

```powershell
py -m pip install -e .[backend]
py -m uvicorn backend.app.api.main:app --reload
```

## 6. Build the Windows Executable

Close the desktop app before rebuilding, otherwise Windows may lock packaged DLL files.

```powershell
cd D:\PDF-RAG
py -m pip install -e .[desktop]
py -m PyInstaller desktop/packaging/local_pdf_rag_desktop.spec --clean -y
```

The rebuilt executable will be here:

```text
D:\PDF-RAG\dist\Local PDF RAG\Local PDF RAG.exe
```

## 7. Common Problems

### PySide6 Import Error in VS Code

Make sure VS Code is using Python 3.13 from:

```text
C:\Users\Windows 11\AppData\Local\Microsoft\WindowsApps\python3.exe
```

Then install desktop dependencies:

```powershell
cd D:\PDF-RAG
py -m pip install -e .[desktop]
```

### PDF Parsing Error

Install the desktop or PDF dependencies:

```powershell
py -m pip install -e .[desktop]
```

For stronger optional PDF parsing/OCR support:

```powershell
py -m pip install -e .[pdf]
```

### Ollama Is Disabled

Open `Settings` in the desktop app and enable Ollama. Then confirm the selected models are installed with:

```powershell
ollama list
```

If a model is missing:

```powershell
ollama pull qwen3.5:4b
ollama pull qwen3-embedding:4b
```

### App Is Slow

Use a smaller active model first:

```text
qwen3.5:4b
```

Also make sure the embedding model is already pulled locally:

```powershell
ollama pull qwen3-embedding:4b
```
