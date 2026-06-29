# Offline Desktop App

The desktop product uses PySide6 and calls the existing Python RAG service in-process. It does not require a hosted server, Next.js, React, or internet during normal use.

## Run Locally

```powershell
py -m pip install -e .[desktop]
py -m desktop.app
```

PDF parsing and production local model behavior still require the optional backend/PDF/model setup described in the root README.

## Offline Model Setup

The v1 desktop app does not bundle Ollama model weights. Install Ollama and pull the required models once:

```powershell
ollama pull qwen3.5:4b
ollama pull qwen3-embedding:4b
```

Then run the app offline. The Settings dialog reports whether Ollama is reachable and which models are missing.

The Settings dialog also lets you enable/disable Ollama and choose any installed Ollama model for answer generation and embeddings. Use **Refresh Models** after pulling new models.

## Windows Build

```powershell
py -m pip install -e .[desktop]
pyinstaller desktop/packaging/local_pdf_rag_desktop.spec
```

The generated app is under `dist/Local PDF RAG/`.
