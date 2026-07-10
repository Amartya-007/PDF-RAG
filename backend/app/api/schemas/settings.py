"""API schemas for application settings resources.

Requirements: 21.7
"""
from __future__ import annotations

from pydantic import BaseModel, Field, HttpUrl

class SettingsOut(BaseModel):
    """Read representation of the current application settings.

    Only exposes settings that are safe and meaningful for the frontend
    to read.  Sensitive or server-only keys are omitted.

    Attributes:
        ollama_base_url:       Base URL of the local Ollama instance.
        generation_model:      Active Ollama model for answer synthesis.
        use_ollama:            Whether Ollama synthesis is enabled.
        max_upload_size_mb:    Maximum permitted upload size in megabytes.
        allowed_file_extensions: File extensions accepted for upload.
        backend_host:          Interface the server is bound to.
        backend_port:          Port the server is listening on.
        frontend_origin:       CORS-allowed frontend origin.
        debug_mode:            Whether debug diagnostics are enabled.
    """

    # OPTIMIZATION: Using HttpUrl forces the system to validate the URL structure.
    ollama_base_url: HttpUrl = Field(..., description="Base URL of the local Ollama instance")
    generation_model: str = Field(..., description="Active Ollama model for answer synthesis")
    use_ollama: bool = Field(..., description="Whether Ollama answer synthesis is enabled")
    max_upload_size_mb: int = Field(default=100, ge=1, description="Maximum upload file size in MB")
    allowed_file_extensions: list[str] = Field(
        default_factory=lambda: [".pdf", ".txt", ".md"],
        description="File extensions accepted for document upload",
    )
    backend_host: str = Field(default="127.0.0.1", description="Interface the server is bound to")
    backend_port: int = Field(default=8000, ge=1, le=65535, description="Server port")
    frontend_origin: HttpUrl = Field(
        default="http://localhost:3000",
        description="CORS-allowed frontend origin",
    )
    debug_mode: bool = Field(default=False, description="Whether debug diagnostics are enabled")

    model_config = {"from_attributes": True}


class SettingsPatchRequest(BaseModel):
    """Body for ``PATCH /api/settings``.

    All fields are optional so the client can update only what has
    changed.  Only the subset of settings that can safely be mutated at
    runtime are included.

    Attributes:
        generation_model: Ollama model to use for answer synthesis.
        use_ollama:       Enable or disable Ollama synthesis.
        debug_mode:       Enable or disable debug diagnostics.
    """

    generation_model: str | None = Field(
        default=None,
        description="Ollama model to use for answer synthesis",
    )
    use_ollama: bool | None = Field(
        default=None,
        description="Enable or disable Ollama answer synthesis",
    )
    debug_mode: bool | None = Field(
        default=None,
        description="Enable or disable debug diagnostics",
    )