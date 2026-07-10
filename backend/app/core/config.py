from __future__ import annotations

from pathlib import Path
from functools import lru_cache
from pydantic import Field, HttpUrl, DirectoryPath
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """Application settings, populated from environment variables."""
    
    # Environment variable prefix (e.g., RAG_DATA_DIR matches data_dir)
    model_config = SettingsConfigDict(env_prefix="RAG_", env_file=".env", extra="ignore")

    # Paths
    data_dir: DirectoryPath = Field(default=Path("backend/data"))
    
    # SQLite
    sqlite_path_value: str = Field(default="metadata.sqlite3", alias="RAG_SQLITE_PATH")

    # Models & URLs
    ollama_base_url: HttpUrl = Field(default="http://localhost:11434")
    generation_model: str = Field(default="llama3.2")
    development_model: str = Field(default="llama3.2")
    active_model: str = Field(default="llama3.2")

    # Numerical Settings with Validation
    sparse_top_k: int = Field(default=40, gt=0)
    final_context_chunks: int = Field(default=8, gt=0)
    temperature: float = Field(default=0.1, ge=0.0, le=1.0)

    # Boolean flags (Pydantic handles 'true', '1', 'yes', 'on' automatically)
    use_ollama: bool = Field(default=False)
    force_ocr: bool = Field(default=False)
    use_tree_search: bool = Field(default=True)

    @property
    def documents_dir(self) -> Path:
        return self.data_dir / "documents"

    @property
    def indexes_dir(self) -> Path:
        return self.data_dir / "indexes"

    @property
    def okf_dir(self) -> Path:
        return self.data_dir / "knowledge"

    @property
    def trees_dir(self) -> Path:
        return self.data_dir / "trees"

    @property
    def sqlite_path(self) -> Path | str:
        if self.sqlite_path_value == ":memory:":
            return ":memory:"
        return (self.data_dir / self.sqlite_path_value).resolve()

@lru_cache
def get_settings() -> Settings:
    """Returns a cached settings object."""
    return Settings()

def ensure_data_dirs(settings: Settings) -> None:
    """Ensures that required data directories exist."""
    for path in [
        settings.data_dir,
        settings.documents_dir,
        settings.indexes_dir,
        settings.okf_dir,
        settings.trees_dir,
    ]:
        path.mkdir(parents=True, exist_ok=True)