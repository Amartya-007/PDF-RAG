from __future__ import annotations

import pytest

from backend.app.core.config import get_settings


@pytest.fixture(autouse=True)
def _fresh_settings_cache():
    """Reset get_settings()'s cache before and after every test.

    Many tests set RAG_* environment variables and then call get_settings()
    expecting a freshly-constructed Settings object. But get_settings() is
    decorated with a process-wide @lru_cache, so whichever test happens to
    call it *first* in the whole pytest session "wins": every later test
    silently reuses that first, frozen Settings object (and its data_dir/
    sqlite_path) regardless of the env vars it just set. This produced
    order-dependent failures that passed in isolation but failed as part
    of the full suite.
    """
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
