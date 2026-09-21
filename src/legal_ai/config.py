"""Project-wide configuration.

PORTABILITY RULE: every path in this project is derived from PROJECT_ROOT,
which is computed from this file's own location. Never write an absolute
path like C:/Users/... anywhere in the codebase -- it will break the moment
the project moves to another device.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# config.py -> legal_ai -> src -> PROJECT_ROOT
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
CORPUS_DIR = DATA_DIR / "corpus"          # market-standard NDA templates
GOLDEN_DIR = DATA_DIR / "golden"          # eval set with planted aggressive terms
CORPUS_INDEX_DIR = DATA_DIR / "corpus_index"  # generated; gitignored
CACHE_DIR = PROJECT_ROOT / ".cache"       # LLM response cache; gitignored

for _d in (CORPUS_DIR, GOLDEN_DIR, CORPUS_INDEX_DIR, CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

load_dotenv(PROJECT_ROOT / ".env")

# --- Models -----------------------------------------------------------------
# Flip EXTRACTION_MODEL to "claude-sonnet-5" if you want to cut cost on the
# high-volume corpus pass; keep the debate on Opus, that's where quality shows.
EXTRACTION_MODEL = "claude-opus-5"
NEGOTIATION_MODEL = "claude-opus-5"

MAX_TOKENS = 8000


def get_api_key() -> str:
    """Read the Anthropic key, with an error message that says how to fix it."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set.\n"
            f"Create a file at {PROJECT_ROOT / '.env'} containing:\n"
            "    ANTHROPIC_API_KEY=sk-ant-...\n"
            "That file is gitignored and will NOT travel with the repo -- "
            "you must recreate it on each device."
        )
    return key
