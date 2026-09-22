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

# --- Provider and models -----------------------------------------------------
# The pipeline calls through legal_ai.llm, so changing provider is this constant
# plus the model IDs below -- no changes anywhere in the pipeline itself.
PROVIDER = os.environ.get("LLM_PROVIDER", "groq")

MODELS = {
    "groq": {
        # gpt-oss-120b is the most capable chat model this key reaches, and it
        # supports strict json_schema output, which the whole pipeline needs.
        "extraction": "openai/gpt-oss-120b",
        "negotiation": "openai/gpt-oss-120b",
    },
    "anthropic": {
        "extraction": "claude-opus-5",
        "negotiation": "claude-opus-5",
    },
}

# Env var per provider, so a key is never read from the wrong variable.
KEY_VARIABLE = {"groq": "GROQ_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}
KEY_PREFIX = {"groq": "gsk_", "anthropic": "sk-ant-"}

if PROVIDER not in MODELS:
    raise RuntimeError(
        f"LLM_PROVIDER='{PROVIDER}' is not recognised. "
        f"Use one of: {', '.join(MODELS)}."
    )

EXTRACTION_MODEL = MODELS[PROVIDER]["extraction"]
NEGOTIATION_MODEL = MODELS[PROVIDER]["negotiation"]

# Output ceiling per request. Provider-specific because Groq counts the
# REQUESTED max_tokens against the per-minute budget, not just what the model
# generates -- so a large ceiling alone can exceed the limit and every request
# is rejected with a 413 before the model even runs.
MAX_TOKENS_BY_PROVIDER = {"groq": 1700, "anthropic": 8000}
MAX_TOKENS = MAX_TOKENS_BY_PROVIDER[PROVIDER]

# --- Rate limiting -----------------------------------------------------------
# We are on a free tier, and the Groq key is SHARED with a teammate. Two things
# follow, and both are deliberate:
#
# 1. Pace requests BEFORE sending rather than firing and retrying whatever
#    bounces. A rejected request still costs the provider work, and
#    retry-on-rejection is how free tiers get withdrawn.
# 2. Leave the other user real headroom. The account's 8,000 TPM is shared, so
#    consuming all of it would give them unexplained 429s caused by this process.
#    6,000 keeps 2,000 TPM free at any instant.
#
# WHY NOT LOWER? Because there are two different courtesy metrics and they pull
# in opposite directions. The JSON schema (~1,970 tokens) is re-sent on EVERY
# call, so a lower per-minute cap forces smaller clause batches, which means
# more calls, which means the schema is re-sent more times. Measured over a full
# corpus + eval run:
#
#     4,000 TPM -> batch 2 -> 66 calls -> 215,622 tokens total
#     6,000 TPM -> batch 4 -> 33 calls -> 140,811 tokens total
#
# The "safer" 4,000 setting consumes 53% MORE of the shared quota overall. 6,000
# leaves instantaneous headroom while using far less of the account's total
# allowance. Raise to ~7,600 only when nobody else is using the key.
RATE_LIMITS = {
    "groq": {
        "tokens_per_minute": 6000,   # of 8000 published; see the note below
        "min_seconds_between_calls": 3.0,
    },
    "anthropic": {
        "tokens_per_minute": 400_000,
        "min_seconds_between_calls": 0.0,
    },
}
RATE_LIMIT = RATE_LIMITS[PROVIDER]

# gpt-oss models emit reasoning tokens that count against max_tokens. Measured on
# a 4-clause extraction: 1059 reasoning tokens at "medium" versus 283 at "low",
# with no observable difference in extracted values. Extraction is a mechanical
# reading task, not a reasoning problem, so the budget is better spent on output.
# Ignored by providers that do not support the parameter.
REASONING_EFFORT = "low"


def get_api_key() -> str:
    """Read the API key for the active provider, with a fix-it error message."""
    variable = KEY_VARIABLE[PROVIDER]
    key = os.environ.get(variable)

    if not key:
        raise RuntimeError(
            f"{variable} is not set (LLM_PROVIDER is '{PROVIDER}').\n"
            f"Create a file at {PROJECT_ROOT / '.env'} containing:\n"
            f"    LLM_PROVIDER={PROVIDER}\n"
            f"    {variable}=your-key-here\n"
            "That file is gitignored and will NOT travel with the repo -- "
            "you must recreate it on each device."
        )

    # A key from the wrong provider otherwise surfaces as a bare 401 several
    # layers down, which is a confusing way to learn you pasted the wrong key.
    expected = KEY_PREFIX[PROVIDER]
    if not key.startswith(expected):
        raise RuntimeError(
            f"{variable} does not look like a {PROVIDER} key -- expected it to "
            f"start with '{expected}'. If you meant a different provider, set "
            f"LLM_PROVIDER in .env."
        )
    return key
