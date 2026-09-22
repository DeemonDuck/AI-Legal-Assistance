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
MAX_TOKENS_BY_PROVIDER = {"groq": 2400, "anthropic": 8000}
MAX_TOKENS = MAX_TOKENS_BY_PROVIDER[PROVIDER]

# --- Rate limiting -----------------------------------------------------------
# We are on a free tier: 8,000 tokens per minute. Requests are paced BEFORE being
# sent rather than fired and retried on rejection -- a rejected request still
# costs the provider work, and retry-on-rejection is how free tiers get withdrawn.
#
# 7,600 of 8,000, with sole use of the key. The 400 held back absorbs error in
# the per-clause token estimate, which is derived from a sample and will be wrong
# on unusually dense documents.
#
# TWO THINGS NOT TO CHANGE CASUALLY.
#
# Lowering this is not automatically safer. The JSON schema (~1,970 tokens) is
# re-sent on EVERY call, so a lower cap forces smaller clause batches, more
# calls, and more schema resends. Measured over a full corpus + eval run:
#
#     4,000 TPM -> batch 2 -> 66 calls -> 215,622 tokens total
#     6,000 TPM -> batch 4 -> 33 calls -> 140,811 tokens total
#
# The "safer" setting consumed 53% more of the account's allowance overall.
#
# Raising it to ~7,800 allows 6-clause batches and shaves ~3 minutes off a full
# run, but leaves only ~150 tokens of headroom -- one denser-than-average
# document then reserves more than the budget and the call is refused outright.
# Not worth it for an unattended run.
#
# If the key is ever shared again, drop this to ~6,000 so the other user keeps
# usable headroom.
RATE_LIMITS = {
    "groq": {
        "tokens_per_minute": 7600,   # sole use of the key; 400 held back for estimate error
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
