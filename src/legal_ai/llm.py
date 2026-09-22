"""Provider-agnostic LLM calls.

WHY THIS LAYER EXISTS

The pipeline was written against the Anthropic API and is currently running on
Groq, which serves open models (Llama, Qwen, GPT-OSS) behind an OpenAI-compatible
interface. Those are different SDKs with different request shapes, and without a
seam between them the provider choice would be smeared across extract.py and
negotiate.py -- so switching back, or adding a third, would mean editing the
pipeline rather than a config value.

Everything above this module asks for the same two things: "run this prompt and
give me an instance of this Pydantic model", or "run this prompt and give me
text". Which provider serves that is set by PROVIDER in config.py.

A NOTE ON STRICT JSON SCHEMAS

Pydantic emits JSON Schema that is valid but not *strict* in the sense both
providers require: strict mode needs `additionalProperties: false` on every
object and every property listed in `required`. Pydantic omits the first and
only lists non-defaulted fields in the second. `strictify()` walks the schema and
fixes both. This is why schemas.py declares fields as required-but-nullable
rather than optional-with-defaults -- a defaulted field would silently drop out
of `required` and the provider would reject the schema.
"""

from __future__ import annotations

import json
import time
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from legal_ai import config

ModelT = TypeVar("ModelT", bound=BaseModel)


def strictify(node):
    """Make a Pydantic JSON Schema acceptable to strict structured-output modes.

    Mutates in place and returns the node. Recurses through `$defs`, nested
    objects and arrays, because a single non-strict object anywhere in the tree
    is enough for the provider to reject the whole schema.
    """
    if isinstance(node, dict):
        if node.get("type") == "object" and "properties" in node:
            node["additionalProperties"] = False
            node["required"] = list(node["properties"].keys())
        for value in node.values():
            strictify(value)
    elif isinstance(node, list):
        for value in node:
            strictify(value)
    return node


class LLMError(RuntimeError):
    """Raised for any provider failure, with a message meant for a user.

    Provider-specific exception types stop here deliberately: callers handle one
    error type and their messages stay readable when the provider changes.
    """


# --- Rate limiting -----------------------------------------------------------


class _TokenBudget:
    """Keeps request volume inside the provider's published per-minute limit.

    This paces requests BEFORE sending rather than retrying whatever gets
    rejected. On a free tier that distinction matters: a rejected request still
    costs the provider work, and retry-on-rejection is exactly the pattern that
    gets free tiers withdrawn. We wait our turn instead.

    The window is a rolling 60 seconds of (timestamp, tokens) pairs. Before each
    call we estimate its cost, drop entries older than the window, and sleep
    until enough budget has aged out.
    """

    WINDOW_SECONDS = 60.0

    def __init__(self) -> None:
        self._spent: list[tuple[float, int]] = []
        self._last_call = 0.0

    def _prune(self, now: float) -> None:
        cutoff = now - self.WINDOW_SECONDS
        self._spent = [(ts, n) for ts, n in self._spent if ts > cutoff]

    def reserve(self, estimated_tokens: int) -> None:
        """Block until `estimated_tokens` fits inside the budget, then record it."""
        limit = config.RATE_LIMIT["tokens_per_minute"]
        gap = config.RATE_LIMIT["min_seconds_between_calls"]
        if limit <= 0:
            return

        # A single request larger than the whole per-minute budget can never fit,
        # no matter how long we wait. Caught explicitly because the waiting loop
        # below would otherwise call min() on an empty window and die with
        # "min() iterable argument is empty" -- which is what actually happened:
        # the corpus build appeared to hang while it had in fact crashed on the
        # first uncached document, producing no output and no API calls.
        if estimated_tokens > limit:
            raise LLMError(
                f"A single request reserves {estimated_tokens} tokens but the "
                f"per-minute budget is only {limit}. It can never be sent.\n"
                "Either lower MAX_TOKENS in config.py (which shrinks the clause "
                "batch size to match), or raise tokens_per_minute -- but only up "
                "to the provider's published limit, and only if nobody else is "
                "using the key."
            )

        while True:
            now = time.monotonic()
            self._prune(now)
            used = sum(n for _, n in self._spent)

            if used + estimated_tokens <= limit:
                break

            # Wait until the oldest entry leaves the window and frees its budget.
            oldest = min(ts for ts, _ in self._spent)
            wait = max(0.5, (oldest + self.WINDOW_SECONDS) - now)
            print(
                f"  (rate limit: waiting {wait:.0f}s -- {used}/{limit} tokens "
                f"used this minute)"
            )
            time.sleep(wait)

        since_last = time.monotonic() - self._last_call
        if gap and since_last < gap:
            time.sleep(gap - since_last)

        self._last_call = time.monotonic()
        self._spent.append((self._last_call, estimated_tokens))


_BUDGET = _TokenBudget()


# The estimate is what the rate limiter reserves, so it must never come in under
# the real cost -- an under-estimate means we quietly exceed the provider's limit
# while believing we are inside it. Applied after the component costs are summed.
ESTIMATE_SAFETY_MARGIN = 1.15


def estimate_tokens(system: str, user: str, max_tokens: int, schema: dict | None = None) -> int:
    """Upper-bound token cost of a request.

    Three components, and missing any one of them under-reserves:

    1. The prompt text.
    2. The JSON schema. This is the one that caught us out -- `response_format`
       ships the full schema on EVERY request and it counts as input. For this
       project's schema that is ~1970 tokens, which was 3x the entire rest of the
       estimate. Omitting it meant reserving 2420 tokens for a request that
       actually cost 3532, so the limiter let through ~46% more than the budget
       allowed and we sailed past the provider's limit while believing we were
       under it.
    3. The requested `max_tokens`, because providers count the reserved ceiling
       rather than what the model actually generates.

    ~4 characters per token, then a safety margin. Erring high costs a little
    throughput; erring low costs someone else their rate limit.
    """
    prompt_tokens = (len(system) + len(user)) // 4
    schema_tokens = len(json.dumps(schema)) // 4 if schema else 0
    raw = prompt_tokens + schema_tokens + max_tokens
    return int(raw * ESTIMATE_SAFETY_MARGIN)


# --- Groq --------------------------------------------------------------------


def _groq_client():
    import groq

    return groq.Groq(api_key=config.get_api_key())


def _groq_structured(system: str, user: str, model_cls, model: str, max_tokens: int):
    import groq

    schema = strictify(model_cls.model_json_schema())

    # gpt-oss-120b intermittently returns a bare array where the schema's root
    # is an object, and Groq rejects it server-side with json_validate_failed.
    # It is a generation slip rather than a prompt error -- the same input
    # succeeds on retry -- so retry once with the shape restated before failing.
    # Seen live on 2 of 11 documents.
    for attempt in range(2):
        try:
            return _groq_attempt(system, user, model_cls, model, max_tokens, schema,
                                 retry=attempt > 0)
        except _SchemaSlip as slip:
            if attempt == 1:
                raise LLMError(
                    "The model returned output in the wrong shape twice in a row. "
                    f"Last error: {slip}"
                ) from slip


def _retry_after_seconds(exc, default: float = 60.0) -> float:
    """Read the provider's retry-after header, falling back to a full window."""
    headers = getattr(getattr(exc, "response", None), "headers", {}) or {}
    for key in ("retry-after", "x-ratelimit-reset-tokens"):
        raw = headers.get(key)
        if raw:
            try:
                return float(str(raw).rstrip("s"))
            except ValueError:
                pass
    return default


class _SchemaSlip(Exception):
    """Internal: the provider rejected the generated JSON's shape."""


def _groq_attempt(system, user, model_cls, model, max_tokens, schema, *, retry):
    import groq

    if retry:
        user = (
            f"{user}\n\nIMPORTANT: respond with a single JSON OBJECT matching the "
            f"schema. The top level must be an object, not an array."
        )

    _BUDGET.reserve(estimate_tokens(system, user, max_tokens, schema))

    try:
        response = _groq_client().chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            reasoning_effort=config.REASONING_EFFORT,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": model_cls.__name__,
                    "schema": schema,
                    "strict": True,
                },
            },
        )
    except groq.RateLimitError as exc:
        # Pacing should prevent this. If it happens anyway our estimate was low,
        # so honour the provider's own retry-after rather than guessing.
        wait = _retry_after_seconds(exc)
        raise LLMError(
            f"Groq rate limit reached despite pacing. Wait {wait:.0f}s and rerun "
            "-- already-extracted documents are cached and will not be re-sent."
        ) from exc
    except groq.APIStatusError as exc:
        if getattr(exc, "status_code", None) == 400 and "json_validate_failed" in str(exc):
            raise _SchemaSlip(str(exc)[:200]) from exc
        raise LLMError(f"Groq API error {exc.status_code}: {exc.message}") from exc
    except groq.APIConnectionError as exc:
        raise LLMError("Could not reach the Groq API. Check your connection.") from exc

    choice = response.choices[0]
    if choice.finish_reason == "length":
        raise LLMError(
            f"Response hit the {max_tokens}-token cap and is truncated. "
            "Raise MAX_TOKENS in config.py, or process the document in batches."
        )

    content = choice.message.content or ""
    try:
        return model_cls.model_validate(json.loads(content))
    except (json.JSONDecodeError, ValidationError) as exc:
        # Strict mode should make this impossible, but a truncated or refused
        # response can still land here -- better a clear error than a crash
        # somewhere downstream with a half-built object.
        raise LLMError(
            f"The model returned output that did not match the expected schema: {exc}"
        ) from exc


def _groq_text(system: str, user: str, model: str, max_tokens: int) -> str:
    import groq

    _BUDGET.reserve(estimate_tokens(system, user, max_tokens))

    try:
        response = _groq_client().chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
    except groq.APIStatusError as exc:
        raise LLMError(f"Groq API error {exc.status_code}: {exc.message}") from exc
    except groq.APIConnectionError as exc:
        raise LLMError("Could not reach the Groq API. Check your connection.") from exc

    return (response.choices[0].message.content or "").strip()


# --- Anthropic ---------------------------------------------------------------
# Retained so PROVIDER can be switched back without rewriting the pipeline.


def _anthropic_structured(system: str, user: str, model_cls, model: str, max_tokens: int):
    import anthropic

    try:
        response = anthropic.Anthropic(api_key=config.get_api_key()).messages.parse(
            model=model,
            max_tokens=max_tokens,
            # Cached because the system prompt is identical across every
            # document and every corpus pass.
            system=[{
                "type": "text",
                "text": system,
                "cache_control": {"type": "ephemeral"},
            }],
            messages=[{"role": "user", "content": user}],
            output_format=model_cls,
        )
    except anthropic.RateLimitError as exc:
        raise LLMError("Anthropic rate limit reached. Wait and retry.") from exc
    except anthropic.APIStatusError as exc:
        raise LLMError(f"Anthropic API error {exc.status_code}: {exc.message}") from exc
    except anthropic.APIConnectionError as exc:
        raise LLMError("Could not reach the Anthropic API.") from exc

    if response.stop_reason == "max_tokens":
        raise LLMError(
            f"Response hit the {max_tokens}-token cap and is truncated. "
            "Raise MAX_TOKENS in config.py."
        )
    return response.parsed_output


def _anthropic_text(system: str, user: str, model: str, max_tokens: int) -> str:
    import anthropic

    try:
        response = anthropic.Anthropic(api_key=config.get_api_key()).messages.create(
            model=model,
            max_tokens=max_tokens,
            system=[{
                "type": "text",
                "text": system,
                "cache_control": {"type": "ephemeral"},
            }],
            messages=[{"role": "user", "content": user}],
        )
    except anthropic.APIStatusError as exc:
        raise LLMError(f"Anthropic API error {exc.status_code}: {exc.message}") from exc

    return "".join(b.text for b in response.content if b.type == "text").strip()


# --- Public API --------------------------------------------------------------


def structured(
    system: str,
    user: str,
    model_cls: type[ModelT],
    *,
    model: str,
    max_tokens: int | None = None,
) -> ModelT:
    """Run a prompt and return a validated instance of `model_cls`."""
    max_tokens = max_tokens or config.MAX_TOKENS

    if config.PROVIDER == "groq":
        return _groq_structured(system, user, model_cls, model, max_tokens)
    if config.PROVIDER == "anthropic":
        return _anthropic_structured(system, user, model_cls, model, max_tokens)
    raise LLMError(f"Unknown PROVIDER '{config.PROVIDER}' in config.py.")


def text(
    system: str,
    user: str,
    *,
    model: str,
    max_tokens: int = 2000,
) -> str:
    """Run a prompt and return plain text."""
    if config.PROVIDER == "groq":
        return _groq_text(system, user, model, max_tokens)
    if config.PROVIDER == "anthropic":
        return _anthropic_text(system, user, model, max_tokens)
    raise LLMError(f"Unknown PROVIDER '{config.PROVIDER}' in config.py.")
