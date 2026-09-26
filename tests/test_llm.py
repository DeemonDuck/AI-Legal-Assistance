"""Offline tests for the provider seam: schema strictification and pacing.

WHY THESE THREE

llm.py had no tests, and it holds the pure functions whose failures are the
least visible in the whole project. Each one is described in that module by a
bug that was found the hard way, and none of them was pinned afterwards:

  strictify()      If this is wrong, EVERY structured call fails -- or worse,
                   quietly drops a field from `required` and the model starts
                   omitting it.
  estimate_tokens() Its own docstring: an under-estimate means "we quietly
                   exceed the provider's limit while believing we are inside
                   it". Omitting the schema from the sum once let through ~46%
                   more than the budget allowed.
  _TokenBudget     Reserving more than the whole per-minute budget used to hit
                   `min()` on an empty window and die, so a corpus build looked
                   like a hang when it had in fact crashed on the first
                   uncached document.

No API key and no network: none of this touches a provider. The waiting path
is driven by a fake clock rather than real sleeps, so the suite stays instant.

Run:  py tests/test_llm.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pydantic import BaseModel  # noqa: E402

from legal_ai import llm  # noqa: E402
from legal_ai.llm import (  # noqa: E402
    ESTIMATE_SAFETY_MARGIN,
    LLMError,
    _TokenBudget,
    estimate_tokens,
    strictify,
)

failures: list[str] = []


def check(label: str, actual, expected) -> None:
    if actual == expected:
        print(f"  PASS  {label}")
    else:
        print(f"  FAIL  {label}: expected {expected!r}, got {actual!r}")
        failures.append(label)


print("\n--- strictify: every object must be closed and fully required ---")

check("additionalProperties is closed",
      strictify({"type": "object", "properties": {"a": {"type": "string"}}}),
      {"type": "object",
       "properties": {"a": {"type": "string"}},
       "additionalProperties": False,
       "required": ["a"]})

# The failure that motivates the required-but-nullable rule in schemas.py: a
# field with a default drops out of Pydantic's `required`, and strict mode
# rejects the schema. strictify must put every property back.
partial = {"type": "object",
           "properties": {"a": {"type": "string"}, "b": {"type": "integer"}},
           "required": ["a"]}
check("a defaulted field is added back to required",
      sorted(strictify(partial)["required"]), ["a", "b"])

# One unstrict object ANYWHERE is enough for the provider to reject the whole
# schema, so the walk has to reach nested objects, arrays and $defs.
nested = {
    "$defs": {"Inner": {"type": "object", "properties": {"x": {"type": "string"}}}},
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {"type": "object", "properties": {"y": {"type": "integer"}}},
        },
    },
}
strictify(nested)
check("$defs are reached",
      nested["$defs"]["Inner"]["additionalProperties"], False)
check("objects inside arrays are reached",
      nested["properties"]["items"]["items"]["additionalProperties"], False)
check("nested required is populated",
      nested["$defs"]["Inner"]["required"], ["x"])

# Non-objects must be left alone rather than grown spurious keys.
leaf = {"type": "string"}
check("a scalar schema is untouched", strictify(leaf), {"type": "string"})
check("an object with no properties is untouched",
      strictify({"type": "object"}), {"type": "object"})


print("\n--- strictify on the real pipeline schemas ---")


def every_object(node):
    """Walk a schema yielding every object node, the way a provider would."""
    if isinstance(node, dict):
        if node.get("type") == "object" and "properties" in node:
            yield node
        for value in node.values():
            yield from every_object(value)
    elif isinstance(node, list):
        for value in node:
            yield from every_object(value)


class _Nested(BaseModel):
    flag: bool | None
    note: str = "defaulted"


class _Outer(BaseModel):
    name: str
    children: list[_Nested]


schema = strictify(_Outer.model_json_schema())
objects = list(every_object(schema))
check("the walk found every object in a real Pydantic schema", len(objects) >= 2, True)
check("every object is closed",
      all(o["additionalProperties"] is False for o in objects), True)
check("every property of every object is required",
      all(sorted(o["required"]) == sorted(o["properties"]) for o in objects), True)


print("\n--- estimate_tokens: never under-reserve ---")

system, user = "s" * 400, "u" * 400
schema_obj = {"big": "x" * 4000}

bare = estimate_tokens(system, user, max_tokens=1000)
withschema = estimate_tokens(system, user, max_tokens=1000, schema=schema_obj)

# The bug: response_format ships the full schema on EVERY request and it counts
# as input. Leaving it out of the sum reserved 2420 for a request costing 3532.
check("the schema is counted, not ignored", withschema > bare, True)

# The requested ceiling counts too -- providers charge the reservation, not
# what the model actually generates.
check("max_tokens is counted",
      estimate_tokens(system, user, 4000) > estimate_tokens(system, user, 1000), True)

# Erring high costs throughput; erring low costs someone else their rate limit.
raw = (len(system) + len(user)) // 4 + 1000
check("the estimate exceeds the raw sum", bare > raw, True)
check("the safety margin is applied", bare, int(raw * ESTIMATE_SAFETY_MARGIN))
check("the margin errs high, never low", ESTIMATE_SAFETY_MARGIN > 1.0, True)
check("an empty prompt still reserves its ceiling",
      estimate_tokens("", "", 500) >= 500, True)


print("\n--- _TokenBudget: pacing, not retrying ---")


class FakeClock:
    """A clock the test drives, so the waiting path costs no wall time."""

    def __init__(self) -> None:
        self.now = 1000.0
        self.slept: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


def with_budget(limit, gap, body):
    """Run `body(budget, clock)` against a fake clock and a chosen limit."""
    clock = FakeClock()
    real_limit, real_time = llm.config.RATE_LIMIT, llm.time
    llm.config.RATE_LIMIT = {"tokens_per_minute": limit,
                             "min_seconds_between_calls": gap}
    llm.time = clock
    try:
        return body(_TokenBudget(), clock)
    finally:
        llm.config.RATE_LIMIT, llm.time = real_limit, real_time


# A request bigger than the entire per-minute budget can never fit, however
# long you wait. Before the explicit guard this reached min() on an empty
# window and raised "min() iterable argument is empty" -- so the corpus build
# appeared to hang while it had actually crashed, with no output and no calls.
def _oversized(budget, _clock):
    try:
        budget.reserve(9999)
        return "no error"
    except LLMError as exc:
        return "guarded" if "can never be sent" in str(exc) else f"wrong message: {exc}"


check("a request larger than the whole budget is refused, not awaited",
      with_budget(1000, 0.0, _oversized), "guarded")


# Inside the budget, nothing waits.
def _fits(budget, clock):
    budget.reserve(300)
    budget.reserve(300)
    return clock.slept


check("requests inside the budget never sleep", with_budget(1000, 0.0, _fits), [])


# Over the budget, it waits for the oldest entry to age out of the window --
# it does not fire the request and retry on rejection. On a free tier that
# distinction is the difference between being paced and being cut off.
def _waits(budget, clock):
    budget.reserve(700)
    clock.now += 10          # 10s into the window
    budget.reserve(700)      # 1400 > 1000, so this must wait out the first
    return clock.slept


slept = with_budget(1000, 0.0, _waits)
check("exceeding the budget waits", len(slept) >= 1, True)
check("it waits for the window to clear, not a fixed guess",
      abs(sum(slept) - 50.0) < 1.0, True)


# The minimum gap between calls is honoured separately from the token budget.
def _gap(budget, clock):
    budget.reserve(1)
    budget.reserve(1)
    return clock.slept


check("the minimum inter-call gap is enforced",
      with_budget(100000, 3.0, _gap), [3.0])


# A non-positive limit disables pacing entirely rather than blocking forever.
def _unlimited(budget, clock):
    budget.reserve(10**9)
    return clock.slept


check("a zero limit disables pacing", with_budget(0, 0.0, _unlimited), [])


print("\n--- Window pruning ---")

# The window is 60s, so pruning at t=200 keeps anything after t=140.
budget = _TokenBudget()
budget._spent = [(100.0, 500), (130.0, 500)]
budget._prune(200.0)
check("entries older than the window are dropped", budget._spent, [])

budget._spent = [(100.0, 500), (150.0, 500), (190.0, 400)]
budget._prune(200.0)
check("entries inside the window are kept",
      budget._spent, [(150.0, 500), (190.0, 400)])

# Exactly on the boundary is treated as expired. Worth pinning either way:
# the alternative silently keeps 60 seconds of spend a moment too long, and
# the whole point of this class is not to overshoot the provider's limit.
budget._spent = [(140.0, 500)]
budget._prune(200.0)
check("an entry exactly at the cutoff is dropped", budget._spent, [])


print("\n" + "=" * 60)
if failures:
    print(f"{len(failures)} FAILED: {', '.join(failures)}")
    raise SystemExit(1)
print("All LLM-seam tests passed.")
