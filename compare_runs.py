"""Diff two schema versions' extraction output for the same document.

    py compare_runs.py                      compare every eval document, v2 -> v3
    py compare_runs.py --old 2 --new 3      pick the versions explicitly
    py compare_runs.py --doc aggressive_nda_01

Why this exists: extraction results are cached by a key that includes
SCHEMA_VERSION, so changing the schema does not overwrite the previous run --
it writes alongside it under a different filename. That makes "what did the
model change its mind about?" answerable directly from disk, with no API calls
and no second run.

This is the tool for checking a prompt change did what it was supposed to do.
A passing eval tells you the score improved; this tells you WHICH fields moved,
which is the difference between "it works" and "it works for the reason I
think".

Makes zero API calls. Reads only what is already cached.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from legal_ai import config  # noqa: E402
from legal_ai.parsing import parse_document  # noqa: E402

# Shown first in the per-clause diff, because these are the fields whose
# descriptions have been rewritten and therefore the ones a prompt change is
# most likely to have moved.
PRIORITY_FIELDS = ("carve_outs_present", "is_mutual", "notice_period_days")


def cache_path(clauses, schema_version: str) -> Path:
    """Recompute the cache key extract.py would use for this schema version.

    Deliberately duplicated from extract._cache_path rather than imported,
    because that function reads SCHEMA_VERSION from the module. We need keys for
    versions the code is NOT currently on, which the original cannot produce.
    """
    payload = json.dumps(
        {
            "clauses": [c.full_text for c in clauses],
            "model": config.EXTRACTION_MODEL,
            "schema_version": schema_version,
        },
        sort_keys=True,
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return config.CACHE_DIR / f"extract_{digest}.json"


def load(path: Path) -> list[dict] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))["clauses"]


def describe(value) -> str:
    if value is None:
        return "null"
    if isinstance(value, list):
        return "[]" if not value else ", ".join(str(v) for v in value)
    return str(value)


def diff_clause(old: dict, new: dict) -> list[tuple[str, str, str]]:
    """Field-level differences between one clause's old and new extraction."""
    changes: list[tuple[str, str, str]] = []

    for field in ("clause_type", "party_favored", "plain_summary", "aggressiveness_signals"):
        if old.get(field) != new.get(field):
            changes.append((field, describe(old.get(field)), describe(new.get(field))))

    old_attrs = old.get("attributes", {})
    new_attrs = new.get("attributes", {})
    for field in sorted(set(old_attrs) | set(new_attrs)):
        if old_attrs.get(field) != new_attrs.get(field):
            changes.append((field, describe(old_attrs.get(field)), describe(new_attrs.get(field))))

    # Priority fields first; the rest keep their existing order.
    changes.sort(key=lambda c: PRIORITY_FIELDS.index(c[0]) if c[0] in PRIORITY_FIELDS else 99)
    return changes


def compare_document(path: Path, old_version: str, new_version: str) -> bool:
    """Print a diff for one document. Returns False if either side is missing."""
    clauses = parse_document(path)
    old_file = cache_path(clauses, old_version)
    new_file = cache_path(clauses, new_version)
    old, new = load(old_file), load(new_file)

    print(f"\n{'=' * 74}")
    print(f"{path.name}")
    print("=" * 74)

    if old is None or new is None:
        missing = []
        if old is None:
            missing.append(f"v{old_version} ({old_file.name})")
        if new is None:
            missing.append(f"v{new_version} ({new_file.name})")
        print(f"  cannot compare -- not cached: {', '.join(missing)}")
        if new is None:
            print(f"  run `py evaluate.py` first to produce the v{new_version} side.")
        return False

    if len(old) != len(new):
        print(f"  note: {len(old)} clauses in v{old_version}, {len(new)} in v{new_version}")

    total = 0
    for i, (o, n) in enumerate(zip(old, new), start=1):
        changes = diff_clause(o, n)
        if not changes:
            continue
        total += len(changes)
        heading = n.get("heading") or o.get("heading") or "(no heading)"
        print(f"\n  clause {i} -- {heading}")
        for field, before, after in changes:
            marker = "  <-- " if field in PRIORITY_FIELDS else "      "
            print(f"    {field}")
            print(f"        v{old_version}: {before}")
            print(f"        v{new_version}: {after}{marker}")

    if total == 0:
        print(f"\n  no differences -- v{new_version} extracted this document identically.")
    else:
        print(f"\n  {total} field(s) changed.")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Diff extraction output across schema versions.")
    parser.add_argument("--old", default="2", help="Schema version to compare FROM (default 2).")
    parser.add_argument("--new", default=None, help="Schema version to compare TO (default: current).")
    parser.add_argument("--doc", default=None, help="One document stem, e.g. aggressive_nda_01.")
    args = parser.parse_args()

    from legal_ai.extract import SCHEMA_VERSION

    new_version = args.new or SCHEMA_VERSION
    if new_version == args.old:
        print(f"--old and --new are both {args.old}; nothing to compare.", file=sys.stderr)
        return 1

    if args.doc:
        candidates = [config.GOLDEN_DIR / f"{args.doc}.txt"]
        if not candidates[0].exists():
            print(f"No such document: {candidates[0]}", file=sys.stderr)
            return 1
    else:
        candidates = sorted(config.GOLDEN_DIR.glob("*.txt"))

    print(f"Comparing schema v{args.old} -> v{new_version}   (no API calls)")
    compared = sum(compare_document(p, args.old, new_version) for p in candidates)

    print(f"\n{'-' * 74}")
    print(f"{compared} of {len(candidates)} document(s) had both versions cached.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
