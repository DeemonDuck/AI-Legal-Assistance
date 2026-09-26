"""Run every offline test suite and summarise.

    py tests/run_all.py            one line per suite
    py tests/run_all.py --verbose  full output from each

Exits non-zero if any suite fails, so it can gate a commit or a CI job.

No API key and no network: every suite here runs on fabricated data or temp
files. The paid end-to-end measurement is `py evaluate.py`, which is a separate
thing -- these check that the machinery is correct, that one checks that the
answers are right.

Each suite is a standalone script that does its work at module scope and exits
non-zero on failure, so they are run as subprocesses rather than imported. That
keeps them independently runnable (`py tests/test_scoring.py` still works, which
is how you actually debug one) and stops a crash in an early suite from taking
the rest of the run with it.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description="Run all offline test suites.")
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print each suite's full output instead of one summary line.",
    )
    args = parser.parse_args()

    suites = sorted(TESTS_DIR.glob("test_*.py"))
    if not suites:
        print(f"No test suites found in {TESTS_DIR}.", file=sys.stderr)
        return 1

    failed: list[str] = []

    for suite in suites:
        # sys.executable, not "py": CI and venvs need the interpreter that is
        # actually running this, which is the one with the dependencies.
        completed = subprocess.run(
            [sys.executable, str(suite)],
            capture_output=not args.verbose,
            text=True,
        )

        if args.verbose:
            print()
        elif completed.returncode != 0:
            # Only failures earn their full output. A passing suite's log is
            # noise; a failing one is the whole reason you ran this.
            print(completed.stdout or "", end="")
            print(completed.stderr or "", file=sys.stderr, end="")

        mark = "PASS" if completed.returncode == 0 else "FAIL"
        print(f"  {mark}  {suite.name}")
        if completed.returncode != 0:
            failed.append(suite.name)

    print("\n" + "=" * 60)
    if failed:
        print(f"{len(failed)} of {len(suites)} suites FAILED: {', '.join(failed)}")
        return 1
    print(f"All {len(suites)} offline test suites passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
