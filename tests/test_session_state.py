"""Offline tests for the per-document session-state reset in app.py.

THE BUG THIS PINS DOWN

Streamlit keeps session_state for the whole browser session, and uploading a
different file is just another rerun. The negotiating positions and the draft
email are stored there on purpose -- that is what stops a button click being
lost on the next interaction -- but nothing used to clear them when the
document changed. So after reviewing NDA A and generating its positions,
uploading NDA B rendered A's redlines, A's talking points and A's ready-to-send
email underneath B's findings, with no button pressed and nothing on screen
saying they came from a different contract. The download button offered them
under B's filename.

Found by a user reviewing a real NDA after the samples.

HOW THIS RUNS WITHOUT STREAMLIT

app.py executes its whole UI at module scope, so importing it here is not an
option, and installing Streamlit to test a dictionary rule would be absurd. The
function is lifted out of the file by AST and executed on its own -- the same
approach tests/test_accessibility.py uses to read the palette. It therefore
tests the code that actually ships, not a copy of it: rename or change the
function and this fails rather than quietly passing.

Run:  py tests/test_session_state.py
"""

from __future__ import annotations

import ast
import hashlib
import io
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

failures: list[str] = []


def check(label: str, actual, expected) -> None:
    if actual == expected:
        print(f"  PASS  {label}")
    else:
        print(f"  FAIL  {label}: expected {expected!r}, got {actual!r}")
        failures.append(label)


def load_from_app(*names: str) -> dict:
    """Execute just the named top-level definitions from app.py."""
    tree = ast.parse((ROOT / "app.py").read_text(encoding="utf-8"))

    wanted = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            wanted.append(node)
        elif isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id in names for t in node.targets
        ):
            wanted.append(node)

    found = {
        n.name if isinstance(n, ast.FunctionDef) else n.targets[0].id for n in wanted
    }
    missing = set(names) - found
    if missing:
        raise AssertionError(
            f"app.py no longer defines {sorted(missing)} at module level -- "
            "renamed, or moved inside another scope?"
        )

    namespace: dict = {"hashlib": hashlib}
    exec(compile(ast.Module(body=wanted, type_ignores=[]), "app.py", "exec"), namespace)
    return namespace


app = load_from_app("_forget_previous_document", "_PER_DOCUMENT_STATE")
forget = app["_forget_previous_document"]
PER_DOCUMENT = app["_PER_DOCUMENT_STATE"]


class FakeUpload(io.BytesIO):
    """Stands in for Streamlit's UploadedFile, which is also a BytesIO."""

    def __init__(self, content: bytes, name: str = "nda.pdf") -> None:
        super().__init__(content)
        self.name = name


def worked_on(document: FakeUpload) -> dict:
    """A session that has reviewed `document` and generated everything."""
    state: dict = {}
    forget(state, document)
    for key in PER_DOCUMENT:
        state[key] = f"generated for {document.name}"
    return state


NDA_A = FakeUpload(b"1. Term. Two years.", "contract.pdf")
NDA_B = FakeUpload(b"1. Term. Perpetual, and a non-compete.", "contract.pdf")


print("\n--- What counts as per-document state ---")

# If a new generated-on-demand key is added to the app and not listed, it
# inherits exactly the bug this file exists to prevent.
check("negotiations are per-document", "negotiations" in PER_DOCUMENT, True)
check("the draft email is per-document", "email" in PER_DOCUMENT, True)


print("\n--- A different document ---")

state = worked_on(NDA_A)
check("setup: the session holds A's work",
      all(key in state for key in PER_DOCUMENT), True)

forget(state, NDA_B)
check("B's upload drops A's negotiations", "negotiations" in state, False)
check("B's upload drops A's draft email", "email" in state, False)

# Same filename, different bytes. This is the case a name-based check misses,
# and it is the common one -- every second NDA is called contract.pdf.
check("identity is content, not filename", NDA_A.name, NDA_B.name)


print("\n--- The same document again ---")

state = worked_on(NDA_A)
forget(state, NDA_A)
check("re-running the same document keeps its negotiations",
      state.get("negotiations"), "generated for contract.pdf")
check("re-running the same document keeps its email",
      state.get("email"), "generated for contract.pdf")

# Byte-identical content re-uploaded under a different name is the same
# analysis, and the work has already been paid for.
same_bytes = FakeUpload(NDA_A.getvalue(), "renamed-copy.pdf")
forget(state, same_bytes)
check("identical bytes under a new name are not a new document",
      state.get("negotiations"), "generated for contract.pdf")


print("\n--- Removing the file ---")

state = worked_on(NDA_A)
forget(state, None)
check("clearing the uploader drops the negotiations", "negotiations" in state, False)
check("clearing the uploader drops the email", "email" in state, False)
check("no document means no document key", state.get("document_key"), None)

# And going straight back to A afterwards is a change again, so nothing stale
# can survive a remove-then-reupload cycle.
state["negotiations"] = "stale"
forget(state, NDA_A)
check("re-uploading after a clear starts fresh", "negotiations" in state, False)


print("\n--- Unrelated state is left alone ---")

state = worked_on(NDA_A)
state["some_widget_key"] = "keep me"
forget(state, NDA_B)
check("only per-document keys are dropped",
      state.get("some_widget_key"), "keep me")


print("\n" + "=" * 60)
if failures:
    print(f"{len(failures)} FAILED: {', '.join(failures)}")
    raise SystemExit(1)
print("All session-state tests passed.")
