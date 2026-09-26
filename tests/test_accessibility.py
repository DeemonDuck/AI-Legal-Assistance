"""Offline WCAG contrast checks on the shipped colour palette.

WHY THIS IS A TEST AND NOT A NOTE IN THE README

Colour contrast is the one accessibility property that can be settled by
arithmetic rather than opinion, which makes it exactly the kind of claim this
project is supposed to make checkable. "The palette is accessible" is an
assertion nobody can verify; "every badge clears 4.5:1 against its text colour,
here is the ratio" is a fact, and this file is what keeps it true after someone
decides the amber looked washed out and nudges it back.

It caught one real failure: the MODERATE badge shipped at #b26b00, which
measures 4.20:1 against white and fails AA for 12px text.

WHAT IS MEASURED

  Badges      -- white text on each severity colour, from app.py itself.
  Page theme  -- body and primary colours from .streamlit/config.toml.
  Redundancy  -- that severity is carried by WORDS as well as colour, since a
                 reader who cannot separate the red badge from the amber one
                 still has to be able to tell a high risk from a moderate one.

The palette is read out of the real files by AST and TOML rather than copied
here. A copy would drift, and a test that checks a stale copy of the palette
passes while the shipped one fails.

Run:  py tests/test_accessibility.py
"""

from __future__ import annotations

import ast
import html
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from legal_ai.scoring import BOOLEAN_RULES, NUMERIC_RULES, Severity  # noqa: E402

failures: list[str] = []


def check(label: str, actual, expected) -> None:
    if actual == expected:
        print(f"  PASS  {label}")
    else:
        print(f"  FAIL  {label}: expected {expected!r}, got {actual!r}")
        failures.append(label)


# --- WCAG 2.1 relative luminance and contrast -------------------------------
# Formulae from WCAG 2.1 (w3.org/TR/WCAG21/#dfn-relative-luminance and
# #dfn-contrast-ratio), written out rather than pulled from a library so the
# arithmetic is visible in review -- the same reason corpus.py hand-rolls its
# percentiles.

AA_NORMAL_TEXT = 4.5  # 3.0 would apply only to text >=18.66px bold or >=24px.


def _relative_luminance(hex_colour: str) -> float:
    digits = hex_colour.lstrip("#")
    channels = [int(digits[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    linear = [
        c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
        for c in channels
    ]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def contrast_ratio(foreground: str, background: str) -> float:
    lighter, darker = sorted(
        (_relative_luminance(foreground), _relative_luminance(background)),
        reverse=True,
    )
    return (lighter + 0.05) / (darker + 0.05)


def meets_aa(foreground: str, background: str) -> bool:
    return contrast_ratio(foreground, background) >= AA_NORMAL_TEXT


# --- Read the palette that actually ships ------------------------------------


def load_from_app(*names: str) -> dict:
    """Execute just the named top-level definitions from app.py.

    app.py runs Streamlit at module scope, so importing it would build the
    whole page -- and would make this suite need a Streamlit install to check
    some hex strings. Lifting the definitions out by AST tests the code that
    actually ships, rather than a copy that can drift.
    """
    tree = ast.parse((ROOT / "app.py").read_text(encoding="utf-8"))

    wanted = [
        node for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name in names)
        or (isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id in names for t in node.targets))
    ]

    found = {
        n.name if isinstance(n, ast.FunctionDef) else n.targets[0].id for n in wanted
    }
    if set(names) - found:
        raise AssertionError(
            f"app.py no longer defines {sorted(set(names) - found)} at module "
            "level -- renamed, or moved inside another scope?"
        )

    namespace: dict = {
        "html": html,
        "Severity": Severity,
        "NUMERIC_RULES": NUMERIC_RULES,
        "BOOLEAN_RULES": BOOLEAN_RULES,
    }
    exec(compile(ast.Module(body=wanted, type_ignores=[]), "app.py", "exec"), namespace)
    return namespace


def severity_style() -> dict[str, tuple[str, str]]:
    """SEVERITY_STYLE as {severity name: (colour, label)}.

    Re-keyed by name because the loaded dict is keyed by the real Severity
    enum, and the checks below read better against plain strings.
    """
    return {
        severity.name: value
        for severity, value in load_from_app("SEVERITY_STYLE")["SEVERITY_STYLE"].items()
    }

    raise AssertionError("SEVERITY_STYLE not found in app.py -- was it renamed?")


print("\n--- Contrast formula ---")

# Anchored against the two ratios WCAG fixes by definition, so a mistake in the
# luminance maths shows up here rather than silently passing the real palette.
check("black on white is 21:1", round(contrast_ratio("#000000", "#ffffff"), 1), 21.0)
check("a colour against itself is 1:1",
      round(contrast_ratio("#777777", "#777777"), 1), 1.0)
check("ratio is symmetric",
      round(contrast_ratio("#b3261e", "#ffffff"), 3),
      round(contrast_ratio("#ffffff", "#b3261e"), 3))


print("\n--- Severity badges (white text on a coloured chip) ---")

BADGE_TEXT = "#ffffff"
styles = severity_style()

check("all four severities are styled", len(styles), 4)

for severity, (colour, _label) in sorted(styles.items()):
    ratio = contrast_ratio(BADGE_TEXT, colour)
    check(f"{severity} badge {colour} clears AA ({ratio:.2f}:1)",
          meets_aa(BADGE_TEXT, colour), True)

# Severity must survive being read in greyscale. Roughly 1 in 12 men cannot
# reliably separate the red chip from the amber one, and a risk indicator only
# some readers can read is not an indicator.
for severity, (_, label) in sorted(styles.items()):
    check(f"{severity} states its level in words", bool(label.strip()), True)

check("badge labels are distinct",
      len({label for _, label in styles.values()}), len(styles))


print("\n--- Page theme (.streamlit/config.toml) ---")

theme = tomllib.loads(
    (ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8")
)["theme"]

background = theme["backgroundColor"]
secondary = theme["secondaryBackgroundColor"]

body_ratio = contrast_ratio(theme["textColor"], background)
check(f"body text on the page background clears AA ({body_ratio:.2f}:1)",
      meets_aa(theme["textColor"], background), True)

card_ratio = contrast_ratio(theme["textColor"], secondary)
check(f"body text on cards and the sidebar clears AA ({card_ratio:.2f}:1)",
      meets_aa(theme["textColor"], secondary), True)

# primaryColor is link and accent text, not just a button fill, so it carries
# meaning and has to be readable at body size.
primary_ratio = contrast_ratio(theme["primaryColor"], background)
check(f"accent colour on the page background clears AA ({primary_ratio:.2f}:1)",
      meets_aa(theme["primaryColor"], background), True)


print("\n--- Nothing internal reaches the screen ---")

# The pipeline names terms `non_solicit_months`. That string was reaching the
# UI in three places -- negotiation expander titles, the rarity-vs-risk list,
# and the "not checked" caption. A screen reader reads it out as "non
# underscore solicit underscore months", and it is poor plain English for
# everyone else, in a tool whose whole promise is plain English.
app = load_from_app("_readable", "_severity_badge", "SEVERITY_STYLE")
readable, badge = app["_readable"], app["_severity_badge"]

for attribute in list(NUMERIC_RULES) + list(BOOLEAN_RULES):
    label = readable(attribute)
    check(f"{attribute} is shown as words", "_" not in label, True)

check("the label is the one the scorer already uses",
      readable("non_solicit_months"), "Non-solicitation period")
check("carve-out identifiers are spelled out",
      readable("carve_out:independently_developed"),
      "Missing exclusion: independently developed")
# An attribute added later must degrade to readable text, not leak raw.
check("an unmapped attribute still loses its underscores",
      readable("some_future_attribute"), "some future attribute")


print("\n--- The severity chip names what it describes ---")

# The chip renders in its own column beside the finding's title, so without
# this a screen reader reaches a bare "HIGH RISK" with nothing saying what is.
# A roleless <span> does not reliably expose aria-label, hence role='img'.
markup = badge(Severity.HIGH, "Confidentiality term")
check("the chip carries an accessible name",
      "aria-label='HIGH RISK: Confidentiality term'" in markup, True)
check("the chip has a role, or aria-label may be ignored",
      "role='img'" in markup, True)
check("the visible text is still there for sighted users",
      ">HIGH RISK</span>" in markup, True)
check("it degrades to the bare level when nothing is described",
      "aria-label='MODERATE'" in badge(Severity.MEDIUM), True)

# The described text comes from the document, so it can contain quotes and
# ampersands -- unescaped, either would break out of the attribute.
check("the accessible name is HTML-escaped",
      "aria-label='MODERATE: O&#x27;Brien &amp; Co'" in badge(Severity.MEDIUM, "O'Brien & Co"),
      True)


print("\n" + "=" * 60)
if failures:
    print(f"{len(failures)} FAILED: {', '.join(failures)}")
    raise SystemExit(1)
print("All accessibility tests passed.")
