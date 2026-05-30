"""BugInsight AI — Synthetic Bug Report Generator.

Generates a small synthetic dataset (~2000 samples) across four
severity levels for **smoke testing only** — not for real research.
Output is written to ``data/raw/synthetic_bugs.csv``.

Usage::

    python scripts/generate_synthetic_data.py
"""

import csv
import logging
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Bootstrap: ensure project root is importable
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from configs.config_loader import load_config  # noqa: E402
from utils import set_seed, setup_logging  # noqa: E402

logger = logging.getLogger(__name__)


# ===================================================================
# Template banks — varied vocabulary for realistic-looking reports
# ===================================================================

_CRITICAL_TITLES: List[str] = [
    "Application crashes on startup after update",
    "Complete data loss when saving large files",
    "Server unresponsive under normal load",
    "Authentication bypass allows unauthorized access",
    "Database corruption on concurrent writes",
    "Memory leak causes OOM crash within 10 minutes",
    "Payment processing fails silently — no error shown",
    "Security vulnerability in session token generation",
    "System hangs on boot with kernel panic",
    "Unhandled exception crashes entire cluster",
    "Critical API endpoint returns 500 for all users",
    "Data migration script corrupts production records",
    "SSL certificate validation completely disabled",
    "Race condition causes permanent data inconsistency",
    "Build pipeline produces broken artifacts silently",
]

_MAJOR_TITLES: List[str] = [
    "Search results are missing recent entries",
    "Export to PDF produces garbled output",
    "User preferences are not persisted across sessions",
    "Notifications are delayed by several hours",
    "Dashboard charts render incorrectly on Firefox",
    "Bulk import skips every other row without warning",
    "API rate limiting returns wrong HTTP status code",
    "Sorting by date produces wrong ordering",
    "Email templates contain broken HTML links",
    "File upload fails for files larger than 50 MB",
    "Login page redirect loop on certain browsers",
    "Pagination returns duplicate entries on page 2",
    "Report generator omits last column from query",
    "Webhook payload schema changed without versioning",
    "Cache invalidation not triggered on record update",
]

_MINOR_TITLES: List[str] = [
    "Tooltip text overflows container on narrow screens",
    "Breadcrumb trail shows wrong parent category",
    "Footer links have inconsistent hover colour",
    "Settings page takes 3 seconds to render",
    "Dropdown menu flickers when hovering quickly",
    "Placeholder text not translated in French locale",
    "Icon alignment off by two pixels in sidebar",
    "Tab order skips the search field",
    "Scroll position resets after closing modal dialog",
    "Loading spinner appears briefly on fast connections",
    "Contrast ratio fails WCAG AA on help page",
    "Avatar image aspect ratio distorted on profile page",
    "Autocomplete shows stale suggestions after clear",
    "Date picker defaults to wrong timezone",
    "Mobile menu button has small tap target",
]

_TRIVIAL_TITLES: List[str] = [
    "Typo in confirmation dialog: 'sucess' → 'success'",
    "Extra whitespace in footer copyright notice",
    "Console log statement left in production build",
    "README badge links to wrong CI pipeline",
    "Comment typo in configuration module",
    "Variable named 'tmp' should be more descriptive",
    "Unused import in utility module",
    "Docstring missing return type description",
    "Version number not bumped in package.json",
    "License header missing from new source file",
    "Log message has inconsistent capitalisation",
    "Test fixture name does not follow naming convention",
    "Changelog entry missing for v2.3.1 release",
    "CSS class name uses camelCase instead of kebab-case",
    "Translation key missing for OK button label",
]

_DESCRIPTION_TEMPLATES: Dict[str, List[str]] = {
    "Critical": [
        (
            "Steps to reproduce:\n1. {action}\n2. {action2}\n3. Observe {outcome}\n\n"
            "Expected: The system should remain stable.\n"
            "Actual: Complete {failure_type}. No recovery possible without restart.\n"
            "Impact: All users affected. Production is down."
        ),
        (
            "Environment: {env}\n\n"
            "When {trigger}, the application {critical_effect}. "
            "This is a P0 blocker. Stack trace attached.\n\n"
            "```\n{error_class}: {error_msg}\n  at {location}\n```"
        ),
        (
            "Severity justification: {critical_effect} affecting 100%% of users. "
            "Workaround: None. Data integrity is at risk. "
            "Immediate patch required before next release."
        ),
    ],
    "Major": [
        (
            "Steps to reproduce:\n1. {action}\n2. {action2}\n\n"
            "Expected: {expected}.\n"
            "Actual: {major_effect}. Workaround exists but is cumbersome.\n"
            "Frequency: Reproducible every time."
        ),
        (
            "This issue affects the {component} module. When {trigger}, "
            "{major_effect}. Approximately 30%% of users are impacted. "
            "Temporary fix: {workaround}."
        ),
        (
            "Observed on {env}. The {component} feature produces {major_effect} "
            "under normal usage. This blocks the QA sign-off for the sprint."
        ),
    ],
    "Minor": [
        (
            "On {env}, the {component} shows {minor_effect}. "
            "This is a cosmetic / usability issue and does not block core workflows. "
            "Expected: {expected}."
        ),
        (
            "Steps:\n1. Navigate to {component}\n2. {action}\n\n"
            "Observed: {minor_effect}. Low priority but affects user experience."
        ),
        (
            "The UI element in {component} behaves unexpectedly: {minor_effect}. "
            "No data is lost. Workaround: {workaround}."
        ),
    ],
    "Trivial": [
        (
            "Found a minor cosmetic issue: {trivial_effect} in {component}. "
            "No functional impact. Fix at convenience."
        ),
        (
            "Code review finding: {trivial_effect}. "
            "Suggestion: {suggestion}. Non-blocking."
        ),
        (
            "While reviewing {component}, noticed {trivial_effect}. "
            "This has zero user-facing impact but should be cleaned up."
        ),
    ],
}

_FILL_INS: Dict[str, List[str]] = {
    "action": [
        "Open the application", "Navigate to the dashboard",
        "Click the submit button", "Upload a file", "Enter valid credentials",
        "Trigger the batch job", "Run the nightly sync", "Open settings page",
        "Switch between tabs rapidly", "Send a POST request via the API",
    ],
    "action2": [
        "Wait for the page to load", "Select an item from the list",
        "Scroll to the bottom", "Press Enter in the search bar",
        "Toggle the advanced filter", "Click Save", "Switch to dark mode",
    ],
    "outcome": [
        "the application crashes", "data is lost", "the page goes blank",
        "an error dialog appears", "the browser tab freezes",
    ],
    "trigger": [
        "processing a large dataset", "handling concurrent requests",
        "switching locales", "restoring from backup", "using the bulk editor",
        "performing a full-text search", "exporting analytics",
    ],
    "failure_type": [
        "system crash", "data corruption", "service outage",
        "unrecoverable error", "cascading failure",
    ],
    "critical_effect": [
        "crashes immediately", "loses all unsaved data",
        "becomes completely unresponsive", "exposes sensitive user data",
        "corrupts the database schema",
    ],
    "major_effect": [
        "returns incorrect results", "fails with a misleading error message",
        "drops user input silently", "renders the page with broken layout",
        "does not persist changes to the database",
    ],
    "minor_effect": [
        "slight misalignment of elements", "delayed response on interaction",
        "wrong default value in the dropdown", "text truncation on small screens",
        "flicker when transitioning between views",
    ],
    "trivial_effect": [
        "a typo in the tooltip", "inconsistent capitalisation in labels",
        "an unused CSS class", "a leftover debug log statement",
        "a missing period at the end of a sentence",
    ],
    "env": [
        "Windows 11 / Chrome 125", "Ubuntu 22.04 / Firefox 128",
        "macOS Sonoma / Safari 18", "Kaggle T4 notebook environment",
        "Docker container (python:3.11-slim)",
    ],
    "component": [
        "dashboard", "user-settings", "analytics module",
        "file-manager", "notification service", "authentication gateway",
        "search index", "billing module", "report builder",
    ],
    "expected": [
        "Correct data displayed", "Feature works as documented",
        "No visual glitches", "Smooth interaction",
    ],
    "workaround": [
        "Clear the browser cache", "Use a different browser",
        "Manually refresh the page", "Re-enter the data",
        "Disable the problematic plugin",
    ],
    "suggestion": [
        "Rename the variable for clarity",
        "Add the missing docstring",
        "Remove the dead code",
        "Fix the typo",
    ],
    "error_class": [
        "NullPointerException", "RuntimeError", "TypeError",
        "ConnectionResetError", "MemoryError",
    ],
    "error_msg": [
        "Cannot read property 'id' of undefined",
        "Maximum call stack size exceeded",
        "Segmentation fault (core dumped)",
        "Connection refused on port 5432",
    ],
    "location": [
        "src/core/engine.py:142", "lib/handlers/auth.js:88",
        "app/services/data_service.rb:203", "pkg/api/router.go:67",
    ],
}

_PROJECTS: List[str] = [
    "Platform-Core", "WebUI", "MobileApp", "Analytics-Service",
    "Auth-Gateway", "DataPipeline", "CLI-Tools", "InfraOps",
]


# ===================================================================
# Generator logic
# ===================================================================

def _fill_template(template: str) -> str:
    """Replace ``{placeholder}`` tokens with random fill-in values.

    Args:
        template: Template string with ``{key}`` placeholders.

    Returns:
        Filled-in string.
    """
    result = template
    for key, values in _FILL_INS.items():
        placeholder = "{" + key + "}"
        while placeholder in result:
            result = result.replace(placeholder, random.choice(values), 1)
    return result


def generate_samples(
    n_total: int = 2000,
    seed: int = 42,
) -> List[Dict[str, str]]:
    """Generate synthetic bug report records.

    Samples are distributed across severity levels with a slight
    imbalance to mimic real-world datasets:
        - Critical: 15 %
        - Major:    30 %
        - Minor:    35 %
        - Trivial:  20 %

    Args:
        n_total: Total number of samples to generate.
        seed: Random seed for reproducibility.

    Returns:
        List of record dictionaries with unified schema columns.
    """
    set_seed(seed)

    severity_weights: List[Tuple[str, float, List[str]]] = [
        ("Critical", 0.15, _CRITICAL_TITLES),
        ("Major", 0.30, _MAJOR_TITLES),
        ("Minor", 0.35, _MINOR_TITLES),
        ("Trivial", 0.20, _TRIVIAL_TITLES),
    ]

    records: List[Dict[str, str]] = []
    bug_id_counter = 1

    for severity, fraction, title_pool in severity_weights:
        n_samples = int(n_total * fraction)
        templates = _DESCRIPTION_TEMPLATES[severity]

        for _ in range(n_samples):
            title = random.choice(title_pool)
            desc_template = random.choice(templates)
            description = _fill_template(desc_template)
            project = random.choice(_PROJECTS)

            records.append(
                {
                    "bug_id": f"SYN-{bug_id_counter:05d}",
                    "title": title,
                    "description": description,
                    "severity": severity,
                    "project": project,
                }
            )
            bug_id_counter += 1

    # Shuffle to avoid ordering artifacts
    random.shuffle(records)
    logger.info("Generated %d synthetic bug reports", len(records))
    return records


def write_csv(
    records: List[Dict[str, str]],
    output_path: Path,
) -> None:
    """Write synthetic records to a CSV file.

    Args:
        records: List of record dictionaries.
        output_path: Destination CSV path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["bug_id", "title", "description", "severity", "project"]

    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    logger.info("Synthetic data written → %s (%d records)", output_path, len(records))


# ===================================================================
# Main
# ===================================================================

def main() -> None:
    """Entry point: generate synthetic data and write to disk."""
    config = load_config()
    setup_logging(project_root=config.project_root)

    seed = config.seed
    raw_dir = config.get_path("dataset.raw_dir")
    output_path = raw_dir / "synthetic_bugs.csv"

    logger.info("=" * 60)
    logger.info("BugInsight AI — Synthetic Data Generator")
    logger.info("=" * 60)
    logger.info("Output path : %s", output_path)
    logger.info("Seed        : %d", seed)

    records = generate_samples(n_total=2000, seed=seed)
    write_csv(records, output_path)

    # Quick sanity summary
    from collections import Counter

    dist = Counter(r["severity"] for r in records)
    logger.info("Severity distribution: %s", dict(dist))
    logger.info("Done.")


if __name__ == "__main__":
    main()
