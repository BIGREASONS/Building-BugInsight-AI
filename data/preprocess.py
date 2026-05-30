"""BugInsight AI — Bug Report Preprocessing Pipeline.

Loads raw bug report data from CSV, JSON, or Bugzilla XML exports in
``data/raw/``, standardises every record to a unified schema, applies
severity mapping from ``config.yaml``, cleans text, and writes a single
processed CSV to ``data/processed/bugs_standardized.csv``.

Usage::

    from data.preprocess import preprocess_all
    stats = preprocess_all()
"""

import html
import json
import logging
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from configs.config_loader import Config, load_config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
UNIFIED_COLUMNS: List[str] = [
    "bug_id",
    "title",
    "description",
    "severity",
    "project",
]


# ===================================================================
# Text-cleaning utilities
# ===================================================================

def strip_html_tags(text: str) -> str:
    """Remove HTML / XML tags and decode HTML entities.

    Args:
        text: Raw text potentially containing HTML markup.

    Returns:
        Cleaned plain-text string.
    """
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    return text


def strip_urls(text: str) -> str:
    """Remove URLs (http, https, ftp) from text.

    Args:
        text: Raw text potentially containing URLs.

    Returns:
        Text with URLs replaced by a single space.
    """
    return re.sub(r"https?://\S+|ftp://\S+", " ", text)


def normalise_whitespace(text: str) -> str:
    """Collapse excessive whitespace and strip leading/trailing blanks.

    Args:
        text: Text with potentially irregular spacing.

    Returns:
        Cleaned text with single spaces between words.
    """
    return re.sub(r"\s+", " ", text).strip()


def clean_text(text: str) -> str:
    """Run the full text-cleaning pipeline on a single string.

    Steps:
        1. Strip HTML tags and decode entities.
        2. Remove URLs.
        3. Normalise whitespace.

    Args:
        text: Raw text.

    Returns:
        Cleaned text.
    """
    if not isinstance(text, str):
        return ""
    text = strip_html_tags(text)
    text = strip_urls(text)
    text = normalise_whitespace(text)
    return text


# ===================================================================
# File-level loaders
# ===================================================================

def _load_csv(filepath: Path) -> pd.DataFrame:
    """Load a CSV file into a DataFrame.

    The loader attempts to auto-detect the relevant columns by looking for
    common column-name variants (``summary`` → ``title``, ``bug_severity`` →
    ``severity``, etc.) and renames them to the unified schema.

    Args:
        filepath: Path to the CSV file.

    Returns:
        DataFrame with columns mapped towards the unified schema.
    """
    logger.info("Loading CSV: %s", filepath)
    df = pd.read_csv(filepath, low_memory=False)
    df.columns = [c.strip().lower() for c in df.columns]
    logger.info("  → %d rows, columns: %s", len(df), list(df.columns))

    # Canonical rename mapping (source_col → unified_col)
    rename_map: Dict[str, str] = {}
    col_set = set(df.columns)

    # bug_id
    for candidate in ("bug_id", "id", "issue_id", "issue_key", "bug_number"):
        if candidate in col_set:
            rename_map[candidate] = "bug_id"
            break

    # title
    for candidate in ("title", "summary", "short_desc", "subject", "issue_title"):
        if candidate in col_set:
            rename_map[candidate] = "title"
            break

    # description
    for candidate in ("description", "long_desc", "body", "comment", "text"):
        if candidate in col_set:
            rename_map[candidate] = "description"
            break

    # severity
    for candidate in ("severity", "bug_severity", "priority", "issue_type"):
        if candidate in col_set:
            rename_map[candidate] = "severity"
            break

    # project
    for candidate in ("project", "product", "component", "repo"):
        if candidate in col_set:
            rename_map[candidate] = "project"
            break

    df = df.rename(columns=rename_map)
    return df


def _load_json(filepath: Path) -> pd.DataFrame:
    """Load a JSON file (array-of-objects) into a DataFrame.

    Args:
        filepath: Path to the JSON file.

    Returns:
        DataFrame with columns mapped towards the unified schema.
    """
    logger.info("Loading JSON: %s", filepath)
    with open(filepath, "r", encoding="utf-8") as fh:
        records = json.load(fh)

    if isinstance(records, dict) and "bugs" in records:
        records = records["bugs"]

    df = pd.DataFrame(records)
    df.columns = [c.strip().lower() for c in df.columns]
    logger.info("  → %d rows, columns: %s", len(df), list(df.columns))

    # Apply the same rename logic as CSV
    rename_map: Dict[str, str] = {}
    col_set = set(df.columns)

    for candidate in ("bug_id", "id", "issue_id"):
        if candidate in col_set:
            rename_map[candidate] = "bug_id"
            break
    for candidate in ("title", "summary", "short_desc"):
        if candidate in col_set:
            rename_map[candidate] = "title"
            break
    for candidate in ("description", "long_desc", "body", "text"):
        if candidate in col_set:
            rename_map[candidate] = "description"
            break
    for candidate in ("severity", "bug_severity", "priority"):
        if candidate in col_set:
            rename_map[candidate] = "severity"
            break
    for candidate in ("project", "product", "component"):
        if candidate in col_set:
            rename_map[candidate] = "project"
            break

    df = df.rename(columns=rename_map)
    return df


def _load_bugzilla_xml(filepath: Path) -> pd.DataFrame:
    """Parse a Bugzilla XML export into a DataFrame.

    Bugzilla exports typically contain ``<bug>`` elements each with child
    elements ``<bug_id>``, ``<short_desc>``, ``<bug_severity>``,
    ``<product>``, and one or more ``<long_desc>`` blocks whose first
    entry is the original description.

    Args:
        filepath: Path to the XML file.

    Returns:
        DataFrame with the unified column naming.
    """
    logger.info("Loading Bugzilla XML: %s", filepath)
    tree = ET.parse(filepath)  # noqa: S314
    root = tree.getroot()

    records: List[Dict[str, str]] = []
    for bug_elem in root.iter("bug"):
        bug_id = _xml_text(bug_elem, "bug_id")
        title = _xml_text(bug_elem, "short_desc")
        severity = _xml_text(bug_elem, "bug_severity")
        project = _xml_text(bug_elem, "product")

        # First <long_desc> child typically holds the initial description
        long_descs = bug_elem.findall("long_desc")
        description = ""
        if long_descs:
            desc_elem = long_descs[0].find("thetext")
            if desc_elem is not None and desc_elem.text:
                description = desc_elem.text

        records.append(
            {
                "bug_id": bug_id,
                "title": title,
                "description": description,
                "severity": severity,
                "project": project,
            }
        )

    df = pd.DataFrame(records, columns=UNIFIED_COLUMNS)
    logger.info("  → %d bug records extracted", len(df))
    return df


def _xml_text(parent: ET.Element, tag: str) -> str:
    """Safely extract the text of a child XML element.

    Args:
        parent: Parent XML element.
        tag: Tag name of the child element.

    Returns:
        Text content or an empty string.
    """
    elem = parent.find(tag)
    if elem is not None and elem.text:
        return elem.text.strip()
    return ""


# ===================================================================
# Core preprocessing pipeline
# ===================================================================

def load_raw_datasets(raw_dir: Path) -> pd.DataFrame:
    """Discover and load all raw datasets from a directory.

    Supports ``.csv``, ``.json``, and ``.xml`` files.  All loaded frames
    are concatenated and returned.

    Args:
        raw_dir: Path to the directory containing raw data files.

    Returns:
        Combined DataFrame from all discovered files.

    Raises:
        FileNotFoundError: If the raw directory does not exist.
        ValueError: If no supported files are found.
    """
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")

    loaders = {
        ".csv": _load_csv,
        ".json": _load_json,
        ".xml": _load_bugzilla_xml,
    }

    frames: List[pd.DataFrame] = []
    for filepath in sorted(raw_dir.iterdir()):
        suffix = filepath.suffix.lower()
        if suffix in loaders:
            df = loaders[suffix](filepath)
            frames.append(df)

    if not frames:
        raise ValueError(
            f"No CSV, JSON, or XML files found in {raw_dir}. "
            "Please place raw bug-report data there first."
        )

    combined = pd.concat(frames, ignore_index=True)
    logger.info(
        "Combined %d files → %d total rows",
        len(frames),
        len(combined),
    )
    return combined


def standardise_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the DataFrame has exactly the unified columns.

    Missing columns are filled with empty strings.  Extra columns are
    dropped.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with exactly ``UNIFIED_COLUMNS``.
    """
    for col in UNIFIED_COLUMNS:
        if col not in df.columns:
            logger.warning("Missing column '%s' — filling with empty strings", col)
            df[col] = ""

    df = df[UNIFIED_COLUMNS].copy()
    return df


def apply_severity_mapping(
    df: pd.DataFrame,
    mapping: Dict[str, str],
) -> pd.DataFrame:
    """Map raw severity labels to standardised labels.

    Labels mapped to ``__REMOVE__`` are dropped entirely.

    Args:
        df: DataFrame with a ``severity`` column.
        mapping: ``{raw_label: standard_label}`` from config.

    Returns:
        DataFrame with remapped severity and removed rows.
    """
    initial_count = len(df)

    # Normalise raw labels to lowercase for matching
    df["severity"] = df["severity"].astype(str).str.strip().str.lower()

    # Map; keep unmapped values as-is (titlecased)
    df["severity"] = df["severity"].map(
        lambda s: mapping.get(s, s.title())  # noqa: B023
    )

    # Remove rows flagged for removal
    remove_mask = df["severity"] == "__REMOVE__"
    removed_count = remove_mask.sum()
    df = df[~remove_mask].copy()

    logger.info(
        "Severity mapping applied: %d → %d rows (%d removed as __REMOVE__)",
        initial_count,
        len(df),
        removed_count,
    )
    return df


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean text fields and drop invalid rows.

    Steps:
        1. Clean ``title`` and ``description`` via :func:`clean_text`.
        2. Drop rows where both ``title`` and ``description`` are empty.
        3. Drop rows with missing severity.
        4. Convert ``bug_id`` to string.

    Args:
        df: DataFrame in unified schema.

    Returns:
        Cleaned DataFrame.
    """
    initial_count = len(df)

    df["title"] = df["title"].apply(clean_text)
    df["description"] = df["description"].apply(clean_text)

    # Drop rows with empty text
    empty_text = (df["title"] == "") & (df["description"] == "")
    df = df[~empty_text].copy()

    # Drop rows with missing severity
    df = df[df["severity"].astype(str).str.strip() != ""].copy()

    # Ensure string bug_id
    df["bug_id"] = df["bug_id"].astype(str)

    # Drop exact duplicates
    dup_count = df.duplicated(subset=["bug_id"]).sum()
    if dup_count > 0:
        df = df.drop_duplicates(subset=["bug_id"], keep="first")
        logger.info("Dropped %d duplicate bug_id entries", dup_count)

    logger.info(
        "Cleaning: %d → %d rows (%d dropped)",
        initial_count,
        len(df),
        initial_count - len(df),
    )
    return df.reset_index(drop=True)


def compute_dataset_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute and log descriptive statistics for the processed dataset.

    Args:
        df: Processed DataFrame.

    Returns:
        Dictionary of statistics suitable for JSON serialisation.
    """
    # Class distribution
    class_dist = df["severity"].value_counts().to_dict()

    # Text-length statistics
    df["_text_len"] = (df["title"] + " " + df["description"]).str.len()
    avg_text_len = float(df["_text_len"].mean())
    median_text_len = float(df["_text_len"].median())
    max_text_len = int(df["_text_len"].max())
    min_text_len = int(df["_text_len"].min())
    df.drop(columns=["_text_len"], inplace=True)

    # Project distribution
    project_dist = df["project"].value_counts().to_dict()

    stats: Dict[str, Any] = {
        "total_samples": len(df),
        "num_classes": df["severity"].nunique(),
        "class_distribution": class_dist,
        "avg_text_length_chars": round(avg_text_len, 2),
        "median_text_length_chars": round(median_text_len, 2),
        "max_text_length_chars": max_text_len,
        "min_text_length_chars": min_text_len,
        "num_projects": df["project"].nunique(),
        "project_distribution": project_dist,
    }

    logger.info("Dataset statistics:")
    logger.info("  Total samples      : %d", stats["total_samples"])
    logger.info("  Number of classes   : %d", stats["num_classes"])
    logger.info("  Class distribution  : %s", stats["class_distribution"])
    logger.info("  Avg text length     : %.2f chars", stats["avg_text_length_chars"])
    logger.info("  Num projects        : %d", stats["num_projects"])
    return stats


# ===================================================================
# Main entry point
# ===================================================================

def preprocess_all(
    config: Optional[Config] = None,
) -> Dict[str, Any]:
    """Run the full preprocessing pipeline.

    1. Load all raw data from ``data/raw/``.
    2. Standardise to unified schema.
    3. Apply severity mapping.
    4. Clean text.
    5. Save processed CSV.
    6. Compute and return dataset statistics.

    Args:
        config: Optional pre-loaded :class:`Config` instance.
            If ``None``, loads the default configuration.

    Returns:
        Dictionary of dataset statistics.
    """
    if config is None:
        config = load_config()

    raw_dir = config.get_path("dataset.raw_dir")
    processed_file = config.get_path("dataset.processed_file")
    severity_mapping: Dict[str, str] = config.get("dataset.severity_mapping", {})

    logger.info("=" * 60)
    logger.info("BugInsight AI — Preprocessing Pipeline")
    logger.info("=" * 60)
    logger.info("Raw directory   : %s", raw_dir)
    logger.info("Output file     : %s", processed_file)

    # --- Step 1: Load ---
    df = load_raw_datasets(raw_dir)

    # --- Step 2: Standardise schema ---
    df = standardise_schema(df)

    # --- Step 3: Severity mapping ---
    df = apply_severity_mapping(df, severity_mapping)

    # --- Step 4: Clean ---
    df = clean_dataframe(df)

    # --- Step 5: Save ---
    processed_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_file, index=False, encoding="utf-8")
    logger.info("Processed data saved → %s", processed_file)

    # --- Step 6: Statistics ---
    stats = compute_dataset_statistics(df)

    # Persist statistics JSON
    metrics_dir = config.get_path("outputs.metrics_dir")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    stats_path = metrics_dir / "dataset_statistics.json"
    with open(stats_path, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2, default=str)
    logger.info("Dataset statistics saved → %s", stats_path)

    logger.info("Preprocessing complete.")
    return stats


# ===================================================================
# CLI entry point
# ===================================================================

if __name__ == "__main__":
    import sys

    # Minimal bootstrap: add project root to path
    _project_root = Path(__file__).resolve().parent.parent
    if str(_project_root) not in sys.path:
        sys.path.insert(0, str(_project_root))

    from utils import setup_logging  # noqa: E402

    setup_logging(project_root=_project_root)
    preprocess_all()
