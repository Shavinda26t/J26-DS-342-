"""
Shared path resolution for this component.

Scripts must not hard-code machine-specific absolute paths, which
break as soon as the repository is cloned somewhere else.

- COMPONENT_ROOT is derived from this file's location, so component
  paths (results/, data/, config/) resolve from any checkout or drive.
- RAW_DATA_DIR points at the raw SMART extract that lives OUTSIDE the
  repository (it is far too large to commit). It defaults to the
  original location and can be overridden with the HDD_RAW_DATA_DIR
  environment variable:

      set HDD_RAW_DATA_DIR=D:\path\to\data_Q1_2023

Use require_raw_data_dir() in entry points that cannot run without
the raw data: it fails fast with an actionable message instead of a
confusing "0 files found" further down the pipeline.
"""

import os
from pathlib import Path


# This file lives in <component>/scripts/, so its parent is the
# component root regardless of the current working directory.
COMPONENT_ROOT = Path(__file__).resolve().parents[1]

# Repository root (two levels up: <repo>/02_component_2_digital_twin).
REPO_ROOT = COMPONENT_ROOT.parent

# Original machine-specific location, kept only as a fallback.
_DEFAULT_RAW_DATA_DIR = Path(
    r"D:\db module 4th yera 1sem\data_Q1_2023\data_Q1_2023"
)

RAW_DATA_DIR = Path(
    os.environ.get("HDD_RAW_DATA_DIR", str(_DEFAULT_RAW_DATA_DIR))
)


def require_raw_data_dir():
    """
    Return RAW_DATA_DIR, failing fast with instructions when the
    raw SMART extract is not where this machine expects it.
    """

    if not RAW_DATA_DIR.is_dir():
        raise FileNotFoundError(
            "Raw SMART data directory not found:\n"
            f"  {RAW_DATA_DIR}\n"
            "Set the HDD_RAW_DATA_DIR environment variable to the "
            "folder containing the data_Q1_2023 CSV extracts, e.g.\n"
            "  set HDD_RAW_DATA_DIR=D:\\path\\to\\data_Q1_2023"
        )

    return RAW_DATA_DIR
