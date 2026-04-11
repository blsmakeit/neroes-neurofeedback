"""
data_loader.py
──────────────
Loads and validates NeroesSession_Data into a clean pandas DataFrame.

Handles:
- SessionInfo.json (session-level metadata)
- SubSessions/{id}/Info.json (subsession metadata)
- SubSessions/{id}/Data.csv  (signal data)
"""

import json
import warnings
from pathlib import Path

import pandas as pd

# ── Constants ──────────────────────────────────────────────────────────────────
DEFAULT_DATA_DIR = Path("data/raw/NeroesSession_Data")
TARGET_COL       = "ProtocolValue"
SUBSESSION_COL   = "subsession"


# ── Public API ─────────────────────────────────────────────────────────────────

def load_session(data_dir: str | Path = DEFAULT_DATA_DIR) -> dict:
    """
    Load the full session: metadata + all subsession DataFrames.

    Returns
    -------
    dict with keys:
        'session_info'     : dict  — top-level SessionInfo.json
        'subsession_infos' : dict  — {ss_id: info_dict}
        'data'             : pd.DataFrame — all subsessions combined
    """
    data_dir = Path(data_dir)
    _validate_structure(data_dir)

    session_info      = _load_json(data_dir / "SessionInfo.json")
    subsession_infos  = _load_subsession_infos(data_dir / "SubSessions")
    df                = _load_subsession_csvs(data_dir / "SubSessions")

    _validate_target(df)

    return {
        "session_info":     session_info,
        "subsession_infos": subsession_infos,
        "data":             df,
    }


def load_combined_df(data_dir: str | Path = DEFAULT_DATA_DIR) -> pd.DataFrame:
    """Convenience: return just the combined DataFrame."""
    return load_session(data_dir)["data"]


# ── Internal helpers ───────────────────────────────────────────────────────────

def _validate_structure(data_dir: Path) -> None:
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    if not (data_dir / "SessionInfo.json").exists():
        raise FileNotFoundError(f"SessionInfo.json missing in {data_dir}")
    subsessions_dir = data_dir / "SubSessions"
    if not subsessions_dir.exists():
        raise FileNotFoundError(f"SubSessions/ directory missing in {data_dir}")


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_subsession_infos(subsessions_dir: Path) -> dict:
    infos = {}
    for ss_dir in sorted(subsessions_dir.iterdir(), key=lambda x: int(x.name)):
        if not ss_dir.is_dir():
            continue
        try:
            ss_id = int(ss_dir.name)
        except ValueError:
            continue
        info_path = ss_dir / "Info.json"
        if info_path.exists():
            infos[ss_id] = _load_json(info_path)
    return infos


def _load_subsession_csvs(subsessions_dir: Path) -> pd.DataFrame:
    frames = []
    for ss_dir in sorted(subsessions_dir.iterdir(), key=lambda x: int(x.name)):
        if not ss_dir.is_dir():
            continue
        try:
            ss_id = int(ss_dir.name)
        except ValueError:
            continue
        csv_path = ss_dir / "Data.csv"
        if not csv_path.exists():
            warnings.warn(f"Data.csv missing for subsession {ss_id}")
            continue
        df = pd.read_csv(csv_path)
        df[SUBSESSION_COL] = ss_id
        df["sample_idx"]   = range(len(df))
        frames.append(df)

    if not frames:
        raise ValueError("No Data.csv files found.")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values([SUBSESSION_COL, "sample_idx"]).reset_index(drop=True)
    return combined


def _validate_target(df: pd.DataFrame) -> None:
    if TARGET_COL not in df.columns:
        warnings.warn(
            f"Target column '{TARGET_COL}' not found. "
            f"Available columns: {list(df.columns)}"
        )
