"""
run_pipeline.py
───────────────
End-to-end pipeline: load → features → train → evaluate.
Run after all notebooks have been completed and src/ modules are finalised.

Usage:
    python scripts/run_pipeline.py
"""

import sys
from pathlib import Path

# ── Add src/ to path ──────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_loader import load_session
from features    import build_features
from baselines   import compare_baselines
from evaluation  import regression_metrics


def main():
    print("=" * 60)
    print("  NEROES NEUROFEEDBACK PIPELINE")
    print("=" * 60)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("\n[1/4] Loading session data...")
    session = load_session("data/raw/NeroesSession_Data")
    df_raw  = session["data"]
    print(f"      Loaded {len(df_raw):,} rows × {df_raw.shape[1]} cols")
    print(f"      SubSessions: {sorted(df_raw['subsession'].unique())}")

    # ── 2. Feature engineering ────────────────────────────────────────────────
    print("\n[2/4] Building features...")
    df_feat = build_features(df_raw)
    df_feat.to_parquet("data/processed/features.parquet", index=False)
    print(f"      Feature matrix: {df_feat.shape}")
    print(f"      Saved to data/processed/features.parquet")

    # ── 3. Baselines ──────────────────────────────────────────────────────────
    print("\n[3/4] Running baselines...")
    baseline_results = compare_baselines(df_raw)
    print(baseline_results.to_string())

    # ── 4. Placeholder for trained model evaluation ───────────────────────────
    print("\n[4/4] Model evaluation — run notebooks 03–06 first.")
    print("\n✅ Pipeline complete.")


if __name__ == "__main__":
    main()
