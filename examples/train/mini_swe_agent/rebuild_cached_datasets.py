"""
Rebuild train_cached.parquet and val_small.parquet to only include instances
whose Docker images are currently cached locally via podman.

Run after any change to the local image cache.

Usage:
    python rebuild_cached_datasets.py           # rebuild both files
    python rebuild_cached_datasets.py --verify-only  # check if files are stale, exit 1 if so
"""
import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


DATA_DIR = Path.home() / "data" / "swe_gym_subset"
TRAIN_SRC = DATA_DIR / "train.parquet"
VAL_SRC = DATA_DIR / "validation.parquet"
TRAIN_OUT = DATA_DIR / "train_cached.parquet"
VAL_OUT = DATA_DIR / "val_small.parquet"
VAL_SIZE = 10


def get_cached_images() -> set:
    result = subprocess.run(
        ["podman", "images", "--format", "{{.Repository}}:{{.Tag}}"],
        capture_output=True, text=True, check=True,
    )
    return set(result.stdout.strip().split("\n"))


def get_image_name(row) -> str:
    data_source = str(row.get("data_source", "swe-gym")).lower()
    iid = row["instance_id"]
    if "swe-gym" in data_source:
        return f"docker.io/xingyaoww/sweb.eval.x86_64.{iid.replace('__', '_s_')}:latest".lower()
    else:
        return f"docker.io/swebench/sweb.eval.x86_64.{iid.replace('__', '_1776_')}:latest".lower()


def filter_cached(df: pd.DataFrame, cached: set) -> pd.DataFrame:
    images = df.apply(get_image_name, axis=1)
    mask = images.isin(cached)
    return df[mask].copy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-only", action="store_true",
                        help="Check if cached files are still valid; exit 1 if stale.")
    args = parser.parse_args()

    print("[rebuild] Getting locally cached podman images...")
    cached = get_cached_images()
    print(f"[rebuild] {len(cached)} images cached locally.")

    # --- Train ---
    train_df = pd.read_parquet(TRAIN_SRC)
    train_cached = filter_cached(train_df, cached)
    print(f"[rebuild] Train: {len(train_cached)}/{len(train_df)} instances have cached images.")

    # --- Val ---
    val_df = pd.read_parquet(VAL_SRC)
    val_cached = filter_cached(val_df, cached).head(VAL_SIZE)
    print(f"[rebuild] Val:   {len(val_cached)} instances selected (first {VAL_SIZE} with cached images).")

    if args.verify_only:
        stale = False
        if TRAIN_OUT.exists():
            existing = pd.read_parquet(TRAIN_OUT)
            missing = set(existing["instance_id"]) - set(train_cached["instance_id"])
            if missing:
                print(f"[rebuild] STALE: train_cached.parquet has {len(missing)} instances with uncached images.")
                stale = True
            else:
                print("[rebuild] train_cached.parquet OK.")
        else:
            print("[rebuild] MISSING: train_cached.parquet does not exist.")
            stale = True

        if VAL_OUT.exists():
            existing = pd.read_parquet(VAL_OUT)
            missing = set(existing["instance_id"]) - set(
                filter_cached(val_df, cached)["instance_id"]
            )
            if missing:
                print(f"[rebuild] STALE: val_small.parquet has {len(missing)} instances with uncached images.")
                stale = True
            else:
                print("[rebuild] val_small.parquet OK.")
        else:
            print("[rebuild] MISSING: val_small.parquet does not exist.")
            stale = True

        if stale:
            print("[rebuild] Run without --verify-only to rebuild.")
            sys.exit(1)
        return

    train_cached.to_parquet(TRAIN_OUT, index=False)
    print(f"[rebuild] Saved {TRAIN_OUT} ({len(train_cached)} instances)")

    val_cached.to_parquet(VAL_OUT, index=False)
    print(f"[rebuild] Saved {VAL_OUT} ({len(val_cached)} instances)")
    print("[rebuild] Done.")


if __name__ == "__main__":
    main()
