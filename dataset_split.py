"""
dataset_split.py

BATCH 4 — Leakage-safe train/validation/test splitting.

Purpose
-------
Random row-level splitting is invalid for this dataset: neighboring patches
share spatial context, many share the exact same underlying Sentinel-5P CH4
pixel (Section 4/8 of the project plan), and a single location can appear at
multiple dates. This script builds geographic BLOCKS, merges any blocks that
share a CH4 source pixel into one unit (so a shared CH4 retrieval can never
be split across sets), and assigns whole merged blocks to train/val/test —
never splitting a block internally.

Method
------
1. Snap each patch's (lat, lon) to a grid cell ("block") sized larger than
   the S5P CH4 footprint (default 0.08 deg ~ 8-9 km), so all patches at one
   location naturally fall in the same block regardless of date.
2. Build a union-find over blocks: any two blocks that share a `ch4_pixel_id`
   are merged into one unit, guaranteeing CH4-source sharing never crosses
   a split boundary.
3. Greedily assign merged units to train/val/test to approximate a 70/15/15
   patch-count split, largest units first (a bin-packing heuristic).
4. Verify and report: zero overlap in locations and CH4 pixels across splits.

Input
-----
- data/processed/metadata_final.csv (from preprocessing.py; must contain
  patch_id, latitude, longitude, ch4_pixel_id)

Output
------
- data/splits/train_ids.txt, val_ids.txt, test_ids.txt (one patch_id per line)
- data/processed/metadata_with_splits.csv (metadata_final.csv + a `split` column)
- data/splits/leakage_report.txt (summary of the verification checks)

Dependencies
------------
pandas, numpy, scikit-learn (only for reproducible RNG/shuffling)

Usage
-----
    python dataset_split.py \
        --metadata data/processed/metadata_final.csv \
        --out-dir data/splits \
        --block-size-deg 0.08 \
        --train-frac 0.70 --val-frac 0.15 --test-frac 0.15 \
        --seed 42
"""

import argparse
import os
import numpy as np
import pandas as pd


class UnionFind:
    def __init__(self, items):
        self.parent = {item: item for item in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def assign_blocks(meta: pd.DataFrame, block_size_deg: float) -> pd.DataFrame:
    """Snap each row to a grid block id based on lat/lon."""
    meta = meta.copy()
    meta["block_lat"] = (meta["latitude"] // block_size_deg).astype(int)
    meta["block_lon"] = (meta["longitude"] // block_size_deg).astype(int)
    meta["block_id"] = meta["block_lat"].astype(str) + "_" + meta["block_lon"].astype(str)
    return meta


def merge_blocks_sharing_ch4(meta: pd.DataFrame) -> pd.DataFrame:
    """
    Merge any grid blocks that share a ch4_pixel_id into one unit via
    union-find, so a shared CH4 retrieval never crosses a split boundary.
    """
    blocks = meta["block_id"].unique().tolist()
    uf = UnionFind(blocks)

    for ch4_id, group in meta.groupby("ch4_pixel_id"):
        block_ids = group["block_id"].unique()
        for b in block_ids[1:]:
            uf.union(block_ids[0], b)

    meta = meta.copy()
    meta["unit_id"] = meta["block_id"].apply(uf.find)
    return meta


def greedy_assign_units(meta: pd.DataFrame, train_frac: float, val_frac: float,
                         test_frac: float, seed: int) -> dict:
    """
    Greedily assign merged units to train/val/test to approximate the target
    fractions by patch count, largest units first, with a fixed seed for
    reproducible tie-breaking among equal-size units.
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6

    unit_sizes = meta.groupby("unit_id").size().sort_values(ascending=False)
    rng = np.random.RandomState(seed)
    # Shuffle within equal-size groups for reproducible, non-arbitrary tie-breaking
    unit_sizes = unit_sizes.sample(frac=1.0, random_state=seed).sort_values(ascending=False, kind="stable")

    total = unit_sizes.sum()
    targets = {"train": train_frac * total, "val": val_frac * total, "test": test_frac * total}
    current = {"train": 0, "val": 0, "test": 0}
    assignment = {}

    for unit_id, size in unit_sizes.items():
        # Assign to whichever split is furthest below its target (proportionally)
        deficits = {k: targets[k] - current[k] for k in current}
        split = max(deficits, key=deficits.get)
        assignment[unit_id] = split
        current[split] += size

    return assignment


def verify_no_leakage(meta: pd.DataFrame) -> str:
    """Check zero overlap in (lat, lon) locations and ch4_pixel_id across splits."""
    lines = []
    loc_by_split = meta.groupby("split").apply(
        lambda g: set(zip(g["latitude"], g["longitude"]))
    )
    ch4_by_split = meta.groupby("split")["ch4_pixel_id"].apply(set)

    splits = list(loc_by_split.index)
    location_overlap_found = False
    ch4_overlap_found = False
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            s1, s2 = splits[i], splits[j]
            loc_overlap = loc_by_split[s1] & loc_by_split[s2]
            ch4_overlap = ch4_by_split[s1] & ch4_by_split[s2]
            if loc_overlap:
                location_overlap_found = True
                lines.append(f"LEAKAGE: {len(loc_overlap)} shared locations between {s1} and {s2}")
            if ch4_overlap:
                ch4_overlap_found = True
                lines.append(f"LEAKAGE: {len(ch4_overlap)} shared CH4 pixels between {s1} and {s2}")

    if not location_overlap_found and not ch4_overlap_found:
        lines.append("PASS: no shared locations or CH4 source pixels across train/val/test.")

    for split in splits:
        sub = meta[meta["split"] == split]
        lines.append(
            f"{split}: {len(sub)} patches | "
            f"{sub[['latitude','longitude']].drop_duplicates().shape[0]} unique locations | "
            f"{sub['ch4_pixel_id'].nunique()} unique CH4 pixels | "
            f"label balance: {sub['label_enhancement'].value_counts(normalize=True).round(3).to_dict()}"
        )

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Leakage-safe spatial block train/val/test split.")
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--out-dir", default="data/splits")
    parser.add_argument("--block-size-deg", type=float, default=0.08,
                         help="Grid block size in degrees (~8-9 km at this latitude, "
                              "chosen to exceed the S5P CH4 footprint)")
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    meta = pd.read_csv(args.metadata)
    meta = assign_blocks(meta, args.block_size_deg)
    meta = merge_blocks_sharing_ch4(meta)

    assignment = greedy_assign_units(meta, args.train_frac, args.val_frac, args.test_frac, args.seed)
    meta["split"] = meta["unit_id"].map(assignment)

    for split in ["train", "val", "test"]:
        ids = meta.loc[meta["split"] == split, "patch_id"]
        with open(os.path.join(args.out_dir, f"{split}_ids.txt"), "w") as f:
            f.write("\n".join(ids.astype(str)))

    out_meta_path = args.metadata.replace("metadata_final.csv", "metadata_with_splits.csv")
    meta.to_csv(out_meta_path, index=False)

    report = verify_no_leakage(meta)
    report_path = os.path.join(args.out_dir, "leakage_report.txt")
    with open(report_path, "w") as f:
        f.write(report)

    print(report)
    print(f"\nSplit files written to {args.out_dir}")
    print(f"Metadata with splits written to {out_meta_path}")
    print(f"Leakage report written to {report_path}")


if __name__ == "__main__":
    main()
