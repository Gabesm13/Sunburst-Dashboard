#!/usr/bin/env python3
"""
data_gen.py — Generate synthetic data for a 6-level sunburst + path-bar CSVs.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# HELPER FUNCTIONS
# ----------------------------------------------------------------------

def build_raw_tables(seed: int):
    """
    Generate raw synthetic level-1 through level-6 tables.
    """
    np.random.seed(seed)

    lvl1 = pd.DataFrame(
        {"level1": ["Page 1", "Page 2", "Page 3"],
         "value":  [90, 38, 36]}
    )

    lvl2_info = {
        "Page 1": {"Page 2": 12, "Page 3": 34, "Page 4": 18,
                   "Page 5": 13, "Page 6": 4,  "Page 7": 9},
        "Page 2": {"Page 3": 20, "Page 4": 18},
        "Page 3": {"Page 2": 9,  "Page 3": 24},
    }
    lvl2 = pd.DataFrame([
        {"level1": p1, "level2": p2, "value": v}
        for p1, kids in lvl2_info.items() for p2, v in kids.items()
    ])

    lvl3_info = {
        ("Page 1", "Page 2"): {"Page 3": 3,  "Page 5": 9},
        ("Page 1", "Page 3"): {"Page 2": 4,  "Page 4": 5,  "Page 6": 9},
        ("Page 1", "Page 4"): {"Page 3": 3,  "Page 4": 6,  "Page 5": 4, "Page 8": 5},
        ("Page 1", "Page 5"): {"Page 3": 3,  "Page 5": 10},
        ("Page 1", "Page 6"): {"Page 8": 4},
        ("Page 1", "Page 7"): {"Page 2": 9},
        ("Page 2", "Page 3"): {"Page 1": 6,  "Page 4": 5,  "Page 6": 9},
        ("Page 2", "Page 4"): {"Page 3": 3,  "Page 4": 6,  "Page 5": 4, "Page 6": 5},
        ("Page 3", "Page 2"): {"Page 5": 9},
        ("Page 3", "Page 3"): {"Page 3": 3,  "Page 4": 5,  "Page 8": 9},
    }
    lvl3 = pd.DataFrame([
        {"level1": p1, "level2": p2, "level3": p3, "value": v}
        for (p1, p2), kids in lvl3_info.items() for p3, v in kids.items()
    ])

    lvl4_info = {
        ("Page 1","Page 2","Page 5"): {"Page 3":9},
        ("Page 1","Page 3","Page 2"): {"Page 3":4},
        ("Page 1","Page 3","Page 4"): {"Page 2":5},
        ("Page 1","Page 3","Page 6"): {"Page 3":9},
        ("Page 1","Page 4","Page 4"): {"Page 6":6},
        ("Page 1","Page 4","Page 5"): {"Page 3":4},
        ("Page 1","Page 4","Page 8"): {"Page 2":5},
        ("Page 1","Page 5","Page 5"): {"Page 3":10},
        ("Page 1","Page 6","Page 8"): {"Page 3":4},
        ("Page 1","Page 7","Page 2"): {"Page 3":9},
        ("Page 2","Page 3","Page 1"): {"Page 1":6},
        ("Page 2","Page 3","Page 4"): {"Page 2":5},
        ("Page 2","Page 3","Page 6"): {"Page 3":9},
        ("Page 2","Page 4","Page 4"): {"Page 6":6},
        ("Page 2","Page 4","Page 5"): {"Page 3":4},
        ("Page 2","Page 4","Page 6"): {"Page 2":5},
        ("Page 3","Page 2","Page 5"): {"Page 3":9},
        ("Page 3","Page 3","Page 4"): {"Page 2":5},
        ("Page 3","Page 3","Page 8"): {"Page 3":9},
    }
    lvl4 = pd.DataFrame([
        {"level1": p1, "level2": p2, "level3": p3, "level4": p4, "value": v}
        for (p1, p2, p3), kids in lvl4_info.items() for p4, v in kids.items()
    ])

    lvl5_info = {
        ("Page 1","Page 2","Page 5","Page 3"): {"Page 3":5},
        ("Page 1","Page 3","Page 4","Page 2"): {"Page 3":5},
        ("Page 1","Page 3","Page 6","Page 3"): {"Page 3":5},
        ("Page 1","Page 4","Page 4","Page 6"): {"Page 2":6},
        ("Page 1","Page 4","Page 8","Page 2"): {"Page 3":5},
        ("Page 1","Page 5","Page 5","Page 3"): {"Page 1":6},
        ("Page 1","Page 7","Page 2","Page 3"): {"Page 3":5},
        ("Page 2","Page 3","Page 1","Page 1"): {"Page 1":6},
        ("Page 2","Page 3","Page 4","Page 2"): {"Page 3":5},
        ("Page 2","Page 3","Page 6","Page 3"): {"Page 3":5},
        ("Page 2","Page 4","Page 4","Page 6"): {"Page 2":6},
        ("Page 2","Page 4","Page 6","Page 2"): {"Page 3":5},
        ("Page 3","Page 2","Page 5","Page 3"): {"Page 3":5},
        ("Page 3","Page 3","Page 4","Page 2"): {"Page 3":5},
        ("Page 3","Page 3","Page 8","Page 3"): {"Page 3":5},
    }
    lvl5 = pd.DataFrame([
        {"level1": p1, "level2": p2, "level3": p3, "level4": p4, "level5": p5, "value": v}
        for (p1, p2, p3, p4), kids in lvl5_info.items() for p5, v in kids.items()
    ])

    lvl6_info = {
        ("Page 1","Page 4","Page 4","Page 6","Page 2"): {"Page 3":6},
        ("Page 1","Page 5","Page 5","Page 3","Page 1"): {"Page 3":6},
        ("Page 2","Page 3","Page 1","Page 1","Page 1"): {"Page 3":6},
        ("Page 2","Page 4","Page 4","Page 6","Page 2"): {"Page 3":6},
    }
    lvl6 = pd.DataFrame([
        {"level1": p1, "level2": p2, "level3": p3, "level4": p4, "level5": p5, "level6": p6, "value": v}
        for (p1, p2, p3, p4, p5), kids in lvl6_info.items() for p6, v in kids.items()
    ])

    return lvl1, lvl2, lvl3, lvl4, lvl5, lvl6


def build_sunburst_df(level_tables):
    """
    Build a DataFrame for sunburst plotting with unique 'id' and 'parent_id'.
    """
    lvl1, lvl2, lvl3, lvl4, lvl5, lvl6 = level_tables

    def build_level(df, depth, child_col, parent_col=None):
        tbl = df.copy()
        tbl["label"]  = tbl[child_col]
        tbl["parent"] = "" if parent_col is None else tbl[parent_col]
        tbl["depth"]  = depth
        return tbl

    levels = [
        (lvl1, 1, "level1", None),
        (lvl2, 2, "level2", "level1"),
        (lvl3, 3, "level3", "level2"),
        (lvl4, 4, "level4", "level3"),
        (lvl5, 5, "level5", "level4"),
        (lvl6, 6, "level6", "level5"),
    ]
    dfs = [build_level(df, d, c, p) for df, d, c, p in levels]
    sb = pd.concat(dfs, ignore_index=True)

    def make_id(row):
        parts = [row[f"level{i}"] for i in range(1, row.depth+1)
                 if row.get(f"level{i}")]
        return " - ".join(parts)

    sb["id"] = sb.apply(make_id, axis=1)
    sb["parent_id"] = sb.apply(
        lambda r: "" if r.depth == 1 else " - ".join(r["id"].split(" - ")[:-1]),
        axis=1
    )

    cols = ["id","parent_id","label","parent","value","depth",
            "level1","level2","level3","level4","level5","level6"]
    return sb[cols]


def build_pathbar_data(sb: pd.DataFrame):
    """
    Build a path-bar DataFrame for leaf nodes from the sunburst DataFrame.
    """
    internal = sb.apply(lambda r: (sb.parent_id == r.id).any(), axis=1)
    leaves   = sb.loc[~internal]

    recs = []
    for _, leaf in leaves.iterrows():
        parts = leaf.id.split(" - ")
        parts += [""] * (6 - len(parts))
        rec = {f"level{i+1}": parts[i] for i in range(6)}
        rec["count"] = leaf.value
        recs.append(rec)

    return pd.DataFrame(recs)

# ----------------------------------------------------------------------
# MAIN ENTRY POINT
# ----------------------------------------------------------------------

def main(seed: int):
    """
    Generate and write sunburst and path-bar CSV files.
    """
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"
    data_dir.mkdir(exist_ok=True)

    lvl_tables = build_raw_tables(seed)
    sb_df       = build_sunburst_df(lvl_tables)

    bar_df = build_pathbar_data(sb_df)

    sb_df.to_csv(data_dir / "sunburst_full_hierarchy.csv", index=False)
    bar_df.to_csv(data_dir / "path_bar_data.csv",            index=False)
    print("Wrote sunburst_full_hierarchy.csv and path_bar_data.csv")


def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Generate synthetic data for a 6-level sunburst + path-bar CSVs."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.seed)
