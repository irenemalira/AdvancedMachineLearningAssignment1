#!/usr/bin/env python3
"""
dataset_tools.py - Random sampling and train/test splitting for ML datasets.

You choose the dataset and the percentages. Without --seed, every run is different.

MODE 1: sample  - randomly keep X% of the instances
    python dataset_tools.py sample iris 20
    python dataset_tools.py sample wine 10 50 90
    python dataset_tools.py sample mydata.csv 20

MODE 2: split   - hold out a test set, then build training subsets from the rest
    python dataset_tools.py split iris                          # 70/30, subsets 10%..100%
    python dataset_tools.py split iris --test 20                # 80/20 instead
    python dataset_tools.py split wine --train 25 50 75 100     # custom subset sizes
    python dataset_tools.py split mydata.csv --target label     # CSV needs --target

Dataset: iris | wine | breast_cancer | path to a CSV file
"""

import argparse
import os
import pandas as pd
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split

BUILTIN = {"iris": load_iris, "wine": load_wine, "breast_cancer": load_breast_cancer}


def load_dataset(name):
    """Return (DataFrame, target column name or None)."""
    if name in BUILTIN:
        return BUILTIN[name](as_frame=True).frame, "target"
    return pd.read_csv(name), None


def random_sample(df, percent, seed=None):
    """Randomly select `percent`% of the rows (no repeats)."""
    k = round(len(df) * percent / 100)
    return df.sample(n=k, replace=False, random_state=seed)


def split_train_test(df, target, test_pct=30, seed=None):
    """Split into (train, test). Stratified by class so both keep the class balance."""
    return train_test_split(df, test_size=test_pct / 100,
                            stratify=df[target], random_state=seed)


def train_subsets(train, target, percents, seed=None):
    """Return {percent: random subset of the training set}."""
    out = {}
    for p in percents:
        if p >= 100:
            out[p] = train.copy()
            continue
        try:  # keep class balance if possible
            sub, _ = train_test_split(train, train_size=p / 100,
                                      stratify=train[target], random_state=seed)
        except ValueError:  # subset too small to hold every class
            sub = random_sample(train, p, seed)
        out[p] = sub
    return out


def check_percents(percents):
    bad = [p for p in percents if not 0 < p <= 100]
    if bad:
        raise SystemExit(f"Error: percentages must be between 0 and 100 (got {bad}).")


def main():
    ap = argparse.ArgumentParser(description="Random sampling and splitting for ML datasets.")
    sub = ap.add_subparsers(dest="mode", required=True)

    s = sub.add_parser("sample", help="randomly keep X%% of the instances")
    s.add_argument("dataset")
    s.add_argument("percents", type=float, nargs="*", help="e.g. 20 50 (asked if omitted)")

    t = sub.add_parser("split", help="test split + training subsets")
    t.add_argument("dataset")
    t.add_argument("--target", help="target/label column (required for CSV files)")
    t.add_argument("--test", type=float, default=30, help="test set %% (default 30)")
    t.add_argument("--train", type=float, nargs="+", default=list(range(10, 101, 10)),
                   help="training subset %% of the train set (default 10 20 ... 100)")

    for p in (s, t):
        p.add_argument("--out", default="output", help="output folder (default: output)")
        p.add_argument("--seed", type=int, default=None,
                       help="optional seed for reproducible results")

    a = ap.parse_args()
    df, target = load_dataset(a.dataset)
    os.makedirs(a.out, exist_ok=True)
    print(f"Dataset: {a.dataset} | N = {len(df)} instances")

    if a.mode == "sample":
        percents = a.percents or [float(x) for x in input("Enter percentage(s), e.g. 20 50: ").split()]
        check_percents(percents)
        for p in percents:
            part = random_sample(df, p, a.seed)
            path = f"{a.out}/sample_{p:g}pct.csv"
            part.to_csv(path, index=False)
            print(f"  {p:g}% -> {len(part)} instances -> {path}")

    else:
        target = a.target or target
        if target is None or target not in df.columns:
            raise SystemExit("Error: give the label column with --target (e.g. --target species).")
        check_percents(a.train + [a.test])
        train, test = split_train_test(df, target, a.test, a.seed)
        test.to_csv(f"{a.out}/test.csv", index=False)
        print(f"  Train ({100 - a.test:g}%): {len(train)} | Test ({a.test:g}%): {len(test)} -> {a.out}/test.csv")
        for p, part in train_subsets(train, target, a.train, a.seed).items():
            path = f"{a.out}/train_{p:g}pct.csv"
            part.to_csv(path, index=False)
            print(f"  train_{p:g}pct: {len(part)} rows -> {path}")


if __name__ == "__main__":
    main()
