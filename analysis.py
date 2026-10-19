from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "pick_records.csv"
OUT = ROOT / "outputs"
FAILURE_COST_MIN = 120
CHECK_COST_MIN = 12


def policy_metrics(frame: pd.DataFrame, age_threshold: int, qty_threshold: int) -> dict:
    flagged = (frame["stock_age_min"] >= age_threshold) | (
        frame["book_stock_qty"] <= qty_threshold
    )
    failures = frame["failure"].astype(bool)
    caught = int((flagged & failures).sum())
    missed = int((~flagged & failures).sum())
    baseline = int(failures.sum()) * FAILURE_COST_MIN
    policy_cost = int(flagged.sum()) * CHECK_COST_MIN + missed * FAILURE_COST_MIN
    return {
        "records": len(frame),
        "failures": int(failures.sum()),
        "flagged": int(flagged.sum()),
        "caught_failures": caught,
        "missed_failures": missed,
        "precision": caught / int(flagged.sum()) if flagged.any() else 0,
        "recall": caught / int(failures.sum()) if failures.any() else 0,
        "baseline_cost_min": baseline,
        "policy_cost_min": policy_cost,
        "saved_min": baseline - policy_cost,
        "saving_rate": (baseline - policy_cost) / baseline if baseline else 0,
    }


def main() -> None:
    OUT.mkdir(exist_ok=True)
    raw = DATA.read_bytes()
    source_rows = len(pd.read_csv(DATA))

    df = pd.read_csv(DATA).drop_duplicates().copy()
    df["pick_time"] = pd.to_datetime(df["pick_time"], utc=True)
    df["stock_synced_at"] = pd.to_datetime(df["stock_synced_at"], utc=True)
    df["stock_age_min"] = (
        (df["pick_time"] - df["stock_synced_at"]).dt.total_seconds() / 60
    )

    labeled = df.dropna(subset=["is_success"]).sort_values("pick_time").copy()
    labeled["failure"] = (labeled["is_success"] == 0).astype(int)
    split = int(len(labeled) * 0.70)
    train, test = labeled.iloc[:split], labeled.iloc[split:]

    # Search only a small, interpretable rule family on the earlier time period.
    candidates = []
    for age in range(1, 66):
        for qty in range(1, 31):
            result = policy_metrics(train, age, qty)
            candidates.append((result["saved_min"], -result["flagged"], age, qty))
    _, _, best_age, best_qty = max(candidates)

    train_metrics = policy_metrics(train, best_age, best_qty)
    test_metrics = policy_metrics(test, best_age, best_qty)
    all_metrics = policy_metrics(labeled, best_age, best_qty)

    # Bootstrap the fixed rule on the holdout; this reflects sampling uncertainty,
    # not uncertainty in the unobserved causal effect of an inventory check.
    rng = np.random.default_rng(42)
    savings = []
    for _ in range(5000):
        sample = test.iloc[rng.integers(0, len(test), len(test))]
        savings.append(policy_metrics(sample, best_age, best_qty)["saved_min"])
    ci_low, ci_high = np.percentile(savings, [2.5, 97.5])

    age_groups = pd.cut(
        labeled["stock_age_min"],
        bins=[-1, 5, 10, 15, 18, 30, 45, np.inf],
        labels=["0-5", "6-10", "11-15", "16-18", "19-30", "31-45", "46+"],
    )
    age_summary = (
        labeled.assign(age_bucket=age_groups)
        .groupby("age_bucket", observed=True)["failure"]
        .agg(records="size", failures="sum", failure_rate="mean")
        .reset_index()
    )
    age_summary.to_csv(OUT / "failure_rate_by_stock_age.csv", index=False)

    summary = {
        "data_quality": {
            "sha256": hashlib.sha256(raw).hexdigest(),
            "source_rows": source_rows,
            "exact_duplicate_rows_removed": source_rows - len(df),
            "unique_rows": len(df),
            "labeled_rows": len(labeled),
            "interrupted_rows_excluded": int(df["is_success"].isna().sum()),
            "failures": int(labeled["failure"].sum()),
            "overall_failure_rate": labeled["failure"].mean(),
            "period_utc": [str(df["pick_time"].min()), str(df["pick_time"].max())],
        },
        "rule": {
            "stock_age_at_least_min": best_age,
            "book_stock_qty_at_most": best_qty,
            "logic": "OR",
        },
        "train": train_metrics,
        "holdout": test_metrics,
        "holdout_saved_min_bootstrap_95pct": [float(ci_low), float(ci_high)],
        "all_labeled_descriptive_only": all_metrics,
    }
    (OUT / "metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
