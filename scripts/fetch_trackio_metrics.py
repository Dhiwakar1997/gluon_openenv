"""Fetch per-step metrics for one or more Trackio runs from a Hugging Face Space,
save them as CSV/JSON, and plot reward + loss curves across the requested runs.

Usage:
    python scripts/fetch_trackio_metrics.py \
        --space DhiwakarDev/mcm-trackio \
        --project mcm-gemma3-27b-full \
        --runs ticket_booking-20260426-073634 ticket_issuance-20260426-080000 \
        --out-dir trackio_logs
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from gradio_client import Client


def fetch_run(client: Client, project: str, run: str) -> pd.DataFrame:
    rows = client.predict(project, run, api_name="/get_logs")
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "step" in df.columns:
        df = df.sort_values("step").reset_index(drop=True)
    return df


def plot_metric(frames: dict[str, pd.DataFrame], metric: str, out_path: Path, title: str) -> bool:
    fig, ax = plt.subplots(figsize=(9, 5))
    plotted = 0
    for run, df in frames.items():
        if metric not in df.columns or "step" not in df.columns:
            continue
        series = df[["step", metric]].dropna()
        if series.empty:
            continue
        ax.plot(series["step"], series[metric], marker="o", linewidth=1.5, label=run)
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        print(f"  [plot] no runs have '{metric}' — skipping {out_path.name}")
        return False

    ax.set_xlabel("step")
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  [plot] {metric} -> {out_path}")
    return True


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--space", default="DhiwakarDev/mcm-trackio",
                   help="HF Space hosting the Trackio dashboard (user/space).")
    p.add_argument("--project", default="mcm-gemma3-27b-full",
                   help="Trackio project name.")
    p.add_argument("--runs", nargs="+", required=True,
                   help="One or more run names to fetch.")
    p.add_argument("--out-dir", default="trackio_logs",
                   help="Directory to write per-run CSV + JSON and plots.")
    p.add_argument("--reward-metric", default="reward/mean",
                   help="Column name to use for the reward plot.")
    p.add_argument("--loss-metric", default="train/loss",
                   help="Column name to use for the loss plot.")
    p.add_argument("--no-plots", action="store_true",
                   help="Skip generating reward/loss PNG plots.")
    p.add_argument("--hf-token", default=os.getenv("HF_TOKEN"),
                   help="HF token (only required if the Space is private).")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    client = Client(args.space, **({"token": args.hf_token} if args.hf_token else {}))
    print(f"[trackio] connected to {args.space} (project={args.project})")

    summary = []
    frames: dict[str, pd.DataFrame] = {}
    for run in args.runs:
        try:
            df = fetch_run(client, args.project, run)
        except Exception as e:
            print(f"  [{run}] ERROR: {e}")
            summary.append({"run": run, "rows": 0, "error": str(e)})
            continue

        if df.empty:
            print(f"  [{run}] no rows returned")
            summary.append({"run": run, "rows": 0})
            continue

        csv_path = out_dir / f"{run}.csv"
        json_path = out_dir / f"{run}.json"
        df.to_csv(csv_path, index=False)
        df.to_json(json_path, orient="records", indent=2)
        frames[run] = df

        metric_cols = [c for c in df.columns if c not in ("step", "timestamp", "run", "project")]
        print(f"  [{run}] {len(df)} rows -> {csv_path} (metrics: {metric_cols})")
        summary.append({"run": run, "rows": len(df), "metrics": metric_cols})

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[trackio] summary -> {out_dir / 'summary.json'}")

    if not args.no_plots and frames:
        plot_metric(frames, args.reward_metric, out_dir / "reward.png",
                    title=f"{args.reward_metric} across runs")
        plot_metric(frames, args.loss_metric, out_dir / "loss.png",
                    title=f"{args.loss_metric} across runs")


if __name__ == "__main__":
    main()
