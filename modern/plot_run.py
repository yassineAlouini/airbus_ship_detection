"""Static snapshot of a Trackio run (for sharing where the local dashboard is not reachable).

usage: python plot_run.py RUN_NAME OUT_PNG [--project airbus-ship-detection]
"""

import argparse
import json
import sqlite3
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import trackio  # noqa: E402

SURFACE, INK, INK_2, GRID, SERIES = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6"
PANELS = [("train/loss", "Training loss", None), ("train/lr", "Learning rate", "zero"),
          ("system/gpu_util", "GPU utilisation (%)", (0, 105))]


def load(project, run_name):
    con = sqlite3.connect(Path(trackio.TRACKIO_DIR) / f"{project}.db")
    rows = con.execute("select step, metrics from metrics where run_name = ? order by step, id", (run_name,))
    series = {}
    for step, metrics in rows:
        for k, v in json.loads(metrics).items():
            if isinstance(v, (int, float)):
                series.setdefault(k, []).append((step, v))
    return series


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_name")
    ap.add_argument("out_png")
    ap.add_argument("--project", default="airbus-ship-detection")
    args = ap.parse_args()
    series = load(args.project, args.run_name)

    fig, axes = plt.subplots(len(PANELS), 1, figsize=(8, 7.5), sharex=True, facecolor=SURFACE)
    for ax, (key, title, ylim) in zip(axes, PANELS):
        ax.set_facecolor(SURFACE)
        pts = series.get(key, [])
        if pts:
            xs, ys = zip(*pts)
            ax.plot(xs, ys, color=SERIES, linewidth=2, solid_capstyle="round")
            ax.annotate(f"{ys[-1]:.3g}", (xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points",
                        va="center", color=INK, fontsize=9)
        if ylim == "zero" and pts:
            ax.set_ylim(0, max(ys) * 1.1)
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            ax.yaxis.get_offset_text().set_color(INK_2)
        elif ylim:
            ax.set_ylim(*ylim)
        ax.set_title(title, loc="left", color=INK, fontsize=11)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.tick_params(colors=INK_2, labelsize=9, length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
    axes[-1].set_xlabel("Training step", color=INK_2, fontsize=9)
    fig.suptitle(f"{args.project} / {args.run_name}", x=0.01, ha="left", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(args.out_png, dpi=130, facecolor=SURFACE)
    print(f"wrote {args.out_png}")


if __name__ == "__main__":
    main()
