"""Stream a running (or finished) training log into Trackio, without touching the training process.

Follows the timestamped log written by ``small_ships.py`` / ``diagnose.py`` and the 1 s utilisation traces written by
``trace_run.sh``, and logs them to a Trackio run through ``kaggle_tools.tracking.trackio_run``:

* ``train/loss``, ``train/lr``, ``train/progress`` at every logged step, plus ``epoch/mean_loss`` per epoch;
* ``system/gpu_util``, ``system/gpu_mem_gb``, ``system/cpu_busy`` (one point per minute, averaged);
* the validation scores per mode, when the run reports them.

usage: python track_run.py RUN_DIR --name NAME [--project airbus-ship-detection] [--once]
Dashboard: ``trackio show --project airbus-ship-detection``.
"""

import argparse
import json
import re
import time
from pathlib import Path

import pandas as pd
import trackio

from kaggle_tools.tracking import trackio_run

STEP_RE = re.compile(r"epoch (\d+) step (\d+) loss ([\d.]+) lr ([\d.e+-]+) progress ([\d.]+)")
EPOCH_RE = re.compile(r"epoch (\d+) done, mean loss ([\d.]+)")
BEST_BY_MODE_RE = re.compile(r"best by mode: (\{.*\})")
END_RE = re.compile(r"submission: |Traceback")
TIME_RE = re.compile(r"^\[(\d\d:\d\d:\d\d) ")


def line_time(line):
    """Wall-clock time of a timestamped log line (the log only has HH:MM:SS)."""
    m = TIME_RE.match(line)
    if not m:
        return None
    ts = pd.Timestamp(f"{pd.Timestamp.now().date()} {m[1]}")
    return ts - pd.Timedelta(days=1) if ts > pd.Timestamp.now() + pd.Timedelta(minutes=1) else ts


def follow(run_dir, once):
    """Yield new log lines as they are written; stop at the end of the run (or of the file with ``once``)."""
    path = Path(run_dir) / "run.log"
    while not path.exists():
        time.sleep(5)
    with path.open() as f:
        while True:
            line = f.readline()
            if line:
                yield line
                if END_RE.search(line):
                    return
            elif once:
                return
            else:
                time.sleep(10)


def system_points(run_dir, since, until):
    """Per-minute averages of the utilisation traces in (``since``, ``until``]."""
    gpu_path, cpu_path = Path(run_dir) / "gpu_trace.csv", Path(run_dir) / "cpu_trace.csv"
    if not gpu_path.exists() or not cpu_path.exists():
        return []
    gpu = pd.read_csv(gpu_path, header=None, names=["ts", "gpu", "mem"], skipinitialspace=True)
    cpu = pd.read_csv(cpu_path)
    gpu["ts"] = pd.to_datetime(gpu.ts.str.strip(), errors="coerce")
    cpu["ts"] = pd.to_datetime(cpu.timestamp, errors="coerce")
    g = gpu.dropna().set_index("ts").resample("1min").mean(numeric_only=True)
    c = cpu.dropna(subset=["ts"]).set_index("ts")["cpu_busy_pct"].resample("1min").mean()
    df = g.join(c, how="inner")
    df = df[(df.index > since) & (df.index <= until.floor("1min"))]
    return [(ts, {"system/gpu_util": r.gpu, "system/gpu_mem_gb": r.mem / 1024, "system/cpu_busy": r.cpu_busy_pct})
            for ts, r in df.iterrows()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--name", required=True)
    ap.add_argument("--project", default="airbus-ship-detection")
    ap.add_argument("--once", action="store_true", help="import a finished run and exit")
    args = ap.parse_args()

    config_line = None
    run = None
    last_sys = pd.Timestamp(0)
    step = 0
    for line in follow(args.run_dir, args.once):
        now = line_time(line)
        if run is None:
            if "] config " in line:
                config_line = json.loads(line.split("] config ", 1)[1])
            run = trackio_run(args.project, config=config_line or {}, name=args.name, resume="allow",
                              auto_log_gpu=False, auto_log_cpu=False)
        if (m := STEP_RE.search(line)) and now is not None:
            # Minutes before this line belong to the previous step.
            for ts, metrics in system_points(args.run_dir, last_sys, now - pd.Timedelta(minutes=1)):
                trackio.log(metrics, step=step)
                last_sys = ts
        if m := STEP_RE.search(line):
            step = int(m[2])
            trackio.log({"train/loss": float(m[3]), "train/lr": float(m[4]), "train/progress": float(m[5]),
                         "train/epoch": int(m[1])}, step=step)
        elif m := EPOCH_RE.search(line):
            trackio.log({"epoch/mean_loss": float(m[2]), "train/epoch": int(m[1])}, step=step)
        elif m := BEST_BY_MODE_RE.search(line):
            trackio.log({f"valid/f2_{k}": v for k, v in json.loads(m[1]).items()}, step=step)
        # System minutes up to this line's time get the step reached by then, so they line up with the loss curve.
        if now is not None:
            for ts, metrics in system_points(args.run_dir, last_sys, now):
                trackio.log(metrics, step=step)
                last_sys = ts
    if run is not None:
        trackio.finish()


if __name__ == "__main__":
    main()
