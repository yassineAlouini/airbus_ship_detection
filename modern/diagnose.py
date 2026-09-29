"""Stage 3: diagnose the remaining error and try post-processing that needs no retraining.

Stage 2 showed that gating is nearly solved (perfect-gate ceiling 0.887 vs 0.882 reached). The remaining error is on
images that *do* contain ships (0.48 F2). This script reuses the stage-1 U-Net and the stage-2 ViT gate as they are,
and on the shared 4,000-image validation split it:

1. Breaks the error down: F2 per IoU threshold (bad outlines vs missed ships), recall by ship size, merged / split
   ships, and spurious detections.
2. Tries 8-way dihedral TTA against the current 4 flips, both from one set of 8 forward passes.
3. Tries snapping instances to their minimum-area rotated rectangle.

The best configuration found by the exact competition metric is then used to write ``submission.csv``.
"""

import json
import multiprocessing as mp
import os
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from airbus_modern import (CONFIG, TestDataset, _env, gate_sources, image_f2, labels_from_rles, load_masks,
                           predict_dihedral, remove_small, rles_from_labels, segment_instances, snap_to_rectangles,
                           split_ids)
from vit_gate import VIT_CONFIG, ViTGate, find_kernel_output, load_segmenter, predict_gate

DIAG_CONFIG = {
    "vit_gate_path": _env("ASD_VIT_GATE_PATH", ""),
    "infer_batch": _env("ASD_INFER_BATCH", 16, int),
    # CPU processes for post-processing; 0 = all cores minus the data-loader workers and 2 spare.
    "postproc_workers": _env("ASD_POSTPROC_WORKERS", 0, int),
}

TTA_MODES = ["flip4", "dihedral8"]
GATE_GRID = [0.9, 0.93, 0.95, 0.97, 0.98, 0.99]
MASK_GRID = [0.4, 0.5, 0.6]
AREA_GRID = [0, 20, 40]
SNAP_GRID = [None, 0, 50, 150]  # None = no snapping; otherwise snap instances with at least this many pixels.
# Stage-2 submission, to check that this run reproduces its validation score.
BASELINE = {"tta": "flip4", "gate": "mean_aux_vit", "gate_thr": 0.97, "mask_thr": 0.5, "min_area": 20, "snap": None}
SIZE_BINS = [0, 50, 150, 500, 2000, np.inf]
IOU_THRESHOLDS = np.arange(0.5, 1.0, 0.05)


def segmenter_probs(model, image_dir, image_ids, cfg, device, modes=TTA_MODES):
    """Yield (image_id, {tta_mode: (body, border, aux_prob)}) from one set of 8 forward passes per image."""
    model.eval()
    loader = DataLoader(TestDataset(image_dir, image_ids), batch_size=cfg["infer_batch"], num_workers=cfg["workers"],
                        pin_memory=device.type == "cuda", persistent_workers=cfg["workers"] > 0,
                        prefetch_factor=4 if cfg["workers"] > 0 else None)
    k = 0
    for x in loader:
        x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            seg4, cls4, seg8, cls8 = predict_dihedral(model, x)
        out = {"flip4": (seg4.half().cpu().numpy(), cls4.cpu().numpy()),
               "dihedral8": (seg8.half().cpu().numpy(), cls8.cpu().numpy())}
        for i in range(len(x)):
            yield image_ids[k], {m: (out[m][0][i, 0], out[m][0][i, 1], float(out[m][1][i])) for m in modes}
            k += 1


_T0 = time.time()


def log(msg):
    """Timestamped progress line, so stage durations can be read from the log."""
    print(f"[{time.strftime('%H:%M:%S')} +{(time.time() - _T0) / 60:6.1f} min] {msg}", flush=True)


def _init_worker():
    # One thread per process: the parallelism comes from the process pool.
    cv2.setNumThreads(1)
    torch.set_num_threads(1)


def make_pool(cfg):
    n = cfg["postproc_workers"] or max(1, (os.cpu_count() or 4) - cfg["workers"] - 2)
    # spawn: the parent holds a CUDA context, which must not be forked.
    return ProcessPoolExecutor(n, mp_context=mp.get_context("spawn"), initializer=_init_worker), n


def parallel_stream(pool, n_workers, fn, items):
    """Apply ``fn(*item)`` in the pool while ``items`` keeps being produced (by the GPU); results in input order.

    At most a few batches per worker are in flight, which bounds memory while keeping every core busy.
    """
    pending = deque()
    for item in items:
        pending.append(pool.submit(fn, *item))
        if len(pending) >= 4 * n_workers:
            yield pending.popleft().result()
    while pending:
        yield pending.popleft().result()


def postprocess(body, border, mask_thr, min_area, snap):
    labels = remove_small(segment_instances(body, border, mask_thr), min_area)
    return labels if snap is None else snap_to_rectangles(labels, snap)


def overlaps(truth, pred):
    """Intersection matrix [n_true, n_pred] plus the per-instance areas."""
    n_true, n_pred = int(truth.max()), int(pred.max())
    both = np.bincount(truth.ravel() * (n_pred + 1) + pred.ravel(),
                       minlength=(n_true + 1) * (n_pred + 1)).reshape(n_true + 1, n_pred + 1)
    return both[1:, 1:], both[1:, :].sum(1), both[:, 1:].sum(0)


def error_breakdown(truth, pred):
    """Per-image error components for images that contain ships."""
    inter, area_true, area_pred = overlaps(truth, pred)
    n_true, n_pred = len(area_true), len(area_pred)
    if n_pred == 0:
        iou = np.zeros((n_true, 0))
    else:
        iou = inter / np.maximum(area_true[:, None] + area_pred[None, :] - inter, 1)
    best_iou = iou.max(1) if n_pred else np.zeros(n_true)
    tp = np.array([(iou > t).any(1).sum() if n_pred else 0 for t in IOU_THRESHOLDS])
    fp = np.array([n_pred - (iou > t).any(0).sum() if n_pred else 0 for t in IOU_THRESHOLDS])
    fn = n_true - tp
    f2 = 5 * tp / np.maximum(5 * tp + 4 * fn + fp, 1)
    cover_true = inter / np.maximum(area_true[:, None], 1)  # share of each true ship covered by each prediction
    merged = split = 0
    if n_pred:
        # Merged: one prediction covers >= 30% of two or more true ships.
        merged = int(sum(((cover_true[:, j] >= 0.3).sum() >= 2) * (cover_true[:, j] >= 0.3).sum()
                         for j in range(n_pred)))
        # Split: a true ship is covered >= 20% by two or more predictions.
        split = int(((cover_true >= 0.2).sum(1) >= 2).sum())
    spurious = int((iou.max(0) < 0.1).sum()) if n_pred else 0
    return {"f2_per_thr": f2, "areas": area_true, "best_iou": best_iou, "merged": merged, "split": split,
            "spurious": spurious, "n_true": n_true, "n_pred": n_pred}


def summarise(breakdowns, gated_out):
    """Aggregate the per-image breakdowns (ship images only) into a small report."""
    areas = np.concatenate([b["areas"] for b in breakdowns])
    best_iou = np.concatenate([b["best_iou"] for b in breakdowns])
    n_true = int(sum(b["n_true"] for b in breakdowns))
    by_size = []
    for lo, hi in zip(SIZE_BINS[:-1], SIZE_BINS[1:]):
        sel = (areas >= lo) & (areas < hi)
        if sel.any():
            found = best_iou[sel] > 0.5
            by_size.append({"area_px": f"{lo}-{hi}", "ships": int(sel.sum()),
                            "recall_iou50": float(found.mean()),
                            "recall_iou80": float((best_iou[sel] > 0.8).mean()),
                            "median_iou_when_found": float(np.median(best_iou[sel][found])) if found.any() else None})
    f2_thr = np.mean([b["f2_per_thr"] for b in breakdowns], 0)
    return {
        "ship_images": len(breakdowns),
        "ship_images_gated_out": int(gated_out),
        "true_ships": n_true,
        "f2_by_iou_threshold": {f"{t:.2f}": float(v) for t, v in zip(IOU_THRESHOLDS, f2_thr)},
        "by_ship_size": by_size,
        "merged_ship_share": float(sum(b["merged"] for b in breakdowns) / max(n_true, 1)),
        "split_ship_share": float(sum(b["split"] for b in breakdowns) / max(n_true, 1)),
        "spurious_detections_per_ship_image": float(np.mean([b["spurious"] for b in breakdowns])),
    }


COMBOS = [(t, m, a, s) for t in TTA_MODES for m in MASK_GRID for a in AREA_GRID for s in SNAP_GRID]


def score_valid_image(image_id, per_tta, rles, vit_prob):
    """Score every (tta, mask, area, snap) combination on one validation image; runs in a worker process.

    Returns (image_id, f2 row, is_empty, {tta: aux prob}, baseline breakdown or None, gated_out flag).
    """
    truth = labels_from_rles(rles)
    is_empty = truth.max() == 0
    row, cache, breakdown = np.zeros(len(COMBOS)), {}, None
    for c, (t, m, a, s) in enumerate(COMBOS):
        body, border, _ = per_tta[t]
        if (t, m) not in cache:
            cache[(t, m)] = segment_instances(body.astype(np.float32), border.astype(np.float32), m)
        labels = remove_small(cache[(t, m)], a)
        if labels.max() == 0:
            row[c] = 1.0 if is_empty else 0.0
            continue
        labels = labels if s is None else snap_to_rectangles(labels, s)
        row[c] = image_f2(truth, labels)
        if (t, m, a, s) == config_key(BASELINE) and not is_empty:
            breakdown = error_breakdown(truth, labels)
    gated_out = False
    if not is_empty:
        sources = gate_sources(per_tta[BASELINE["tta"]][2], {"vit": vit_prob})
        if sources[BASELINE["gate"]] < BASELINE["gate_thr"]:
            gated_out, breakdown = True, None
        elif breakdown is None:
            # Passed the gate but no instance survived: every true ship is a false negative.
            breakdown = error_breakdown(truth, np.zeros_like(truth))
    return image_id, row, is_empty, {t: per_tta[t][2] for t in TTA_MODES}, breakdown, gated_out


def submission_rows(image_id, body, border, aux_prob, vit_prob, best):
    """Post-process one test image with the chosen configuration; runs in a worker process."""
    labels = np.zeros(body.shape, np.int32)
    if gate_sources(aux_prob, {"vit": vit_prob})[best["gate"]] >= best["gate_thr"]:
        labels = postprocess(body.astype(np.float32), border.astype(np.float32), best["mask_thr"],
                             best["min_area"], best["snap"])
    rles = rles_from_labels(labels)
    return [(image_id, r) for r in rles] if rles else [(image_id, "")]


def config_key(c):
    return (c["tta"], c["mask_thr"], c["min_area"], c["snap"])


def main():
    cfg = {**CONFIG, **VIT_CONFIG, **DIAG_CONFIG}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(cfg["out_dir"]).mkdir(parents=True, exist_ok=True)
    print(json.dumps(cfg, indent=1), device, flush=True)

    log("start")
    segmenter = load_segmenter(cfg, device)
    gate = ViTGate(cfg["vit_model"], 0, cfg["vit_img"]).to(device)
    gate.load_state_dict(torch.load(find_kernel_output("vit_gate.pt", cfg["vit_gate_path"]), map_location=device))

    ids, rles_by_id = load_masks(cfg["data_dir"], cfg["limit"])
    _, _, valid = split_ids(ids, rles_by_id, cfg["valid_images"], cfg["seed"])
    test_dir = Path(cfg["data_dir"]) / "test_v2"
    test_ids = sorted(pd.read_csv(Path(cfg["data_dir"]) / "sample_submission_v2.csv").ImageId)
    if cfg["limit"]:
        test_ids = [i for i in test_ids if (test_dir / i).exists()][:cfg["limit"]]
    valid_vit = predict_gate(gate, Path(cfg["data_dir"]) / "train_v2", valid, cfg, device)
    test_vit = predict_gate(gate, test_dir, test_ids, cfg, device)
    del gate
    torch.cuda.empty_cache()
    # Cache the gate so later stages do not need to re-run the ViT.
    pd.DataFrame({"ImageId": list(valid_vit) + list(test_vit),
                  "split": ["valid"] * len(valid_vit) + ["test"] * len(test_vit),
                  "vit_prob": list(valid_vit.values()) + list(test_vit.values())}).to_csv(
        Path(cfg["out_dir"]) / "vit_gate_probs.csv", index=False)
    log("gate probabilities done")

    # --- validation: the GPU streams predictions while a process pool scores every combination per image ---
    pool, n_workers = make_pool(cfg)
    log(f"validation: post-processing with {n_workers} worker processes")
    valid_items = ((image_id, per_tta, rles_by_id.get(image_id, []), valid_vit[image_id])
                   for image_id, per_tta in segmenter_probs(segmenter, Path(cfg["data_dir"]) / "train_v2", valid,
                                                            cfg, device))
    f2, aux, empty, baseline_breakdowns, baseline_gated_out = [], {t: [] for t in TTA_MODES}, [], [], 0
    for _, row, is_empty, aux_by_tta, breakdown, gated_out in parallel_stream(pool, n_workers, score_valid_image,
                                                                              valid_items):
        f2.append(row)
        empty.append(1.0 if is_empty else 0.0)
        for t in TTA_MODES:
            aux[t].append(aux_by_tta[t])
        baseline_gated_out += gated_out
        if breakdown is not None:
            baseline_breakdowns.append(breakdown)
    f2, empty = np.stack(f2), np.array(empty)
    vit = np.array([valid_vit[i] for i in valid])
    combos = COMBOS

    results = []
    for t in TTA_MODES:
        for gate_name, probs in gate_sources(np.array(aux[t]), {"vit": vit}).items():
            for thr in GATE_GRID:
                keep = probs >= thr
                for c, (ct, m, a, s) in enumerate(combos):
                    if ct != t:
                        continue
                    score = float(np.where(keep, f2[:, c], empty).mean())
                    results.append({"tta": t, "gate": gate_name, "gate_thr": thr, "mask_thr": m, "min_area": a,
                                    "snap": s, "score": score})
    results.sort(key=lambda r: -r["score"])
    best = results[0]
    baseline = next(r for r in results if all(r[k] == BASELINE[k] for k in BASELINE))

    def best_where(**kw):
        return max(r["score"] for r in results if all(r[k] == v for k, v in kw.items()))

    report = {
        "baseline_config": baseline,
        "best_config": best,
        "gain_vs_baseline": best["score"] - baseline["score"],
        "best_by_tta": {t: best_where(tta=t) for t in TTA_MODES},
        "best_by_snap": {str(s): best_where(snap=s) for s in SNAP_GRID},
        "diagnostics_at_baseline": summarise(baseline_breakdowns, baseline_gated_out),
        "top_configs": results[:20],
    }
    log("diagnosis: " + json.dumps(report, indent=1, default=str))
    (Path(cfg["out_dir"]) / "diagnosis_report.json").write_text(json.dumps(report, indent=1, default=str))

    # --- test: write the submission with the best configuration ---
    log("test: inference + post-processing")
    test_items = ((image_id, *per_tta[best["tta"]], test_vit[image_id], best)
                  for image_id, per_tta in segmenter_probs(segmenter, test_dir, test_ids, cfg, device,
                                                           modes=[best["tta"]]))
    rows = [r for image_rows in parallel_stream(pool, n_workers, submission_rows, test_items) for r in image_rows]
    pool.shutdown()
    sub = pd.DataFrame(rows, columns=["ImageId", "EncodedPixels"])
    sub.to_csv(Path(cfg["out_dir"]) / "submission.csv", index=False)
    n_ship = sub.groupby("ImageId").EncodedPixels.apply(lambda s: (s != "").any()).mean()
    log(f"submission: {sub.ImageId.nunique()} images, {len(sub)} rows, {n_ship:.3f} with ships")


if __name__ == "__main__":
    main()
