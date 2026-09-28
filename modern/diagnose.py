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
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from airbus_modern import (CONFIG, TestDataset, _env, gate_sources, image_f2, labels_from_rles, load_masks,
                           predict_dihedral, remove_small, rles_from_labels, segment_instances, snap_to_rectangles,
                           split_ids)
from vit_gate import VIT_CONFIG, ViTGate, find_kernel_output, load_segmenter, predict_gate

DIAG_CONFIG = {"vit_gate_path": _env("ASD_VIT_GATE_PATH", "")}

TTA_MODES = ["flip4", "dihedral8"]
GATE_GRID = [0.9, 0.93, 0.95, 0.97, 0.98, 0.99]
MASK_GRID = [0.4, 0.5, 0.6]
AREA_GRID = [0, 20, 40]
SNAP_GRID = [None, 0, 50, 150]  # None = no snapping; otherwise snap instances with at least this many pixels.
# Stage-2 submission, to check that this run reproduces its validation score.
BASELINE = {"tta": "flip4", "gate": "mean_aux_vit", "gate_thr": 0.97, "mask_thr": 0.5, "min_area": 20, "snap": None}
SIZE_BINS = [0, 50, 150, 500, 2000, np.inf]
IOU_THRESHOLDS = np.arange(0.5, 1.0, 0.05)


def segmenter_probs(model, image_dir, image_ids, cfg, device):
    """Yield (image_id, {tta_mode: (body, border, aux_prob)}) from one set of 8 forward passes per image."""
    model.eval()
    loader = DataLoader(TestDataset(image_dir, image_ids), batch_size=8, num_workers=cfg["workers"])
    k = 0
    for x in loader:
        x = x.to(device).to(memory_format=torch.channels_last)
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            seg4, cls4, seg8, cls8 = predict_dihedral(model, x)
        seg4, seg8 = seg4.half().cpu().numpy(), seg8.half().cpu().numpy()
        cls4, cls8 = cls4.cpu().numpy(), cls8.cpu().numpy()
        for i in range(len(x)):
            yield image_ids[k], {"flip4": (seg4[i, 0], seg4[i, 1], float(cls4[i])),
                                 "dihedral8": (seg8[i, 0], seg8[i, 1], float(cls8[i]))}
            k += 1


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


def config_key(c):
    return (c["tta"], c["mask_thr"], c["min_area"], c["snap"])


def main():
    cfg = {**CONFIG, **VIT_CONFIG, **DIAG_CONFIG}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(cfg["out_dir"]).mkdir(parents=True, exist_ok=True)
    print(json.dumps(cfg, indent=1), device, flush=True)

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
    print("gate probabilities done", flush=True)

    # --- validation: score every (tta, mask, area, snap) combination per image, gate applied afterwards ---
    combos = [(t, m, a, s) for t in TTA_MODES for m in MASK_GRID for a in AREA_GRID for s in SNAP_GRID]
    f2, aux, empty, baseline_breakdowns, baseline_gated_out = [], {t: [] for t in TTA_MODES}, [], [], 0
    for image_id, per_tta in segmenter_probs(segmenter, Path(cfg["data_dir"]) / "train_v2", valid, cfg, device):
        truth = labels_from_rles(rles_by_id.get(image_id, []))
        is_empty = truth.max() == 0
        row, cache = np.zeros(len(combos)), {}
        for c, (t, m, a, s) in enumerate(combos):
            body, border, _ = per_tta[t]
            key = (t, m)
            if key not in cache:
                cache[key] = segment_instances(body.astype(np.float32), border.astype(np.float32), m)
            labels = remove_small(cache[key], a)
            if labels.max() == 0:
                row[c] = 1.0 if is_empty else 0.0
                continue
            labels = labels if s is None else snap_to_rectangles(labels, s)
            row[c] = image_f2(truth, labels)
            if (t, m, a, s) == config_key(BASELINE) and not is_empty:
                sources = gate_sources(per_tta[t][2], {"vit": valid_vit[image_id]})
                if sources[BASELINE["gate"]] >= BASELINE["gate_thr"]:
                    baseline_breakdowns.append(error_breakdown(truth, labels))
        if not is_empty:
            sources = gate_sources(per_tta["flip4"][2], {"vit": valid_vit[image_id]})
            if sources[BASELINE["gate"]] < BASELINE["gate_thr"]:
                baseline_gated_out += 1
            elif remove_small(cache[("flip4", BASELINE["mask_thr"])], BASELINE["min_area"]).max() == 0:
                # Passed the gate but no instance survived: every true ship is a false negative.
                baseline_breakdowns.append(error_breakdown(truth, np.zeros_like(truth)))
        f2.append(row)
        empty.append(1.0 if is_empty else 0.0)
        for t in TTA_MODES:
            aux[t].append(per_tta[t][2])
    f2, empty = np.stack(f2), np.array(empty)
    vit = np.array([valid_vit[i] for i in valid])

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
    print("diagnosis:", json.dumps(report, indent=1, default=str), flush=True)
    (Path(cfg["out_dir"]) / "diagnosis_report.json").write_text(json.dumps(report, indent=1, default=str))

    # --- test: write the submission with the best configuration ---
    rows = []
    for image_id, per_tta in segmenter_probs(segmenter, test_dir, test_ids, cfg, device):
        body, border, aux_prob = per_tta[best["tta"]]
        labels = np.zeros(body.shape, np.int32)
        if gate_sources(aux_prob, {"vit": test_vit[image_id]})[best["gate"]] >= best["gate_thr"]:
            labels = postprocess(body.astype(np.float32), border.astype(np.float32), best["mask_thr"],
                                 best["min_area"], best["snap"])
        rles = rles_from_labels(labels)
        rows += [(image_id, r) for r in rles] if rles else [(image_id, "")]
    sub = pd.DataFrame(rows, columns=["ImageId", "EncodedPixels"])
    sub.to_csv(Path(cfg["out_dir"]) / "submission.csv", index=False)
    n_ship = sub.groupby("ImageId").EncodedPixels.apply(lambda s: (s != "").any()).mean()
    print(f"submission: {sub.ImageId.nunique()} images, {len(sub)} rows, {n_ship:.3f} with ships", flush=True)


if __name__ == "__main__":
    main()
