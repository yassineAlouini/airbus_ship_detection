"""Stage 4: fine-tune the U-Net for small ships, then pick the best model / input-scale mix.

Stage 3 found that small ships are the bottleneck. Ships under 150 px are 29% of all ships; 33-54% of them are
missed, and the ones found reach a median IoU of only 0.65. This stage fine-tunes the stage-1 U-Net with:

* **Multi-scale crops.** A crop is taken at 1x, 1.5x or 2x zoom: a 384/s px window is resized to 384 px, so a 50 px
  ship covers up to 4x more pixels and feature-map cells.
* **Small-ship sampling.** Half of the crops are centred on a ship drawn with probability ~ 1/sqrt(area).
* **Size-weighted loss.** The body BCE weight of each ship pixel is clip(sqrt(500 / area), 1, 5), with the area
  measured at original resolution, so a 20 px ship counts 5x a large one.

Evaluation then compares, on the shared 4,000-image validation split and with the exact metric:

* the old model at 1x;
* the new model at 1x, 1.5x and 2x input scale (upsampled image, predictions resized back);
* the new model averaged over scales;
* old + new averaged.

It also gives the per-size error breakdown for each, and writes ``submission.csv`` with the best one. The ViT gate
probabilities are reused from the stage-3 cache (``vit_gate_probs.csv``).
"""

import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from airbus_modern import (
    CONFIG,
    IMG_SHAPE,
    ShipNet,
    TestDataset,
    _env,
    border_from_labels,
    gate_sources,
    image_f2,
    labels_from_rles,
    load_masks,
    predict_tta,
    remove_small,
    rles_from_labels,
    segment_instances,
    soft_dice_loss,
    split_ids,
)
from diagnose import error_breakdown, log, make_pool, parallel_stream, summarise

SMALL_CONFIG = {
    "base_weights": _env("ASD_BASE_WEIGHTS", "../weights/unet_v2.pt"),
    "gate_probs": _env("ASD_GATE_PROBS", "../out_diag_parallel/vit_gate_probs.csv"),
    "ft_hours": _env("ASD_FT_HOURS", 5.0, float),
    "ft_epochs": _env("ASD_FT_EPOCHS", 30, int),
    "ft_lr": _env("ASD_FT_LR", 1.5e-4, float),
    "ft_empty_ratio": _env("ASD_FT_EMPTY_RATIO", 0.5, float),
    "infer_batch": _env("ASD_INFER_BATCH", 8, int),
    "postproc_workers": _env("ASD_POSTPROC_WORKERS", 12, int),
}

SCALES, SCALE_P = (1.0, 1.5, 2.0), (0.4, 0.3, 0.3)
SMALL_FOCUS = 0.5  # share of crops centred on a ship drawn with p ~ 1/sqrt(area)
WEIGHT_A0, WEIGHT_MAX = 500.0, 5.0

# Candidate probability maps ("modes") scored on validation; each maps to (model, input scales averaged).
MODES = {
    "old": [("old", 1.0)],
    "new_s1": [("new", 1.0)],
    "new_s1.5": [("new", 1.5)],
    "new_s2": [("new", 2.0)],
    "new_ms": [("new", 1.0), ("new", 1.5), ("new", 2.0)],
    "old+new_ms": [("old", 1.0), ("new", 1.0), ("new", 1.5), ("new", 2.0)],
}
GATE_GRID = [0.9, 0.93, 0.95, 0.97, 0.98]
MASK_GRID = [0.4, 0.5, 0.6]
AREA_GRID = [0, 10, 20, 40]
COMBOS = [(md, m, a) for md in MODES for m in MASK_GRID for a in AREA_GRID]
REF = {"mask_thr": 0.5, "min_area": 20}  # thresholds of the stage-2 submission, used for the per-mode breakdown
REF_GATE = ("mean_aux_vit", 0.97)


# ---------------------------------------------------------------------------------------------------------------------
# Training data and loss
# ---------------------------------------------------------------------------------------------------------------------


class MultiScaleShipDataset(Dataset):
    def __init__(self, image_dir, image_ids, rles_by_id, crop=384):
        self.image_dir, self.image_ids, self.rles_by_id, self.crop = Path(image_dir), list(image_ids), rles_by_id, crop

    def __len__(self):
        return len(self.image_ids)

    def _origin(self, labels, areas, win):
        n = len(areas) - 1
        if n > 0 and np.random.rand() < SMALL_FOCUS + 0.25:
            if np.random.rand() < SMALL_FOCUS / (SMALL_FOCUS + 0.25):
                p = 1 / np.sqrt(areas[1:])
                k = np.random.choice(n, p=p / p.sum()) + 1
                ys, xs = np.nonzero(labels == k)
            else:
                ys, xs = np.nonzero(labels)
            i = np.random.randint(len(ys))
            y0 = np.clip(ys[i] - np.random.randint(win), 0, IMG_SHAPE[0] - win)
            x0 = np.clip(xs[i] - np.random.randint(win), 0, IMG_SHAPE[1] - win)
            return y0, x0
        return np.random.randint(IMG_SHAPE[0] - win + 1), np.random.randint(IMG_SHAPE[1] - win + 1)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img = cv2.cvtColor(cv2.imread(str(self.image_dir / image_id)), cv2.COLOR_BGR2RGB)
        labels = labels_from_rles(self.rles_by_id.get(image_id, []))
        areas = np.bincount(labels.ravel())
        scale = np.random.choice(SCALES, p=SCALE_P)
        win = int(round(self.crop / scale))
        y0, x0 = self._origin(labels, areas, win)
        img, labels = img[y0:y0 + win, x0:x0 + win], labels[y0:y0 + win, x0:x0 + win]
        if win != self.crop:
            img = cv2.resize(img, (self.crop, self.crop), interpolation=cv2.INTER_CUBIC)
            idx_map = (np.arange(self.crop) * win / self.crop).astype(int)  # exact nearest-neighbour for labels
            labels = labels[idx_map][:, idx_map]
        k = np.random.randint(4)
        img, labels = np.rot90(img, k), np.rot90(labels, k)
        if np.random.rand() < 0.5:
            img, labels = img[:, ::-1], labels[:, ::-1]
        img = img.astype(np.float32)
        if np.random.rand() < 0.5:
            img = np.clip(img * np.random.uniform(0.8, 1.2) + np.random.uniform(-20, 20), 0, 255)
        labels = np.ascontiguousarray(labels)
        body = (labels > 0).astype(np.float32)
        border = border_from_labels(labels).astype(np.float32)
        inst_weight = np.clip(np.sqrt(WEIGHT_A0 / np.maximum(areas, 1)), 1.0, WEIGHT_MAX)
        inst_weight[0] = 1.0
        weight = inst_weight[labels].astype(np.float32)
        x = torch.from_numpy(np.ascontiguousarray(img).transpose(2, 0, 1) / 255.0)
        return x, torch.from_numpy(np.stack([body, border])), torch.tensor([float(body.max() > 0)]), \
            torch.from_numpy(weight)


def small_ship_loss(seg, cls, y, has_ship, weight):
    pos_weight = torch.tensor(2.0, device=y.device)
    bce_body = F.binary_cross_entropy_with_logits(seg[:, 0], y[:, 0], weight=weight, pos_weight=pos_weight)
    bce_border = F.binary_cross_entropy_with_logits(seg[:, 1], y[:, 1], pos_weight=pos_weight)
    dice = soft_dice_loss(seg[:, 0], y[:, 0]) + 0.5 * soft_dice_loss(seg[:, 1], y[:, 1])
    return 0.5 * (bce_body + bce_border) + dice + 0.5 * F.binary_cross_entropy_with_logits(cls, has_ship)


def finetune(cfg, device, train_ship, train_empty, rles_by_id):
    model = ShipNet(cfg["encoder"], None).to(device).to(memory_format=torch.channels_last)
    model.load_state_dict(torch.load(cfg["base_weights"], map_location=device))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["ft_lr"], weight_decay=1e-4)
    n_empty = min(len(train_empty), int(cfg["ft_empty_ratio"] * len(train_ship)))
    steps_per_epoch = (len(train_ship) + n_empty) // cfg["batch_size"]
    total_steps, budget_s = steps_per_epoch * cfg["ft_epochs"], cfg["ft_hours"] * 3600
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    image_dir = Path(cfg["data_dir"]) / "train_v2"
    start, rng, step_all, progress = time.time(), np.random.RandomState(cfg["seed"] + 1), 0, 0.0
    out_path = Path(cfg["out_dir"]) / "unet_small_ft.pt"
    for epoch in range(cfg["ft_epochs"]):
        epoch_ids = np.concatenate([train_ship, rng.choice(train_empty, n_empty, replace=False)])
        loader = DataLoader(MultiScaleShipDataset(image_dir, epoch_ids, rles_by_id, crop=cfg["crop"]),
                            batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["workers"], drop_last=True,
                            pin_memory=True, persistent_workers=False, prefetch_factor=4)
        model.train()
        running = []
        for x, y, has_ship, weight in loader:
            progress = max(step_all / total_steps, (time.time() - start) / budget_s)
            if progress >= 1:
                break
            # Short warmup (the model is already trained), then cosine to zero over steps or wall-clock time.
            lr = cfg["ft_lr"] * min(1.0, step_all / 300) * 0.5 * (1 + np.cos(np.pi * progress))
            for group in opt.param_groups:
                group["lr"] = lr
            step_all += 1
            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            y, has_ship, weight = (t.to(device, non_blocking=True) for t in (y, has_ship, weight))
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                seg, cls = model(x)
            loss = small_ship_loss(seg.float(), cls.float(), y, has_ship, weight)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running.append(loss.item())
            if step_all % 500 == 0:
                log(f"epoch {epoch} step {step_all} loss {np.mean(running[-500:]):.4f} lr {lr:.2e} "
                    f"progress {progress:.2f}")
        torch.save(model.state_dict(), out_path)
        log(f"epoch {epoch} done, mean loss {np.mean(running):.4f}, checkpoint saved")
        if progress >= 1:
            log("fine-tuning budget reached")
            break
    return model


# ---------------------------------------------------------------------------------------------------------------------
# Multi-scale evaluation
# ---------------------------------------------------------------------------------------------------------------------


@torch.no_grad()
def probs_at_scale(model, x, scale):
    """Flip-4 TTA at a given input scale; probabilities resized back to the input resolution."""
    xs = x if scale == 1.0 else F.interpolate(x, scale_factor=scale, mode="bilinear", align_corners=False)
    seg, cls = predict_tta(model, xs)
    if scale != 1.0:
        seg = F.interpolate(seg, size=x.shape[-2:], mode="area")
    return seg, cls


def mode_probs(models, image_dir, image_ids, cfg, device, modes):
    """Yield (image_id, {mode: (body, border, aux_prob)}) for the requested modes."""
    needed = sorted({ms for md in modes for ms in MODES[md]})
    loader = DataLoader(TestDataset(image_dir, image_ids), batch_size=cfg["infer_batch"], num_workers=cfg["workers"],
                        pin_memory=True, prefetch_factor=4)
    k = 0
    for x in loader:
        x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        raw = {}
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            for name, scale in needed:
                raw[(name, scale)] = probs_at_scale(models[name], x, scale)
        out = {}
        for md in modes:
            seg = sum(raw[ms][0] for ms in MODES[md]) / len(MODES[md])
            cls = sum(raw[ms][1] for ms in MODES[md]) / len(MODES[md])
            out[md] = (seg.half().cpu().numpy(), cls.float().cpu().numpy())
        for i in range(len(x)):
            yield image_ids[k], {md: (out[md][0][i, 0], out[md][0][i, 1], float(out[md][1][i])) for md in modes}
            k += 1


def score_image(image_id, per_mode, rles, vit_prob):
    """Worker: F2 for every (mode, mask, area) combination + per-mode breakdown at the reference thresholds."""
    truth = labels_from_rles(rles)
    is_empty = truth.max() == 0
    row, cache, breakdowns = np.zeros(len(COMBOS)), {}, {}
    for c, (md, m, a) in enumerate(COMBOS):
        body, border, _ = per_mode[md]
        if (md, m) not in cache:
            cache[(md, m)] = segment_instances(body.astype(np.float32), border.astype(np.float32), m)
        labels = remove_small(cache[(md, m)], a)
        row[c] = (1.0 if is_empty else 0.0) if labels.max() == 0 else image_f2(truth, labels)
        if not is_empty and m == REF["mask_thr"] and a == REF["min_area"]:
            passed = gate_sources(per_mode[md][2], {"vit": vit_prob})[REF_GATE[0]] >= REF_GATE[1]
            breakdowns[md] = error_breakdown(truth, labels) if passed else None
    return image_id, row, is_empty, {md: per_mode[md][2] for md in MODES}, breakdowns


def submission_rows(image_id, body, border, aux_prob, vit_prob, best):
    labels = np.zeros(body.shape, np.int32)
    if gate_sources(aux_prob, {"vit": vit_prob})[best["gate"]] >= best["gate_thr"]:
        labels = remove_small(segment_instances(body.astype(np.float32), border.astype(np.float32),
                                                best["mask_thr"]), best["min_area"])
    rles = rles_from_labels(labels)
    return [(image_id, r) for r in rles] if rles else [(image_id, "")]


def main():
    cfg = {**CONFIG, **SMALL_CONFIG}
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    log("config " + json.dumps(cfg))

    ids, rles_by_id = load_masks(cfg["data_dir"], cfg["limit"])
    train_ship, train_empty, valid = split_ids(ids, rles_by_id, cfg["valid_images"], cfg["seed"])
    gate = pd.read_csv(cfg["gate_probs"])
    vit = dict(zip(gate.ImageId, gate.vit_prob))
    missing = [i for i in valid if i not in vit]
    if missing:
        raise ValueError(f"{len(missing)} validation images missing from the cached gate probabilities")
    log(f"train ship={len(train_ship)} empty={len(train_empty)} valid={len(valid)}")

    new = finetune(cfg, device, train_ship, train_empty, rles_by_id).eval()
    old = ShipNet(cfg["encoder"], None).to(device).to(memory_format=torch.channels_last).eval()
    old.load_state_dict(torch.load(cfg["base_weights"], map_location=device))
    models = {"old": old, "new": new}

    log("validation: multi-scale inference + scoring")
    pool, n_workers = make_pool(cfg)
    items = ((image_id, per_mode, rles_by_id.get(image_id, []), vit[image_id])
             for image_id, per_mode in mode_probs(models, Path(cfg["data_dir"]) / "train_v2", valid, cfg, device,
                                                  list(MODES)))
    f2, empty, aux, breakdowns, gated_out = [], [], {md: [] for md in MODES}, {md: [] for md in MODES}, \
        {md: 0 for md in MODES}
    for _, row, is_empty, aux_by_mode, bds in parallel_stream(pool, n_workers, score_image, items):
        f2.append(row)
        empty.append(1.0 if is_empty else 0.0)
        for md in MODES:
            aux[md].append(aux_by_mode[md])
            if not is_empty:
                if bds[md] is None:
                    gated_out[md] += 1
                else:
                    breakdowns[md].append(bds[md])
    f2, empty = np.stack(f2), np.array(empty)
    vit_valid = np.array([vit[i] for i in valid])
    results = []
    for md in MODES:
        for gate_name, probs in gate_sources(np.array(aux[md]), {"vit": vit_valid}).items():
            for thr in GATE_GRID:
                keep = probs >= thr
                for c, (cm, m, a) in enumerate(COMBOS):
                    if cm == md:
                        results.append({"mode": md, "gate": gate_name, "gate_thr": thr, "mask_thr": m,
                                        "min_area": a, "score": float(np.where(keep, f2[:, c], empty).mean())})
    results.sort(key=lambda r: -r["score"])
    best = results[0]
    report = {
        "best_config": best,
        "best_by_mode": {md: max((r for r in results if r["mode"] == md), key=lambda r: r["score"]) for md in MODES},
        "breakdown_by_mode_at_ref_thresholds": {md: summarise(breakdowns[md], gated_out[md]) for md in MODES},
        "top_configs": results[:20],
        "config": cfg,
    }
    (out_dir / "small_ships_report.json").write_text(json.dumps(report, indent=1, default=str))
    log("best by mode: " + json.dumps({md: round(r["score"], 5) for md, r in report["best_by_mode"].items()}))
    log("best config: " + json.dumps(best))

    log("test: inference + post-processing")
    test_dir = Path(cfg["data_dir"]) / "test_v2"
    test_ids = sorted(pd.read_csv(Path(cfg["data_dir"]) / "sample_submission_v2.csv").ImageId)
    if cfg["limit"]:
        test_ids = [i for i in test_ids if (test_dir / i).exists()][:cfg["limit"]]
    test_items = ((image_id, *per_mode[best["mode"]], vit[image_id], best)
                  for image_id, per_mode in mode_probs(models, test_dir, test_ids, cfg, device, [best["mode"]]))
    rows = [r for rs in parallel_stream(pool, n_workers, submission_rows, test_items) for r in rs]
    pool.shutdown()
    sub = pd.DataFrame(rows, columns=["ImageId", "EncodedPixels"])
    sub.to_csv(out_dir / "submission.csv", index=False)
    n_ship = sub.groupby("ImageId").EncodedPixels.apply(lambda s: (s != "").any()).mean()
    log(f"submission: {sub.ImageId.nunique()} images, {len(sub)} rows, {n_ship:.3f} with ships")


if __name__ == "__main__":
    main()
