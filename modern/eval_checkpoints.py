"""Evaluate each epoch checkpoint of a running fine-tune and log validation metrics to its Trackio run.

The training loop in ``small_ships.py`` only logs training loss. This sidecar waits for each "epoch N done" line
(written right after the checkpoint is saved), loads the checkpoint and computes, on the first ``--n`` images of the
shared validation split at full resolution with a single forward pass:

* ``valid/loss``: the same size-weighted loss as training;
* ``valid/f2``: the competition metric with the stage-2 thresholds (gate = mean of the U-Net aux head and the cached
  ViT gate >= 0.97, mask 0.5, minimum area 20 px);
* ``valid/recall_small``: share of ships under 150 px found at IoU > 0.5;
* ``valid/f2_ship_images``: F2 restricted to images with ships.

The base model is evaluated first, at step 0, as the reference.

usage: python eval_checkpoints.py RUN_DIR --name RUN_NAME [--n 1000]
"""

import argparse
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

import trackio
from airbus_modern import (CONFIG, ShipNet, border_from_labels, gate_sources, image_f2, labels_from_rles, load_masks,
                           remove_small, segment_instances, split_ids)
from diagnose import error_breakdown, make_pool, parallel_stream
from kaggle_tools.tracking import trackio_run
from small_ships import WEIGHT_A0, WEIGHT_MAX, small_ship_loss

EPOCH_RE = re.compile(r"epoch (\d+) done")
STEP_RE = re.compile(r"epoch \d+ step (\d+) ")
GATE, GATE_THR, MASK_THR, MIN_AREA, SMALL_AREA = "mean_aux_vit", 0.97, 0.5, 20, 150


class FullImageDataset(Dataset):
    def __init__(self, image_dir, image_ids, rles_by_id):
        self.image_dir, self.image_ids, self.rles_by_id = Path(image_dir), list(image_ids), rles_by_id

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        import cv2

        img = cv2.cvtColor(cv2.imread(str(self.image_dir / self.image_ids[idx])), cv2.COLOR_BGR2RGB)
        labels = labels_from_rles(self.rles_by_id.get(self.image_ids[idx], []))
        areas = np.bincount(labels.ravel())
        inst_weight = np.clip(np.sqrt(WEIGHT_A0 / np.maximum(areas, 1)), 1.0, WEIGHT_MAX)
        inst_weight[0] = 1.0
        body = (labels > 0).astype(np.float32)
        border = border_from_labels(labels).astype(np.float32)
        x = torch.from_numpy(img.astype(np.float32).transpose(2, 0, 1) / 255.0)
        return x, torch.from_numpy(np.stack([body, border])), torch.tensor([float(body.max() > 0)]), \
            torch.from_numpy(inst_weight[labels].astype(np.float32)), idx


def score(image_id, body, border, aux_prob, vit_prob, rles):
    """Worker: F2 with the stage-2 thresholds, plus the breakdown on images with ships."""
    truth = labels_from_rles(rles)
    labels = np.zeros(truth.shape, np.int32)
    if gate_sources(aux_prob, {"vit": vit_prob})[GATE] >= GATE_THR:
        labels = remove_small(segment_instances(body.astype(np.float32), border.astype(np.float32), MASK_THR),
                              MIN_AREA)
    has_ship = truth.max() > 0
    f2 = image_f2(truth, labels)
    bd = error_breakdown(truth, labels) if has_ship else None
    return f2, has_ship, bd


@torch.no_grad()
def evaluate(model, loader, ids, rles_by_id, vit, pool, n_workers, device):
    model.eval()
    losses, items = [], []
    for x, y, has_ship, weight, idx in loader:
        x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        y, has_ship, weight = (t.to(device, non_blocking=True) for t in (y, has_ship, weight))
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            seg, cls = model(x)
        losses.append(small_ship_loss(seg.float(), cls.float(), y, has_ship, weight).item() * len(x))
        probs = torch.sigmoid(seg.float()).half().cpu().numpy()
        aux = torch.sigmoid(cls.float())[:, 0].cpu().numpy()
        for i, j in enumerate(idx.tolist()):
            image_id = ids[j]
            items.append((image_id, probs[i, 0], probs[i, 1], float(aux[i]), vit[image_id],
                          rles_by_id.get(image_id, [])))
    results = list(parallel_stream(pool, n_workers, score, items))
    f2 = np.array([r[0] for r in results])
    ships = np.array([r[1] for r in results])
    bds = [r[2] for r in results if r[2] is not None]
    areas = np.concatenate([b["areas"] for b in bds])
    best_iou = np.concatenate([b["best_iou"] for b in bds])
    small = areas < SMALL_AREA
    return {"valid/loss": float(np.sum(losses) / len(ids)), "valid/f2": float(f2.mean()),
            "valid/f2_ship_images": float(f2[ships].mean()),
            "valid/recall_small": float((best_iou[small] > 0.5).mean()),
            "valid/recall_large": float((best_iou[~small] > 0.5).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--name", required=True)
    ap.add_argument("--project", default="airbus-ship-detection")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--data-dir", default="../data")
    ap.add_argument("--base-weights", default="../weights/unet_v2.pt")
    ap.add_argument("--gate-probs", default="../out_diag_parallel/vit_gate_probs.csv")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    run_dir = Path(args.run_dir)

    cfg = {**CONFIG, "data_dir": args.data_dir, "workers": args.workers, "postproc_workers": args.workers}
    device = torch.device("cuda")
    ids, rles_by_id = load_masks(cfg["data_dir"], 0)
    _, _, valid = split_ids(ids, rles_by_id, cfg["valid_images"], cfg["seed"])
    valid = list(valid[:args.n])
    gate = pd.read_csv(args.gate_probs)
    vit = dict(zip(gate.ImageId, gate.vit_prob))
    loader = DataLoader(FullImageDataset(Path(cfg["data_dir"]) / "train_v2", valid, rles_by_id), batch_size=8,
                        num_workers=args.workers, pin_memory=True, persistent_workers=True)
    pool, n_workers = make_pool(cfg)
    model = ShipNet(cfg["encoder"], None).to(device).to(memory_format=torch.channels_last)
    trackio_run(args.project, name=args.name, resume="allow", auto_log_gpu=False, auto_log_cpu=False)

    def log_checkpoint(path, step, epoch):
        for attempt in range(5):  # the file may still be being written; retry briefly
            try:
                model.load_state_dict(torch.load(path, map_location=device))
                break
            except Exception as exc:  # noqa: BLE001
                if attempt == 4:
                    raise
                print(f"retrying load of {path}: {exc}", flush=True)
                time.sleep(5)
        t0 = time.time()
        metrics = evaluate(model, loader, valid, rles_by_id, vit, pool, n_workers, device)
        trackio.log({**metrics, "valid/epoch": epoch}, step=step)
        print(f"epoch {epoch} step {step}: {metrics} ({time.time() - t0:.0f}s)", flush=True)

    log_checkpoint(args.base_weights, 0, -1)

    # Follow the training log; evaluate each checkpoint as soon as its "epoch N done" line appears.
    log_path, step = run_dir / "run.log", 0
    with log_path.open() as f:
        while True:
            line = f.readline()
            if not line:
                if not _training_running():
                    break
                time.sleep(15)
                continue
            if m := STEP_RE.search(line):
                step = int(m[1])
            elif m := EPOCH_RE.search(line):
                # Only the newest checkpoint exists on disk: skip epochs that were already overwritten.
                if not _newer_epoch_logged(log_path, int(m[1])):
                    log_checkpoint(run_dir / "unet_small_ft.pt", step, int(m[1]))
            elif "fine-tuning budget reached" in line:
                break
    pool.shutdown()
    trackio.finish()


def _training_running():
    import psutil

    return any("small_ships.py" in " ".join(p.info["cmdline"] or []) for p in psutil.process_iter(["cmdline"]))


def _newer_epoch_logged(log_path, epoch):
    return any(int(m[1]) > epoch for m in EPOCH_RE.finditer(log_path.read_text()))


if __name__ == "__main__":
    main()
