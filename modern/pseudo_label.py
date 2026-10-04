"""Stage 5: pseudo-label the test scenes with the best ensemble, then fine-tune a student on train + pseudo-labels.

Stage 5 exists because of section 5.5: the test set is made of scenes never seen in training, and no validation
drawn from the training scenes predicts the leaderboard. Pseudo-labelling is the standard way to adapt to unseen
scenes without using any test label.

1. **Teacher:** the U-Nets in ``--teacher`` averaged (flip-4 TTA, 1x), with the gate mean(aux, ViT).
2. **Pseudo-labels:**
   * an image with gate < 0.05 and max ship probability < 0.3 becomes a confident empty image;
   * an image with gate >= 0.97 gets its instances (mask 0.5, minimum area 20 px) as labels, and every pixel with
     probability in (0.3, 0.7) is masked out of the loss (weight 0);
   * every other image is skipped.
3. **Student:** fine-tuned from ``--init`` with the plain stage-1 recipe (1x crops, BCE + Dice). Each epoch mixes
   real training images with the pseudo-labelled test images, the pseudo share bounded at ~25% so teacher mistakes
   are not over-learnt. Training curves go to Trackio (project ``airbus-ship-detection``).

usage: python pseudo_label.py --out ../out_pseudo --teacher W1 W2 --init W1 [--hours 2]
"""

import argparse
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
    border_from_labels,
    gate_sources,
    labels_from_rles,
    load_masks,
    remove_small,
    rle_decode,
    rle_encode,
    rles_from_labels,
    segment_instances,
    split_ids,
)
from diagnose import log, make_pool, parallel_stream
from small_ships import MODES, SMALL_CONFIG, mode_probs

EMPTY_GATE, EMPTY_MAX_PROB = 0.05, 0.3
SHIP_GATE, MASK_THR, MIN_AREA = 0.97, 0.5, 20
IGNORE_LOW, IGNORE_HIGH = 0.3, 0.7


# ---------------------------------------------------------------------------------------------------------------------
# 1-2. Teacher predictions -> pseudo-labels
# ---------------------------------------------------------------------------------------------------------------------


def make_pseudo_label(image_id, body, border, aux_prob, vit_prob):
    """Worker: (image_id, kind, instance RLEs joined by '|', ignore-band RLE)."""
    body, border = body.astype(np.float32), border.astype(np.float32)
    gate = gate_sources(aux_prob, {"vit": vit_prob})["mean_aux_vit"]
    if gate < EMPTY_GATE and body.max() < EMPTY_MAX_PROB:
        return image_id, "empty", "", ""
    if gate >= SHIP_GATE:
        labels = remove_small(segment_instances(body, border, MASK_THR), MIN_AREA)
        if labels.max() > 0:
            ignore = ((body > IGNORE_LOW) & (body < IGNORE_HIGH)).astype(np.uint8)
            return image_id, "ships", "|".join(rles_from_labels(labels)), rle_encode(ignore) if ignore.any() else ""
    return image_id, "skipped", "", ""


def build_pseudo_labels(teacher_paths, cfg, device, out_dir):
    models = {}
    for i, path in enumerate(teacher_paths):
        m = ShipNet(cfg["encoder"], None).to(device).to(memory_format=torch.channels_last).eval()
        m.load_state_dict(torch.load(path, map_location=device))
        models[f"teacher{i}"] = m
    MODES["teacher"] = [(name, 1.0) for name in models]
    gate = pd.read_csv(cfg["gate_probs"])
    vit = dict(zip(gate.ImageId, gate.vit_prob))
    test_dir = Path(cfg["data_dir"]) / "test_v2"
    test_ids = sorted(pd.read_csv(Path(cfg["data_dir"]) / "sample_submission_v2.csv").ImageId)
    pool, n_workers = make_pool(cfg)
    items = ((i, *pm["teacher"], vit[i]) for i, pm in mode_probs(models, test_dir, test_ids, cfg, device, ["teacher"]))
    rows = list(parallel_stream(pool, n_workers, make_pseudo_label, items))
    pool.shutdown()
    del models
    torch.cuda.empty_cache()
    df = pd.DataFrame(rows, columns=["ImageId", "kind", "rles", "ignore"])
    df.to_csv(out_dir / "pseudo_labels.csv", index=False)
    log("pseudo-labels: " + json.dumps(df.kind.value_counts().to_dict()))
    return df


# ---------------------------------------------------------------------------------------------------------------------
# 3. Student
# ---------------------------------------------------------------------------------------------------------------------


class MixedDataset(Dataset):
    """Items are (image path, list of instance RLEs, ignore RLE); 1x random crops as in stage 1, plus a loss weight."""

    def __init__(self, items, crop=384):
        self.items, self.crop = items, crop

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, rles, ignore_rle = self.items[idx]
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        labels = labels_from_rles(rles)
        weight = 1.0 - (rle_decode(ignore_rle) if ignore_rle else np.zeros(IMG_SHAPE, np.uint8)).astype(np.float32)
        size = self.crop
        if labels.max() > 0 and np.random.rand() < 0.66:
            ys, xs = np.nonzero(labels)
            i = np.random.randint(len(ys))
            y0 = np.clip(ys[i] - np.random.randint(size), 0, IMG_SHAPE[0] - size)
            x0 = np.clip(xs[i] - np.random.randint(size), 0, IMG_SHAPE[1] - size)
        else:
            y0, x0 = np.random.randint(IMG_SHAPE[0] - size + 1), np.random.randint(IMG_SHAPE[1] - size + 1)
        img, labels, weight = (a[y0:y0 + size, x0:x0 + size] for a in (img, labels, weight))
        k = np.random.randint(4)
        img, labels, weight = (np.rot90(a, k) for a in (img, labels, weight))
        if np.random.rand() < 0.5:
            img, labels, weight = (a[:, ::-1] for a in (img, labels, weight))
        img = img.astype(np.float32)
        if np.random.rand() < 0.5:
            img = np.clip(img * np.random.uniform(0.8, 1.2) + np.random.uniform(-20, 20), 0, 255)
        labels = np.ascontiguousarray(labels)
        body = (labels > 0).astype(np.float32)
        border = border_from_labels(labels).astype(np.float32)
        x = torch.from_numpy(np.ascontiguousarray(img).transpose(2, 0, 1) / 255.0)
        has_ship = torch.tensor([float(len(rles) > 0)])
        return x, torch.from_numpy(np.stack([body, border])), has_ship, torch.from_numpy(np.ascontiguousarray(weight))


def masked_loss(seg, cls, y, has_ship, weight):
    """Stage-1 loss with the ignore band masked out of BCE and Dice."""
    w = weight[:, None].expand_as(y)
    bce = F.binary_cross_entropy_with_logits(seg, y, weight=w, pos_weight=torch.tensor(2.0, device=y.device))
    prob = torch.sigmoid(seg) * w
    target = y * w

    def dice(c):
        return 1 - (2 * (prob[:, c] * target[:, c]).sum() + 1) / (prob[:, c].sum() + target[:, c].sum() + 1)

    return bce + dice(0) + 0.5 * dice(1) + 0.5 * F.binary_cross_entropy_with_logits(cls, has_ship)


def train_student(cfg, device, pseudo, init_path, hours, out_dir):
    import trackio
    from kaggle_tools.tracking import trackio_run

    ids, rles_by_id = load_masks(cfg["data_dir"], 0)
    train_ship, train_empty, _ = split_ids(ids, rles_by_id, cfg["valid_images"], cfg["seed"])
    train_dir, test_dir = Path(cfg["data_dir"]) / "train_v2", Path(cfg["data_dir"]) / "test_v2"
    real_ship = [(str(train_dir / i), rles_by_id[i], "") for i in train_ship]
    real_empty = [(str(train_dir / i), [], "") for i in train_empty]
    ps = pseudo[pseudo.kind == "ships"]
    pe = pseudo[pseudo.kind == "empty"]
    pseudo_ship = [(str(test_dir / r.ImageId), r.rles.split("|"), r.ignore if isinstance(r.ignore, str) else "")
                   for r in ps.itertuples()]
    pseudo_empty = [(str(test_dir / i), [], "") for i in pe.ImageId]
    # Per epoch: 20k real ship + 10k real empty images, and pseudo images at ~25% of the mix (ship-heavy).
    n_real_ship, n_real_empty = min(20000, len(real_ship)), 10000
    n_pseudo_ship = min(len(pseudo_ship) * 2, 5000)
    n_pseudo_empty = min(len(pseudo_empty), 10000 - n_pseudo_ship)
    log(f"student mix per epoch: real ship {n_real_ship}, real empty {n_real_empty}, pseudo ship {n_pseudo_ship} "
        f"(from {len(pseudo_ship)}), pseudo empty {n_pseudo_empty} (from {len(pseudo_empty)})")

    model = ShipNet(cfg["encoder"], None).to(device).to(memory_format=torch.channels_last)
    model.load_state_dict(torch.load(init_path, map_location=device))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr_student"], weight_decay=1e-4)
    scaler = torch.amp.GradScaler()
    rng = np.random.default_rng(cfg["seed"])
    trackio_run("airbus-ship-detection", name=cfg["run_name"], config={k: v for k, v in cfg.items()
                                                                      if isinstance(v, (int, float, str))},
                auto_log_gpu=False, auto_log_cpu=False)
    start, step, budget = time.time(), 0, hours * 3600
    epoch, progress = 0, 0.0
    while progress < 1:
        def pick(pool, n):
            return [pool[j] for j in rng.choice(len(pool), n, replace=n > len(pool))]

        items = (pick(real_ship, n_real_ship) + pick(real_empty, n_real_empty) + pick(pseudo_ship, n_pseudo_ship)
                 + pick(pseudo_empty, n_pseudo_empty))
        loader = DataLoader(MixedDataset(items, cfg["crop"]), batch_size=cfg["batch_size"], shuffle=True,
                            num_workers=cfg["workers"], drop_last=True, pin_memory=True, prefetch_factor=4)
        model.train()
        losses = []
        for x, y, has_ship, weight in loader:
            progress = (time.time() - start) / budget
            if progress >= 1:
                break
            lr = cfg["lr_student"] * min(1.0, step / 300) * 0.5 * (1 + np.cos(np.pi * progress))
            for g in opt.param_groups:
                g["lr"] = lr
            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            y, has_ship, weight = (t.to(device, non_blocking=True) for t in (y, has_ship, weight))
            with torch.autocast("cuda", dtype=torch.float16):
                seg, cls = model(x)
            loss = masked_loss(seg.float(), cls.float(), y, has_ship, weight)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(loss.item())
            step += 1
            if step % 200 == 0:
                trackio.log({"train/loss": float(np.mean(losses[-200:])), "train/lr": lr, "train/progress": progress,
                             "train/epoch": epoch}, step=step)
        torch.save(model.state_dict(), out_dir / "student.pt")
        log(f"epoch {epoch} done, step {step}, mean loss {np.mean(losses):.4f}, progress {progress:.2f}")
        trackio.log({"epoch/mean_loss": float(np.mean(losses))}, step=step)
        epoch += 1
    trackio.finish()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../out_pseudo")
    ap.add_argument("--teacher", nargs="+", required=True)
    ap.add_argument("--init", required=True)
    ap.add_argument("--hours", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--name", default="stage5-pseudo-label")
    ap.add_argument("--data-dir", default="../data")
    ap.add_argument("--reuse-pseudo", action="store_true", help="reuse OUT/pseudo_labels.csv if it exists")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = {**CONFIG, **SMALL_CONFIG, "data_dir": args.data_dir, "workers": 14, "postproc_workers": 12,
           "batch_size": 32, "lr_student": args.lr, "run_name": args.name}
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    log("config " + json.dumps({k: v for k, v in cfg.items() if isinstance(v, (int, float, str))}))
    pseudo_path = out / "pseudo_labels.csv"
    if args.reuse_pseudo and pseudo_path.exists():
        pseudo = pd.read_csv(pseudo_path, keep_default_na=False)
    else:
        log("teacher: pseudo-labelling the test set")
        pseudo = build_pseudo_labels(args.teacher, cfg, device, out)
    log("student: fine-tuning")
    train_student(cfg, device, pseudo, args.init, args.hours, out)
    log("student done")


if __name__ == "__main__":
    main()
