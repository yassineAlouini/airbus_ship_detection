"""Modern single-kernel solution for the Airbus Ship Detection Challenge.

One script that trains, validates (with the real competition metric), tunes the post-processing and writes
``submission.csv``. It is meant to run as a Kaggle GPU kernel (the competition data is mounted there), but every
path and budget can be overridden through environment variables so it can be smoke-tested locally on CPU.

Pipeline summary (see ``MODERN_SOLUTION.md`` for the reasoning):

1. U-Net with an ImageNet-pretrained EfficientNetV2-S encoder, trained at *full resolution* on 384x384 crops.
2. Two segmentation channels (ship body + ship border) so that touching ships can be split into instances.
3. An auxiliary image-level "has ship" head that gates the whole image: with ~78% empty images, a single false
   positive pixel blob on an empty image costs a full point for that image, so this gate matters most.
4. Post-processing (gate threshold, mask threshold, minimum instance area) tuned on a held-out split with the exact
   competition metric (mean F2 over IoU thresholds 0.5:0.95), then flip TTA at inference.
"""

import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from torch.utils.data import DataLoader, Dataset

IMG_SHAPE = (768, 768)
IOU_THRESHOLDS = np.arange(0.5, 1.0, 0.05)


def _env(name, default, cast=str):
    return cast(os.environ.get(name, default))


CONFIG = {
    "data_dir": _env("ASD_DATA_DIR", "/kaggle/input/competitions/airbus-ship-detection"),
    "out_dir": _env("ASD_OUT_DIR", "/kaggle/working"),
    "encoder": _env("ASD_ENCODER", "tu-tf_efficientnetv2_s"),
    "encoder_weights": _env("ASD_ENCODER_WEIGHTS", "imagenet"),
    "crop": _env("ASD_CROP", 384, int),
    "batch_size": _env("ASD_BATCH_SIZE", 24, int),
    "lr": _env("ASD_LR", 4e-4, float),
    "max_epochs": _env("ASD_MAX_EPOCHS", 16, int),
    # Wall-clock budget for training (Kaggle kernels are killed after 12h).
    "train_hours": _env("ASD_TRAIN_HOURS", 8.5, float),
    # Empty images seen per epoch, as a fraction of the images with ships.
    "empty_ratio": _env("ASD_EMPTY_RATIO", 1.0, float),
    "valid_images": _env("ASD_VALID_IMAGES", 4000, int),
    "workers": _env("ASD_WORKERS", 4, int),
    # Limit the number of train / test images (smoke tests only).
    "limit": _env("ASD_LIMIT", 0, int),
    "seed": 42,
}


# ---------------------------------------------------------------------------------------------------------------------
# RLE helpers (column-major, 1-indexed, as in the competition)
# ---------------------------------------------------------------------------------------------------------------------


def rle_decode(mask_rle, shape=IMG_SHAPE):
    s = np.asarray(mask_rle.split(), dtype=int)
    starts, lengths = s[0::2] - 1, s[1::2]
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, ln in zip(starts, lengths):
        img[lo:lo + ln] = 1
    return img.reshape(shape[::-1]).T


def rle_encode(mask):
    pixels = np.concatenate([[0], mask.T.flatten(), [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


# ---------------------------------------------------------------------------------------------------------------------
# Competition metric: mean over images of the F2 score averaged over IoU thresholds 0.5, 0.55, ..., 0.95
# ---------------------------------------------------------------------------------------------------------------------


def _iou_matrix(true_labels, pred_labels):
    n_true, n_pred = int(true_labels.max()), int(pred_labels.max())
    both = np.bincount(true_labels.ravel() * (n_pred + 1) + pred_labels.ravel(),
                       minlength=(n_true + 1) * (n_pred + 1)).reshape(n_true + 1, n_pred + 1)
    inter = both[1:, 1:]
    area_true, area_pred = both[1:, :].sum(1), both[:, 1:].sum(0)
    union = area_true[:, None] + area_pred[None, :] - inter
    return inter / np.maximum(union, 1)


def image_f2(true_labels, pred_labels):
    """F2 of one image, given integer instance label maps (0 = background)."""
    n_true, n_pred = int(true_labels.max()), int(pred_labels.max())
    if n_true == 0 and n_pred == 0:
        return 1.0
    if n_true == 0 or n_pred == 0:
        return 0.0
    iou = _iou_matrix(true_labels, pred_labels)
    scores = []
    for t in IOU_THRESHOLDS:
        # IoU > 0.5 guarantees a one-to-one matching, so counting rows / columns is exact.
        hits = iou > t
        tp = hits.any(1).sum()
        fn = n_true - tp
        fp = n_pred - hits.any(0).sum()
        scores.append(5 * tp / (5 * tp + 4 * fn + fp))
    return float(np.mean(scores))


def labels_from_rles(rles):
    labels = np.zeros(IMG_SHAPE, dtype=np.int32)
    for k, rle in enumerate(r for r in rles if isinstance(r, str) and r):
        labels[rle_decode(rle).astype(bool)] = k + 1
    return labels


# ---------------------------------------------------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------------------------------------------------


def border_from_labels(labels, width=2):
    """Pixels on the border of any instance, plus the gaps between touching instances."""
    border = np.zeros(labels.shape, dtype=np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    for k in range(1, int(labels.max()) + 1):
        inst = (labels == k).astype(np.uint8)
        border |= cv2.dilate(inst, kernel, iterations=width) - cv2.erode(inst, kernel, iterations=1)
    return border


class ShipDataset(Dataset):
    def __init__(self, image_dir, image_ids, rles_by_id, crop=None, train=False):
        self.image_dir, self.image_ids, self.rles_by_id = Path(image_dir), list(image_ids), rles_by_id
        self.crop, self.train = crop, train

    def __len__(self):
        return len(self.image_ids)

    def _crop_origin(self, labels):
        size = self.crop
        # Two thirds of the crops are centred on a ship (when there is one) so that small ships are not starved.
        if labels.max() > 0 and np.random.rand() < 0.66:
            ys, xs = np.nonzero(labels)
            i = np.random.randint(len(ys))
            y0 = np.clip(ys[i] - np.random.randint(size), 0, IMG_SHAPE[0] - size)
            x0 = np.clip(xs[i] - np.random.randint(size), 0, IMG_SHAPE[1] - size)
            return y0, x0
        return np.random.randint(IMG_SHAPE[0] - size + 1), np.random.randint(IMG_SHAPE[1] - size + 1)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img = cv2.cvtColor(cv2.imread(str(self.image_dir / image_id)), cv2.COLOR_BGR2RGB)
        labels = labels_from_rles(self.rles_by_id.get(image_id, []))
        if self.train:
            if self.crop:
                y0, x0 = self._crop_origin(labels)
                img = img[y0:y0 + self.crop, x0:x0 + self.crop]
                labels = labels[y0:y0 + self.crop, x0:x0 + self.crop]
            # Satellite imagery has no canonical orientation: all 8 dihedral transforms are valid.
            k = np.random.randint(4)
            img, labels = np.rot90(img, k), np.rot90(labels, k)
            if np.random.rand() < 0.5:
                img, labels = img[:, ::-1], labels[:, ::-1]
            img = img.astype(np.float32)
            if np.random.rand() < 0.5:
                img = img * np.random.uniform(0.8, 1.2) + np.random.uniform(-20, 20)
            img = np.clip(img, 0, 255)
        body = (labels > 0).astype(np.float32)
        border = border_from_labels(np.ascontiguousarray(labels)).astype(np.float32)
        x = torch.from_numpy(np.ascontiguousarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0)
        y = torch.from_numpy(np.stack([body, border]))
        return x, y, torch.tensor([float(body.max() > 0)])


class TestDataset(Dataset):
    def __init__(self, image_dir, image_ids):
        self.image_dir, self.image_ids = Path(image_dir), list(image_ids)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img = cv2.cvtColor(cv2.imread(str(self.image_dir / self.image_ids[idx])), cv2.COLOR_BGR2RGB)
        return torch.from_numpy(img.astype(np.float32).transpose(2, 0, 1) / 255.0)


# ---------------------------------------------------------------------------------------------------------------------
# Model and losses
# ---------------------------------------------------------------------------------------------------------------------

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class ShipNet(nn.Module):
    def __init__(self, encoder, encoder_weights):
        super().__init__()
        import segmentation_models_pytorch as smp

        self.net = smp.Unet(encoder, encoder_weights=None if encoder_weights == "none" else encoder_weights, classes=2,
                            aux_params={"classes": 1, "dropout": 0.2})
        self.register_buffer("mean", MEAN.clone())
        self.register_buffer("std", STD.clone())

    def forward(self, x):
        return self.net((x - self.mean) / self.std)


def soft_dice_loss(logits, target, eps=1.0):
    prob = torch.sigmoid(logits)
    inter = (prob * target).sum()
    return 1 - (2 * inter + eps) / (prob.sum() + target.sum() + eps)


def loss_fn(seg_logits, cls_logits, y, has_ship):
    bce = F.binary_cross_entropy_with_logits(seg_logits, y, pos_weight=torch.tensor(2.0, device=y.device))
    dice = soft_dice_loss(seg_logits[:, 0], y[:, 0]) + 0.5 * soft_dice_loss(seg_logits[:, 1], y[:, 1])
    cls = F.binary_cross_entropy_with_logits(cls_logits, has_ship)
    return bce + dice + 0.5 * cls


@torch.no_grad()
def predict_tta(model, x):
    """Average sigmoid outputs over identity + 3 flips. Returns (seg probs [B,2,H,W], ship prob [B])."""
    seg_sum, cls_sum = 0, 0
    for dims in [(), (3,), (2,), (2, 3)]:
        xi = torch.flip(x, dims) if dims else x
        seg, cls = model(xi)
        seg = torch.sigmoid(seg.float())
        seg_sum = seg_sum + (torch.flip(seg, dims) if dims else seg)
        cls_sum = cls_sum + torch.sigmoid(cls.float())[:, 0]
    return seg_sum / 4, cls_sum / 4


# ---------------------------------------------------------------------------------------------------------------------
# Post-processing: probabilities -> non-overlapping instances
# ---------------------------------------------------------------------------------------------------------------------


def instances_from_probs(body_prob, border_prob, ship_prob, gate_thr, mask_thr, min_area):
    """Return an int32 instance label map. Instances never overlap (the competition rejects overlaps)."""
    if ship_prob < gate_thr:
        return np.zeros(body_prob.shape, dtype=np.int32)
    return remove_small(segment_instances(body_prob, border_prob, mask_thr), min_area)


def segment_instances(body_prob, border_prob, mask_thr):
    mask = body_prob > mask_thr
    if not mask.any():
        return np.zeros(body_prob.shape, dtype=np.int32)
    # Seeds = ship interiors (body minus predicted border); watershed grows them back to the full body, which
    # splits ships that are moored side by side and would otherwise be a single connected component.
    seeds, _ = ndi.label(mask & (border_prob < 0.5))
    labels = watershed(-body_prob, seeds, mask=mask) if seeds.max() > 0 else ndi.label(mask)[0]
    # Body pixels not reached by any seed become their own components.
    orphans, n_orphans = ndi.label(mask & (labels == 0))
    if n_orphans:
        labels = np.where(orphans > 0, orphans + labels.max(), labels)
    return labels.astype(np.int32)


def remove_small(labels, min_area):
    if labels.max() == 0 or min_area <= 0:
        return labels
    areas = np.bincount(labels.ravel())
    small = areas < min_area
    small[0] = False
    return relabel_sequential(np.where(small[labels], 0, labels))


def relabel_sequential(labels):
    uniq = np.unique(labels)
    lut = np.zeros(int(uniq.max()) + 1, dtype=np.int32)
    lut[uniq] = np.arange(len(uniq))
    return lut[labels]


def rles_from_labels(labels):
    return [rle_encode((labels == k).astype(np.uint8)) for k in range(1, int(labels.max()) + 1)]


# ---------------------------------------------------------------------------------------------------------------------
# Train / validate / predict
# ---------------------------------------------------------------------------------------------------------------------


def load_masks(data_dir, limit):
    masks = pd.read_csv(Path(data_dir) / "train_ship_segmentations_v2.csv")
    image_dir = Path(data_dir) / "train_v2"
    if limit:
        available = set(os.listdir(image_dir))
        masks = masks[masks.ImageId.isin(available)]
    rles_by_id = masks.dropna().groupby("ImageId").EncodedPixels.apply(list).to_dict()
    ids = masks.ImageId.unique()
    return ids, rles_by_id


def split_ids(ids, rles_by_id, n_valid, seed):
    rng = np.random.RandomState(seed)
    ids = rng.permutation(ids)
    # Validation keeps the natural ~22% ship prior so the gate threshold is tuned for the real test distribution.
    n_valid = min(n_valid, len(ids) // 4)
    valid, train = ids[:n_valid], ids[n_valid:]
    train_ship = np.array([i for i in train if i in rles_by_id])
    train_empty = np.array([i for i in train if i not in rles_by_id])
    return train_ship, train_empty, valid


def train(cfg, device):
    ids, rles_by_id = load_masks(cfg["data_dir"], cfg["limit"])
    train_ship, train_empty, valid = split_ids(ids, rles_by_id, cfg["valid_images"], cfg["seed"])
    print(f"train ship={len(train_ship)} empty={len(train_empty)} valid={len(valid)}", flush=True)
    image_dir = Path(cfg["data_dir"]) / "train_v2"

    model = ShipNet(cfg["encoder"], cfg["encoder_weights"]).to(device).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    n_empty = min(len(train_empty), int(cfg["empty_ratio"] * len(train_ship)))
    steps_per_epoch = max(1, (len(train_ship) + n_empty) // cfg["batch_size"])
    total_steps, budget_s = steps_per_epoch * cfg["max_epochs"], cfg["train_hours"] * 3600
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    start, rng = time.time(), np.random.RandomState(cfg["seed"])
    global_step, progress = 0, 0.0
    for epoch in range(cfg["max_epochs"]):
        # Fresh random subset of empty images every epoch: over the run the model sees most of them.
        epoch_ids = np.concatenate([train_ship, rng.choice(train_empty, n_empty, replace=False)])
        loader = DataLoader(ShipDataset(image_dir, epoch_ids, rles_by_id, crop=cfg["crop"], train=True),
                            batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["workers"], drop_last=True,
                            pin_memory=True, persistent_workers=False)
        model.train()
        running = []
        for step, (x, y, has_ship) in enumerate(loader):
            # Cosine schedule driven by whichever runs out first, steps or wall-clock time, so the learning rate is
            # always annealed by the end even when the kernel is slower than expected.
            progress = max(global_step / total_steps, (time.time() - start) / budget_s)
            if progress >= 1:
                break
            lr = cfg["lr"] * min(1.0, global_step / 500) * 0.5 * (1 + np.cos(np.pi * progress))
            for group in opt.param_groups:
                group["lr"] = lr
            global_step += 1
            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            y, has_ship = y.to(device, non_blocking=True), has_ship.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                seg, cls = model(x)
            loss = loss_fn(seg.float(), cls.float(), y, has_ship)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running.append(loss.item())
            if step % 200 == 0:
                print(f"epoch {epoch} step {step}/{len(loader)} loss {np.mean(running[-200:]):.4f} "
                      f"lr {lr:.2e} elapsed {(time.time() - start) / 3600:.2f}h", flush=True)
        torch.save(model.state_dict(), Path(cfg["out_dir"]) / "model.pt")
        hours = (time.time() - start) / 3600
        print(f"epoch {epoch} done, mean loss {np.mean(running):.4f}, {hours:.2f}h", flush=True)
        if progress >= 1:
            print("stopping: training budget reached", flush=True)
            break
    return model, valid, rles_by_id


def predict_probs(model, image_dir, image_ids, cfg, device):
    """Yield (image_id, body_prob, border_prob, ship_prob) with flip TTA at full resolution."""
    model.eval()
    loader = DataLoader(TestDataset(image_dir, image_ids), batch_size=8, num_workers=cfg["workers"])
    k = 0
    for x in loader:
        x = x.to(device).to(memory_format=torch.channels_last)
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            seg, cls = predict_tta(model, x)
        # float16 halves the RAM needed to cache validation predictions.
        seg, cls = seg.half().cpu().numpy(), cls.cpu().numpy()
        for i in range(len(x)):
            yield image_ids[k], seg[i, 0], seg[i, 1], float(cls[i])
            k += 1


# Finer steps near 1: the best gate threshold sat on the grid edge (0.95) in the first full run.
GATE_GRID = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.98, 0.99]
MASK_GRID = [0.3, 0.4, 0.5, 0.6, 0.7]
AREA_GRID = [0, 20, 40, 80, 120, 160]


def gate_sources(aux_prob, extra_probs):
    """Candidate image-level ship probabilities: the U-Net aux head, each extra gate, and its mean with the head."""
    sources = {"aux": aux_prob}
    for name, prob in extra_probs.items():
        sources[name] = prob
        sources[f"mean_aux_{name}"] = (aux_prob + prob) / 2
    return sources


def tune_postprocessing(model, valid, rles_by_id, cfg, device, extra_gates=None):
    """Grid-search the post-processing on the validation split with the exact competition metric.

    ``extra_gates`` maps a gate name to ``{image_id: ship probability}`` (e.g. a dedicated classifier); every gate
    source from ``gate_sources`` is searched alongside the thresholds.

    Predictions are streamed (caching 768x768 maps for thousands of images does not fit in RAM). The gate only
    zeroes whole images and min_area only drops instances, so one watershed per mask threshold is enough.
    """
    extra_gates = extra_gates or {}
    image_dir = Path(cfg["data_dir"]) / "train_v2"
    image_ids, aux_probs, empty_scores, f2 = [], [], [], []  # f2: [image, mask_thr, min_area]
    for image_id, body, border, ship_prob in predict_probs(model, image_dir, valid, cfg, device):
        truth = labels_from_rles(rles_by_id.get(image_id, []))
        body, border = body.astype(np.float32), border.astype(np.float32)
        scores = np.zeros((len(MASK_GRID), len(AREA_GRID)))
        for i, mask_thr in enumerate(MASK_GRID):
            labels = segment_instances(body, border, mask_thr)
            for j, min_area in enumerate(AREA_GRID):
                scores[i, j] = image_f2(truth, remove_small(labels, min_area))
        image_ids.append(image_id)
        aux_probs.append(ship_prob)
        empty_scores.append(1.0 if truth.max() == 0 else 0.0)
        f2.append(scores)
    empty_scores, f2 = np.array(empty_scores), np.stack(f2)
    extra = {name: np.array([probs[i] for i in image_ids]) for name, probs in extra_gates.items()}
    sources = gate_sources(np.array(aux_probs), extra)
    ships = empty_scores == 0
    results, gate_auc = [], {}
    for gate, ship_probs in sources.items():
        if ships.any() and (~ships).any():
            from sklearn.metrics import roc_auc_score
            gate_auc[gate] = float(roc_auc_score(ships, ship_probs))
        for gate_thr in GATE_GRID:
            gated = np.where((ship_probs >= gate_thr)[:, None, None], f2, empty_scores[:, None, None]).mean(0)
            for i, mask_thr in enumerate(MASK_GRID):
                for j, min_area in enumerate(AREA_GRID):
                    results.append({"gate": gate, "gate_thr": gate_thr, "mask_thr": mask_thr, "min_area": min_area,
                                    "score": float(gated[i, j])})
    results = sorted(results, key=lambda r: -r["score"])
    best = results[0]
    bi, bj = MASK_GRID.index(best["mask_thr"]), AREA_GRID.index(best["min_area"])
    gated_best = np.where(sources[best["gate"]] >= best["gate_thr"], f2[:, bi, bj], empty_scores)
    best_per_gate = {g: max(r["score"] for r in results if r["gate"] == g) for g in sources}
    breakdown = {"all_empty_baseline": float(empty_scores.mean()), "best_score": best["score"],
                 "score_on_ship_images": float(gated_best[ships].mean()) if ships.any() else None,
                 "score_on_empty_images": float(gated_best[~ships].mean()) if (~ships).any() else None,
                 "ungated_score": float(f2[:, bi, bj].mean()), "gate_auc": gate_auc,
                 "best_score_per_gate": best_per_gate,
                 # Score with a perfect gate: the ceiling any classifier improvement can reach with this segmenter.
                 "oracle_gate_score": float(np.where(ships, f2[:, bi, bj], 1.0).mean())}
    print(f"validation: {json.dumps(breakdown)}; top configs:", flush=True)
    for r in results[:5]:
        print("  ", r, flush=True)
    return best, {**breakdown, "grid": results}


def write_submission(model, best, cfg, device, extra_gates=None):
    extra_gates = extra_gates or {}
    test_dir = Path(cfg["data_dir"]) / "test_v2"
    test_ids = sorted(pd.read_csv(Path(cfg["data_dir"]) / "sample_submission_v2.csv").ImageId)
    if cfg["limit"]:
        test_ids = [i for i in test_ids if (test_dir / i).exists()][:cfg["limit"]]
    rows = []
    for image_id, body, border, aux_prob in predict_probs(model, test_dir, test_ids, cfg, device):
        sources = gate_sources(aux_prob, {name: probs[image_id] for name, probs in extra_gates.items()})
        ship_prob = sources[best.get("gate", "aux")]
        labels = instances_from_probs(body.astype(np.float32), border.astype(np.float32), ship_prob,
                                      best["gate_thr"], best["mask_thr"], best["min_area"])
        rles = rles_from_labels(labels)
        rows += [(image_id, r) for r in rles] if rles else [(image_id, "")]
    sub = pd.DataFrame(rows, columns=["ImageId", "EncodedPixels"])
    sub.to_csv(Path(cfg["out_dir"]) / "submission.csv", index=False)
    n_ship = sub.groupby("ImageId").EncodedPixels.apply(lambda s: (s != "").any()).mean()
    print(f"submission: {sub.ImageId.nunique()} images, {len(sub)} rows, {n_ship:.3f} with ships", flush=True)
    return sub


def main():
    try:
        import segmentation_models_pytorch  # noqa: F401
    except ImportError:
        import subprocess
        subprocess.run(["pip", "install", "-q", "segmentation-models-pytorch"], check=True)
    cfg = CONFIG
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(cfg["out_dir"]).mkdir(parents=True, exist_ok=True)
    print(json.dumps(cfg, indent=1), device, flush=True)
    model, valid, rles_by_id = train(cfg, device)
    best, report = tune_postprocessing(model, valid, rles_by_id, cfg, device)
    report["best"] = best
    report["config"] = cfg
    (Path(cfg["out_dir"]) / "validation_report.json").write_text(json.dumps(report, indent=1))
    write_submission(model, best, cfg, device)


if __name__ == "__main__":
    main()
