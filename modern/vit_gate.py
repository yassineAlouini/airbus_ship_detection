"""Stage 2: a dedicated ViT ship / no-ship gate on top of the U-Net from ``airbus_modern.py``.

The first full run showed that the gate is the bottleneck. Its best threshold sat on the grid edge (0.95), and it
added only +0.015 F2 on top of the ungated model. Image-level "is there a ship anywhere?" is a global decision, which
is where a ViT shines, so this stage:

1. Fine-tunes a DINOv2 ViT-S/14 (registers variant) classifier on *all* training images except the exact validation
   split used by the segmenter (same ``split_ids`` and seed), at 518 px.
2. Loads the already-trained U-Net (``model.pt`` from the ``airbus-ship-detection-modern-solution`` kernel output).
3. Tunes gate source (U-Net aux head / ViT / their mean) x gate / mask thresholds x minimum area with the exact
   competition metric, then writes ``submission.csv``.

The ViT is deliberately *not* used for the masks. With 14 px patches a whole small ship (5% are < 35 px) fits in one
token, and the metric's IoU thresholds up to 0.95 punish the resulting coarse boundaries.
"""

import glob
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from airbus_modern import CONFIG, MEAN, STD, ShipNet, _env, load_masks, split_ids, tune_postprocessing, write_submission

VIT_CONFIG = {
    "vit_model": _env("ASD_VIT_MODEL", "vit_small_patch14_reg4_dinov2.lvd142m"),
    "vit_pretrained": _env("ASD_VIT_PRETRAINED", 1, int),
    "vit_img": _env("ASD_VIT_IMG", 518, int),
    "vit_batch": _env("ASD_VIT_BATCH", 32, int),
    "vit_lr": _env("ASD_VIT_LR", 5e-5, float),
    "vit_epochs": _env("ASD_VIT_EPOCHS", 10, int),
    "vit_hours": _env("ASD_VIT_HOURS", 8.0, float),
    # Empty images per epoch as a multiple of the ship images (the full set is ~3.5x).
    "vit_empty_ratio": _env("ASD_VIT_EMPTY_RATIO", 1.5, float),
    # Empty = search every /kaggle/input subfolder for the stage-1 model.pt.
    "segmenter_path": _env("ASD_SEGMENTER_PATH", ""),
}


class GateDataset(Dataset):
    def __init__(self, image_dir, image_ids, labels, img_size, train=False):
        self.image_dir, self.image_ids, self.labels = Path(image_dir), list(image_ids), labels
        self.img_size, self.train = img_size, train

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img = cv2.cvtColor(cv2.imread(str(self.image_dir / image_id)), cv2.COLOR_BGR2RGB)
        # INTER_AREA keeps small ships visible when going from 768 to 518 px.
        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        if self.train:
            img = np.rot90(img, np.random.randint(4))
            if np.random.rand() < 0.5:
                img = img[:, ::-1]
            img = img.astype(np.float32)
            if np.random.rand() < 0.5:
                img = np.clip(img * np.random.uniform(0.8, 1.2) + np.random.uniform(-20, 20), 0, 255)
        x = torch.from_numpy(np.ascontiguousarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0)
        return x, torch.tensor([float(self.labels.get(image_id, 0))])


class ViTGate(nn.Module):
    def __init__(self, model_name, pretrained, img_size):
        super().__init__()
        import timm

        self.net = timm.create_model(model_name, pretrained=bool(pretrained), num_classes=1, img_size=img_size,
                                     drop_path_rate=0.1)
        self.register_buffer("mean", MEAN.clone())
        self.register_buffer("std", STD.clone())

    def forward(self, x):
        return self.net((x - self.mean) / self.std)


def param_groups(model, lr, head_lr_mult=10.0):
    """Pretrained backbone at ``lr``, freshly initialised head at ``lr * head_lr_mult``; no decay on norms/biases."""
    groups = {}
    for name, p in model.named_parameters():
        is_head = name.startswith("net.head")
        no_decay = p.ndim == 1 or "token" in name or "pos_embed" in name
        key = (is_head, no_decay)
        groups.setdefault(key, []).append(p)
    out = []
    for (is_head, no_decay), ps in groups.items():
        group_lr = lr * (head_lr_mult if is_head else 1.0)
        out.append({"params": ps, "lr": group_lr, "base_lr": group_lr, "weight_decay": 0.0 if no_decay else 0.05})
    return out


def train_gate(cfg, device):
    ids, rles_by_id = load_masks(cfg["data_dir"], cfg["limit"])
    # Same split as the segmenter: the validation images are unseen by *both* models.
    train_ship, train_empty, valid = split_ids(ids, rles_by_id, cfg["valid_images"], cfg["seed"])
    labels = {i: 1 for i in rles_by_id}
    print(f"gate train ship={len(train_ship)} empty={len(train_empty)} valid={len(valid)}", flush=True)
    image_dir = Path(cfg["data_dir"]) / "train_v2"

    model = ViTGate(cfg["vit_model"], cfg["vit_pretrained"], cfg["vit_img"]).to(device)
    opt = torch.optim.AdamW(param_groups(model, cfg["vit_lr"]))
    n_empty = min(len(train_empty), int(cfg["vit_empty_ratio"] * len(train_ship)))
    steps_per_epoch = max(1, (len(train_ship) + n_empty) // cfg["vit_batch"])
    total_steps, budget_s = steps_per_epoch * cfg["vit_epochs"], cfg["vit_hours"] * 3600
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    start, rng, global_step, progress = time.time(), np.random.RandomState(cfg["seed"]), 0, 0.0
    for epoch in range(cfg["vit_epochs"]):
        epoch_ids = np.concatenate([train_ship, rng.choice(train_empty, n_empty, replace=False)])
        loader = DataLoader(GateDataset(image_dir, epoch_ids, labels, cfg["vit_img"], train=True),
                            batch_size=cfg["vit_batch"], shuffle=True, num_workers=cfg["workers"], drop_last=True,
                            pin_memory=True)
        model.train()
        running = []
        for step, (x, y) in enumerate(loader):
            # Same wall-clock-aware warmup + cosine schedule as the segmenter.
            progress = max(global_step / total_steps, (time.time() - start) / budget_s)
            if progress >= 1:
                break
            scale = min(1.0, global_step / 500) * 0.5 * (1 + np.cos(np.pi * progress))
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * scale
            global_step += 1
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(x)
            loss = F.binary_cross_entropy_with_logits(logits.float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            running.append(loss.item())
            if step % 200 == 0:
                print(f"gate epoch {epoch} step {step}/{len(loader)} loss {np.mean(running[-200:]):.4f} "
                      f"elapsed {(time.time() - start) / 3600:.2f}h", flush=True)
        torch.save(model.state_dict(), Path(cfg["out_dir"]) / "vit_gate.pt")
        print(f"gate epoch {epoch} done, mean loss {np.mean(running):.4f}, {(time.time() - start) / 3600:.2f}h",
              flush=True)
        if progress >= 1:
            print("stopping: gate training budget reached", flush=True)
            break
    return model, valid, rles_by_id


@torch.no_grad()
def predict_gate(model, image_dir, image_ids, cfg, device):
    """{image_id: ship probability}, averaged over the 8 dihedral transforms."""
    model.eval()
    loader = DataLoader(GateDataset(image_dir, image_ids, {}, cfg["vit_img"]), batch_size=cfg["vit_batch"],
                        num_workers=cfg["workers"])
    probs = []
    for x, _ in loader:
        x = x.to(device)
        total = 0
        for k in range(4):
            xr = torch.rot90(x, k, (2, 3))
            for xi in (xr, torch.flip(xr, (3,))):
                with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                    total = total + torch.sigmoid(model(xi).float())[:, 0]
        probs.append((total / 8).cpu().numpy())
    return dict(zip(image_ids, np.concatenate(probs).tolist()))


def find_kernel_output(filename, override=""):
    """Locate a file produced by an earlier kernel and mounted through ``kernel_sources``."""
    # Kernel outputs are mounted under /kaggle/input/notebooks/<user>/<slug>/ (older layout: /kaggle/input/<slug>/).
    # Only look there: a generic depth-4 glob walks the 192k competition images and took ~16 min.
    candidates = glob.glob(f"/kaggle/input/notebooks/*/*/{filename}") + glob.glob(f"/kaggle/input/*/{filename}")
    path = override or next(iter(sorted(candidates)), "")
    if not path:
        raise FileNotFoundError(f"{filename} not found in /kaggle/input; pass its path explicitly")
    print(f"loading {filename} from {path}", flush=True)
    return path


def load_segmenter(cfg, device):
    path = find_kernel_output("model.pt", cfg["segmenter_path"])
    # Weights come from the pretrained checkpoint, so the encoder does not need to download ImageNet weights.
    model = ShipNet(cfg["encoder"], None).to(device).to(memory_format=torch.channels_last)
    model.load_state_dict(torch.load(path, map_location=device))
    return model


def main():
    try:
        import segmentation_models_pytorch  # noqa: F401
    except ImportError:
        import subprocess
        subprocess.run(["pip", "install", "-q", "segmentation-models-pytorch"], check=True)
    cfg = {**CONFIG, **VIT_CONFIG}
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(cfg["out_dir"]).mkdir(parents=True, exist_ok=True)
    print(json.dumps(cfg, indent=1), device, flush=True)

    # Load the stage-1 segmenter first (on CPU) so a missing / broken checkpoint fails in seconds, not after hours.
    segmenter = load_segmenter(cfg, torch.device("cpu"))
    gate, valid, rles_by_id = train_gate(cfg, device)
    valid_gate = predict_gate(gate, Path(cfg["data_dir"]) / "train_v2", valid, cfg, device)
    test_ids = sorted(pd.read_csv(Path(cfg["data_dir"]) / "sample_submission_v2.csv").ImageId)
    test_dir = Path(cfg["data_dir"]) / "test_v2"
    if cfg["limit"]:
        test_ids = [i for i in test_ids if (test_dir / i).exists()][:cfg["limit"]]
    test_gate = predict_gate(gate, test_dir, test_ids, cfg, device)
    del gate
    torch.cuda.empty_cache()

    segmenter = segmenter.to(device)
    best, report = tune_postprocessing(segmenter, valid, rles_by_id, cfg, device, extra_gates={"vit": valid_gate})
    report["best"] = best
    report["config"] = cfg
    (Path(cfg["out_dir"]) / "validation_report.json").write_text(json.dumps(report, indent=1))
    write_submission(segmenter, best, cfg, device, extra_gates={"vit": test_gate})


if __name__ == "__main__":
    main()
