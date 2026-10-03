"""Measure train/validation scene leakage and check whether a leak-free validation subset tracks the leaderboard.

The 768 px images are crops of larger scenes, taken at 256 px aligned offsets and JPEG-encoded identically, so
overlapping crops share bit-identical 256x256 blocks. This script:

1. Hashes the 9 blocks of every train and test image (CPU, all cores).
2. Links images sharing a *textured* block (pixel std >= 8 and the block seen in <= 20 images, so flat sea does not
   create false links) and groups them into scenes with union-find.
3. Marks a validation image as leaky when its scene contains a training image (any image outside the 4,000-image
   validation split, since every such image could be sampled for training).
4. Re-scores the five variants that have leaderboard scores on all / clean / leaky validation images, so we can see
   whether the clean subset ranks them like the leaderboard does.

Outputs (in OUT_DIR): ``scenes.csv`` (image, scene, split, leaky), ``leakage_report.json``.
usage: python scene_leakage.py [--out ../out_scene]
"""

import argparse
import hashlib
import json
import os
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from airbus_modern import (CONFIG, ShipNet, TestDataset, gate_sources, image_f2, labels_from_rles, load_masks,
                           predict_dihedral, remove_small, segment_instances, split_ids)
from diagnose import log, make_pool, parallel_stream

BLOCK, MIN_STD, MAX_OWNERS = 256, 8.0, 20
# The five submissions with leaderboard scores (public, private), and how to reproduce each on validation.
VARIANTS = {
    "stage1": dict(model="old", tta="flip4", gate="aux", gate_thr=0.95, mask_thr=0.5, min_area=20,
                   lb=(0.73128, 0.84920)),
    "stage2": dict(model="old", tta="flip4", gate="mean_aux_vit", gate_thr=0.97, mask_thr=0.5, min_area=20,
                   lb=(0.73301, 0.85040)),
    "stage3": dict(model="old", tta="dihedral8", gate="vit", gate_thr=0.97, mask_thr=0.6, min_area=20,
                   lb=(0.73020, 0.84906)),
    "stage4": dict(model="new", tta="flip4", gate="mean_aux_vit", gate_thr=0.90, mask_thr=0.5, min_area=20,
                   lb=(0.72113, 0.84896)),
    "stage4_gate97": dict(model="new", tta="flip4", gate="mean_aux_vit", gate_thr=0.97, mask_thr=0.5,
                          min_area=20, lb=(0.73039, 0.84830)),
}


# ---------------------------------------------------------------------------------------------------------------------
# 1-2. Block hashing and scene grouping
# ---------------------------------------------------------------------------------------------------------------------


def block_hashes(path):
    img = cv2.imread(path)
    out = []
    for by in range(img.shape[0] // BLOCK):
        for bx in range(img.shape[1] // BLOCK):
            b = img[by * BLOCK:(by + 1) * BLOCK, bx * BLOCK:(bx + 1) * BLOCK]
            out.append((hashlib.blake2b(b.tobytes(), digest_size=12).digest(), float(b.std())))
    return out


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, i):
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def group_scenes(names, hashes):
    owners = defaultdict(set)
    for i, blocks in enumerate(hashes):
        for h, sd in blocks:
            if sd >= MIN_STD:
                owners[h].add(i)
    uf = UnionFind(len(names))
    links = 0
    for members in owners.values():
        if 2 <= len(members) <= MAX_OWNERS:
            members = list(members)
            for j in members[1:]:
                uf.union(members[0], j)
            links += len(members) - 1
    return np.array([uf.find(i) for i in range(len(names))]), links


# ---------------------------------------------------------------------------------------------------------------------
# 4. Re-score the leaderboard variants on validation
# ---------------------------------------------------------------------------------------------------------------------


def score_variants(image_id, probs, rles, vit_prob):
    """Worker: per-variant F2 for one validation image. ``probs`` maps (model, tta) -> (body, border, aux)."""
    truth = labels_from_rles(rles)
    out = {}
    for name, v in VARIANTS.items():
        body, border, aux = probs[(v["model"], v["tta"])]
        labels = np.zeros(truth.shape, np.int32)
        if gate_sources(aux, {"vit": vit_prob})[v["gate"]] >= v["gate_thr"]:
            labels = remove_small(segment_instances(body.astype(np.float32), border.astype(np.float32),
                                                    v["mask_thr"]), v["min_area"])
        out[name] = image_f2(truth, labels)
    return image_id, out


def validation_probs(models, image_dir, ids, cfg, device):
    loader = DataLoader(TestDataset(image_dir, ids), batch_size=8, num_workers=cfg["workers"], pin_memory=True,
                        prefetch_factor=4)
    k = 0
    with torch.no_grad():
        for x in loader:
            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            out = {}
            with torch.autocast(device.type, dtype=torch.float16):
                for name, model in models.items():
                    seg4, cls4, seg8, cls8 = predict_dihedral(model, x)
                    out[(name, "flip4")] = (seg4.half().cpu().numpy(), cls4.float().cpu().numpy())
                    out[(name, "dihedral8")] = (seg8.half().cpu().numpy(), cls8.float().cpu().numpy())
            for i in range(len(x)):
                yield ids[k], {key: (s[i, 0], s[i, 1], float(c[i])) for key, (s, c) in out.items()}
                k += 1


def spearman(a, b):
    return float(pd.Series(a).rank().corr(pd.Series(b).rank()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../out_scene")
    ap.add_argument("--data-dir", default="../data")
    ap.add_argument("--old-weights", default="../weights/unet_v2.pt")
    ap.add_argument("--new-weights", default="../out_small/unet_small_ft.pt")
    ap.add_argument("--gate-probs", default="../out_diag_parallel/vit_gate_probs.csv")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data = Path(args.data_dir)

    # --- 1-2. scenes ---
    train_names = sorted(os.listdir(data / "train_v2"))
    test_names = sorted(os.listdir(data / "test_v2"))
    paths = [str(data / "train_v2" / n) for n in train_names] + [str(data / "test_v2" / n) for n in test_names]
    log(f"hashing {len(paths)} images")
    with Pool(max(1, (os.cpu_count() or 4) - 2)) as pool:
        hashes = pool.map(block_hashes, paths, chunksize=128)
    names = train_names + test_names
    is_test = np.array([False] * len(train_names) + [True] * len(test_names))
    scene, links = group_scenes(names, hashes)
    log(f"{links} block links, {len(set(scene))} scenes")

    cfg = {**CONFIG, "data_dir": args.data_dir, "workers": 8, "postproc_workers": 12}
    ids, rles_by_id = load_masks(cfg["data_dir"], 0)
    _, _, valid = split_ids(ids, rles_by_id, cfg["valid_images"], cfg["seed"])
    valid_set = set(valid)
    split = np.where(is_test, "test", np.where([n in valid_set for n in names], "valid", "train"))
    df = pd.DataFrame({"ImageId": names, "scene": scene, "split": split})
    train_scenes = set(df.loc[df.split == "train", "scene"])
    df["touches_train"] = df.scene.isin(train_scenes) & (df.split != "train")
    # A scene of size 1 is an image with no detected overlap at all.
    df["scene_size"] = df.groupby("scene").ImageId.transform("size")
    df.to_csv(out / "scenes.csv", index=False)
    v, t = df[df.split == "valid"], df[df.split == "test"]
    has_ship = v.ImageId.map(lambda i: i in rles_by_id)
    leak_stats = {
        "valid_images": len(v), "valid_leaky": int(v.touches_train.sum()),
        "valid_clean": int((~v.touches_train).sum()),
        "valid_clean_with_ships": int((~v.touches_train & has_ship).sum()),
        "valid_leaky_with_ships": int((v.touches_train & has_ship).sum()),
        "test_images": len(t), "test_touching_train": int(t.touches_train.sum()),
        "train_images_in_multi_tile_scenes": float((df[df.split == "train"].scene_size > 1).mean()),
    }
    log("leakage: " + json.dumps(leak_stats))

    # --- 4. re-score the leaderboard variants ---
    device = torch.device("cuda")
    models = {}
    for name, path in [("old", args.old_weights), ("new", args.new_weights)]:
        m = ShipNet(cfg["encoder"], None).to(device).to(memory_format=torch.channels_last).eval()
        m.load_state_dict(torch.load(path, map_location=device))
        models[name] = m
    gate = pd.read_csv(args.gate_probs)
    vit = dict(zip(gate.ImageId, gate.vit_prob))
    pool, n_workers = make_pool(cfg)
    items = ((i, probs, rles_by_id.get(i, []), vit[i])
             for i, probs in validation_probs(models, data / "train_v2", list(valid), cfg, device))
    rows = {i: s for i, s in parallel_stream(pool, n_workers, score_variants, items)}
    pool.shutdown()
    scores = pd.DataFrame.from_dict(rows, orient="index")
    scores["leaky"] = v.set_index("ImageId").loc[scores.index, "touches_train"].values
    scores.to_csv(out / "variant_scores_per_image.csv")

    subsets = {"all": scores, "clean": scores[~scores.leaky], "leaky": scores[scores.leaky]}
    table = {k: {name: float(s[name].mean()) for name in VARIANTS} for k, s in subsets.items()}
    lb_pub = [VARIANTS[n]["lb"][0] for n in VARIANTS]
    lb_priv = [VARIANTS[n]["lb"][1] for n in VARIANTS]
    rank_corr = {k: {"vs_public": spearman([table[k][n] for n in VARIANTS], lb_pub),
                     "vs_private": spearman([table[k][n] for n in VARIANTS], lb_priv)} for k in subsets}
    # Paired bootstrap on the clean subset for the key comparison (stage 2 vs the stage-4 model at the same gate).
    clean = subsets["clean"]
    rng = np.random.default_rng(0)
    diff = (clean["stage4_gate97"] - clean["stage2"]).to_numpy()
    boot = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(2000)] if len(diff) else [0.0]
    report = {"leakage": leak_stats, "mean_f2": table, "spearman_vs_leaderboard": rank_corr,
              "clean_stage4_gate97_minus_stage2": {"mean": float(diff.mean()) if len(diff) else None,
                                                   "ci95": [float(np.percentile(boot, 2.5)),
                                                            float(np.percentile(boot, 97.5))]},
              "variants": {n: {k: v for k, v in d.items()} for n, d in VARIANTS.items()}}
    (out / "leakage_report.json").write_text(json.dumps(report, indent=1))
    log("mean F2: " + json.dumps({k: {n: round(x, 5) for n, x in d.items()} for k, d in table.items()}))
    log("spearman vs leaderboard: " + json.dumps(rank_corr))
    log("clean stage4_gate97 - stage2: " + json.dumps(report["clean_stage4_gate97_minus_stage2"]))


if __name__ == "__main__":
    main()
