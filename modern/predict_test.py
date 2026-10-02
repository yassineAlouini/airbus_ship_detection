"""Write a test submission for one U-Net checkpoint with explicit post-processing settings.

Useful for ablations, e.g. a new model with an old gate threshold, without re-running training or validation.

usage: python predict_test.py WEIGHTS OUT_CSV [--gate mean_aux_vit] [--gate-thr 0.97] [--mask-thr 0.5]
                              [--min-area 20] [--scale 1.0]
"""

import argparse
from pathlib import Path

import pandas as pd
import torch

from airbus_modern import CONFIG, ShipNet
from diagnose import log, make_pool, parallel_stream
from small_ships import MODES, SMALL_CONFIG, mode_probs, submission_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("weights")
    ap.add_argument("out_csv")
    ap.add_argument("--gate", default="mean_aux_vit", choices=["aux", "vit", "mean_aux_vit"])
    ap.add_argument("--gate-thr", type=float, default=0.97)
    ap.add_argument("--mask-thr", type=float, default=0.5)
    ap.add_argument("--min-area", type=int, default=20)
    ap.add_argument("--scale", type=float, default=1.0, help="input scale (1.0, 1.5 or 2.0)")
    ap.add_argument("--data-dir", default="../data")
    ap.add_argument("--gate-probs", default=SMALL_CONFIG["gate_probs"])
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    cfg = {**CONFIG, **SMALL_CONFIG, "data_dir": args.data_dir, "workers": args.workers, "postproc_workers": 12}
    mode = f"single_s{args.scale:g}"
    MODES[mode] = [("model", args.scale)]
    best = {"mode": mode, "gate": args.gate, "gate_thr": args.gate_thr, "mask_thr": args.mask_thr,
            "min_area": args.min_area}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ShipNet(cfg["encoder"], None).to(device).to(memory_format=torch.channels_last).eval()
    model.load_state_dict(torch.load(args.weights, map_location=device))
    gate = pd.read_csv(args.gate_probs)
    vit = dict(zip(gate.ImageId, gate.vit_prob))
    test_dir = Path(cfg["data_dir"]) / "test_v2"
    test_ids = sorted(pd.read_csv(Path(cfg["data_dir"]) / "sample_submission_v2.csv").ImageId)

    pool, n_workers = make_pool(cfg)
    log(f"test inference: {args.weights} {best}")
    items = ((image_id, *per_mode[mode], vit[image_id], best)
             for image_id, per_mode in mode_probs({"model": model}, test_dir, test_ids, cfg, device, [mode]))
    rows = [r for rs in parallel_stream(pool, n_workers, submission_rows, items) for r in rs]
    pool.shutdown()
    sub = pd.DataFrame(rows, columns=["ImageId", "EncodedPixels"])
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(args.out_csv, index=False)
    n_ship = sub.groupby("ImageId").EncodedPixels.apply(lambda s: (s != "").any()).mean()
    log(f"submission: {sub.ImageId.nunique()} images, {len(sub)} rows, {n_ship:.3f} with ships")


# The guard matters: post-processing workers are spawned and re-import this module.
if __name__ == "__main__":
    main()
