#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluates an already trained checkpoint (old or new format) on its own
validation split, case by case, and writes the per-case CSV read by
compare_models.py. This lets a model trained earlier (e.g. the delivered
model) serve as one arm of a paired comparison without retraining it.

    python evaluate_checkpoint.py --ckpt deeponet_6fam_55k.pt \\
        --dataset ~/projetos/tcc_brackets/dataset_v4_treino \\
        --csv delivered_val_errors.csv

The validation split is the one stored in the checkpoint, and the relL2 is
the same as in train_deeponet.py (norm of the error over norm of the field,
per case). A model trained with --edge-distance needs concave_edges.pkl in
the dataset folder.
"""

import os
import argparse

from bracket_dataset import FAMILIES, load_edge_map
from train_deeponet import (load_model, checkpoint_split, eval_field,
                            family_report)


def main():
    ap = argparse.ArgumentParser(
        description="Per-case validation error of a trained checkpoint.")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", required=True,
                    help="dataset folder (with samples/sample_*.npz)")
    ap.add_argument("--csv", required=True,
                    help="output CSV (file,family,relL2)")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = ap.parse_args()

    model, ckpt, cfg = load_model(args.ckpt, args.device)
    sdir = os.path.join(os.path.expanduser(args.dataset), "samples")
    val = [os.path.join(sdir, os.path.basename(f))
           for f in checkpoint_split(ckpt)["val"]]
    missing = [f for f in val if not os.path.exists(f)]
    if missing:
        raise SystemExit(f"{len(missing)} validation cases are not in "
                         f"{sdir}, e.g.: {missing[:3]}")
    edge_map = None
    if cfg.get("edge_distance"):
        edge_map, _ = load_edge_map(val)

    print(f"evaluating {len(val)} validation cases...")
    det = []
    m, md = eval_field(model, val, ckpt["stats"], cfg, args.device,
                       edge_map=edge_map, details=det)
    print(f"relL2 VAL: mean={m:.3f} median={md:.3f}")
    family_report(det, "VAL")
    with open(args.csv, "w") as fh:
        fh.write("file,family,relL2\n")
        for name, fam, e in det:
            fh.write(f"{name},{FAMILIES[fam]},{e:.6f}\n")
    print(f"written to {args.csv}")


if __name__ == "__main__":
    main()
