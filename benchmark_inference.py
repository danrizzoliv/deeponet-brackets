#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inference time of the DeepONet, against the measured time of the solver.

Measures how long the model takes to predict the full von Mises field of a
validation case under the conditions in which the solver was timed: CPU, two
threads, one case at a time. Reading the file and building the inputs
(branch, trunk features) is timed apart from the inference proper; the first
--warmup cases are discarded. The solver is not run: its time per case
(SOLVER_SECONDS, measured for the P2 solver) is a constant. Works with checkpoints of the old and of the current training
code, including models trained with --edge-distance and --hole-proximity.

    python benchmark_inference.py --ckpt deeponet_6fam_e2000.pt \\
        --dataset ~/projetos/tcc_brackets/dataset_v3_p2 --n 50

Old option name --aquecimento (now --warmup) is still accepted.
"""

import os
import time
import glob
import argparse
import numpy as np
import torch

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
torch.set_num_threads(2)

from bracket_dataset import denormalize_vm, load_edge_map
from train_deeponet import (load_model, checkpoint_split, case_features,
                            predict_full)

SOLVER_SECONDS = 22.7   # measured P2 solver time per case (s)


def main():
    ap = argparse.ArgumentParser(
        description="Time the DeepONet inference per case.")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--n", type=int, default=50,
                    help="how many validation cases to time")
    ap.add_argument("--warmup", type=int, default=5,
                    help="cases evaluated first and not timed")
    ap.add_argument("--aquecimento", dest="warmup", type=int,
                    help=argparse.SUPPRESS)       # old name
    args = ap.parse_args()

    model, ckpt, cfg = load_model(args.ckpt, "cpu")
    stats = ckpt["stats"]
    ds = os.path.join(args.dataset, "samples")
    cases = checkpoint_split(ckpt)["val"][: args.n + args.warmup]
    edge_map = None
    if cfg.get("edge_distance"):     # read once, outside the timing
        edge_map, _ = load_edge_map(sorted(glob.glob(os.path.join(ds,
                                                                  "*.npz"))))

    # ---- separates the cost of reading the file from that of inference ----
    t_prep, t_inf, n_nodes = [], [], []
    for i, fb in enumerate(cases):
        p = os.path.join(ds, fb)
        t0 = time.perf_counter()
        b, ft, s, vm, nodes, d = case_features(p, stats, cfg, edge_map)
        t1 = time.perf_counter()
        y = predict_full(model, b, ft, "cpu")
        _ = np.clip(denormalize_vm(y, s, stats), 0, None)
        t2 = time.perf_counter()
        if i >= args.warmup:           # discards the warm-up
            t_prep.append(t1 - t0)
            t_inf.append(t2 - t1)
            n_nodes.append(nodes.shape[0])
    if not t_inf:
        raise SystemExit(f"no case timed: the validation split has "
                         f"{len(cases)} cases and --warmup is {args.warmup}")

    prep = np.array(t_prep) * 1e3
    inf = np.array(t_inf) * 1e3
    tot = prep + inf
    print(f"\n{len(inf)} cases, {np.mean(n_nodes):.0f} nodes on average\n")
    print(f"{'':22s} {'mean':>9s} {'median':>9s} {'p90':>9s}")
    for label, v in (("read + encoding", prep), ("inference", inf),
                     ("total", tot)):
        print(f"{label:22s} {v.mean():8.2f}ms {np.median(v):8.2f}ms "
              f"{np.percentile(v,90):8.2f}ms")
    print(f"\nper node: {1e3*inf.mean()/np.mean(n_nodes):.2f} us")
    print(f"\nP2 solver measured at {SOLVER_SECONDS} s/case -> "
          f"speed-up of {SOLVER_SECONDS*1e3/tot.mean():.0f}x (total) and "
          f"{SOLVER_SECONDS*1e3/inf.mean():.0f}x (inference only)")


if __name__ == "__main__":
    main()
