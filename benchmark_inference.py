#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BENCHMARK — DeepONet evaluation time against the solver.

Measures how long the model takes to predict the full von Mises field of a
case, under the same conditions in which the solver was timed: CPU,
two threads, one case at a time.

    python benchmark_inference.py --ckpt deeponet_6fam_e2000.pt \\
        --dataset ~/projetos/tcc_brackets/dataset_v3_p2 --n 50
"""

import os
import time
import argparse
import numpy as np
import torch

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
torch.set_num_threads(2)

from bracket_dataset import case_branch, trunk_feats, denormalize_vm
from train_deeponet import SimpleDeepONet, predict_full


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--n", type=int, default=50,
                    help="how many validation cases to time")
    ap.add_argument("--aquecimento", type=int, default=5)
    args = ap.parse_args()

    c = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    a = c["args"]
    model = SimpleDeepONet(
        branch_dim=c["stats"]["branch_dim"], p=a.get("p", 128),
        hidden=a.get("hidden", 128), layers=a.get("layers", 4),
        fourier=a.get("fourier", 0),
        d_extra=2 if a.get("dist_furos") else 0,
        decoder=a.get("decoder", "dot"), trunk=a.get("trunk", "tanh"))
    model.load_state_dict(c["model"])
    model.eval()
    use_dist = bool(model.d_extra)
    stats = c["stats"]
    ds = os.path.join(args.dataset, "samples")
    cases = c["split"]["val"][: args.n + args.aquecimento]

    # ---- separates the cost of reading the file from the cost of inference ----
    t_prep, t_inf, n_nodes = [], [], []
    for i, fb in enumerate(cases):
        p = os.path.join(ds, fb)
        t0 = time.perf_counter()
        b, s, vm, nodes, d = case_branch(p, stats)
        ft = trunk_feats(nodes, d, stats)
        if not use_dist:
            ft = ft[:, :3]
        t1 = time.perf_counter()
        y = predict_full(model, b, ft, "cpu")
        _ = np.clip(denormalize_vm(y, s, stats), 0, None)
        t2 = time.perf_counter()
        if i >= args.aquecimento:           # discards the warm-up
            t_prep.append(t1 - t0)
            t_inf.append(t2 - t1)
            n_nodes.append(nodes.shape[0])

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
    print(f"\nP2 solver measured at 22.7 s/case -> "
          f"speed-up of {22.7e3/tot.mean():.0f}x (total) and "
          f"{22.7e3/inf.mean():.0f}x (inference only)")


if __name__ == "__main__":
    main()
