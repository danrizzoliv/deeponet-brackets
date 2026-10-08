#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Error diagnosis: where, at which stress level and in which geometries a
trained model errs. It only evaluates an existing checkpoint (old or new
format) on its own validation split; it trains nothing.

    python diagnose_error.py --ckpt deeponet_6fam_55k.pt \\
        --dataset ~/projetos/tcc_brackets/dataset_v4_treino --n 2000
    python diagnose_error.py --ckpt revised.pt --dataset DATASET \\
        --k 1.5 --csv diagnosis.csv

Three questions, one table each, by family:

 1. REGION. Each node is assigned, in this order of priority, to:
      fixation: within k*DF of the surface of a fixation hole
      load    : within k*DL of the surface of a load hole
      edge    : within k*(RF+T) of a concave edge
      rest    : the remainder of the part
    For each region: fraction of the nodes, fraction of the SQUARED ERROR
    (which is what makes up the relL2) and the local relL2 (error of the
    region / field of the region). A region whose share of the error is
    much larger than its share of the nodes is where the model loses most.

 2. STRESS LEVEL. Fraction of the squared error in the 10% and the 1% most
    stressed nodes of each case, and the peak error (predicted max vs true
    max). If the error concentrates at the peaks, aligning the loss with
    the relL2 (weighting by stress) is the way forward. Also the error in
    the least-stressed half, to see what stress weighting costs there.

 3. GEOMETRY. Spearman correlation between the relL2 of each case and each
    geometric parameter (and the moment fraction of the load). Shows which
    dimensions, as they vary, worsen the error the most.

The edge region needs the concave edges of the dataset (concave_edges.pkl,
or the older fillet_edges.pkl), written by concave_edges.py. A per-case
CSV (--csv) is also written for later analysis.
"""

import os
import argparse
import numpy as np

from bracket_dataset import (FAMILIES, KEY_FAMILY, KEY_FIXATION_HOLES,
                             KEY_LOAD_HOLES, NO_EDGES, hole_distance,
                             edge_distance, load_edge_map)
from train_deeponet import load_model, predict_case, checkpoint_split

PARAMS = ["A", "B", "C", "W", "T", "RF", "RB", "DF", "DL", "NF", "NL"]
REGIONS = ["fixation", "load", "edge", "rest"]


def spearman(x, y):
    """Rank correlation (without scipy). Returns nan if x or y is constant."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


def analyse_case(model, ckpt, cfg, path, edge_map, k, device):
    name = os.path.basename(path)
    edges = edge_map.get(name, NO_EDGES)

    # prediction, with the same features the model was trained with
    vm, vp, nodes, d = predict_case(model, path, ckpt["stats"], cfg, device,
                                    edge_map)
    vm = vm.astype(np.float64)
    vp = np.clip(vp, 0, None).astype(np.float64)
    e2 = (vp - vm) ** 2
    tot_e2 = e2.sum() + 1e-30
    tot_v2 = (vm ** 2).sum() + 1e-30

    # regions (in mm, scaled by the case's own dimensions)
    g = np.asarray(d["geo_params"], np.float64)
    T, RF, DF, DL = g[4], g[5], g[7], g[8]
    d_fix = hole_distance(nodes, d[KEY_FIXATION_HOLES])
    d_load = hole_distance(nodes, d[KEY_LOAD_HOLES])
    d_edge = edge_distance(nodes, edges)
    free = np.ones(len(vm), bool)
    masks = {}
    for r, m in (("fixation", d_fix < k * DF), ("load", d_load < k * DL),
                 ("edge", d_edge < k * (RF + T))):
        masks[r] = m & free
        free &= ~m
    masks["rest"] = free

    out = {"file": name, "family": int(d[KEY_FAMILY]),
           "relL2": float(np.sqrt(tot_e2 / tot_v2))}
    for r, m in masks.items():
        out[f"nodes_{r}"] = float(m.mean())
        out[f"error_{r}"] = float(e2[m].sum() / tot_e2)
        out[f"rel_{r}"] = (float(np.sqrt(e2[m].sum() / ((vm[m] ** 2).sum()
                                                        + 1e-30)))
                           if m.any() else float("nan"))

    # stress level
    for q, tag in ((0.90, "top10"), (0.99, "top1")):
        m = vm >= np.quantile(vm, q)
        out[f"error_{tag}"] = float(e2[m].sum() / tot_e2)
    # LEAST-stressed half: the stress-weighted loss gives these points less
    # weight; this shows whether they got worse. Pointwise relative error
    # |pred - true| / true (median), and the relL2 restricted to them.
    m = vm <= np.median(vm)
    out["low_relpt"] = float(np.median(np.abs(vp[m] - vm[m])
                                       / (vm[m] + 1e-30)))
    out["low_relL2"] = float(np.sqrt(e2[m].sum()
                                     / ((vm[m] ** 2).sum() + 1e-30)))
    out["peak_pct"] = float(100 * (vp.max() - vm.max()) / (vm.max() + 1e-30))
    # where the true peak is
    ip = int(np.argmax(vm))
    out["peak_in"] = next(r for r in REGIONS if masks[r][ip])

    # geometry and load
    for i, p in enumerate(PARAMS[:len(g)]):
        out[p] = float(g[i])
    F = np.linalg.norm(np.asarray(d["loads"], float))
    M = np.linalg.norm(np.asarray(d["moments"], float)) / float(d["l_ref"])
    out["moment_fraction"] = float(M / (np.hypot(F, M) + 1e-30))
    return out


def region_table(rows, title):
    print(f"\n=== 1. REGION — {title} ===")
    print("fraction of nodes / fraction of squared error / local relL2")
    print(f"{'fam':5s} " + " ".join(f"{r:>20s}" for r in REGIONS))
    for name, group in rows:
        cells = []
        for r in REGIONS:
            n = np.mean([x[f"nodes_{r}"] for x in group])
            e = np.mean([x[f"error_{r}"] for x in group])
            rl = np.nanmean([x[f"rel_{r}"] for x in group])
            cells.append(f"{100*n:4.0f}% {100*e:4.0f}% {rl:6.3f}")
        print(f"{name:5s} " + " ".join(f"{c:>20s}" for c in cells))


def stress_table(rows):
    print("\n=== 2. STRESS LEVEL ===")
    print(f"{'fam':5s} {'relL2':>6s} {'err top 10%':>13s} {'err top 1%':>11s} "
          f"{'peak med':>9s} {'|peak| med':>11s} {'lowest 50%':>11s}"
          f"   true peak in")
    for name, group in rows:
        pk = np.array([x["peak_pct"] for x in group])
        where = {r: np.mean([x["peak_in"] == r for x in group])
                 for r in REGIONS}
        where_txt = " ".join(f"{r} {100*v:.0f}%" for r, v in where.items()
                             if v > 0)
        print(f"{name:5s} {np.mean([x['relL2'] for x in group]):6.3f} "
              f"{100*np.mean([x['error_top10'] for x in group]):12.0f}% "
              f"{100*np.mean([x['error_top1'] for x in group]):10.0f}% "
              f"{np.median(pk):+8.1f}% {np.median(np.abs(pk)):10.1f}% "
              f"{np.mean([x['low_relL2'] for x in group]):5.3f}/"
              f"{100*np.median([x['low_relpt'] for x in group]):4.1f}%"
              f"   {where_txt}")
    print("(peak med < 0: the model underestimates the peak)")
    print("(lowest 50%: relL2 on the 50% least-stressed nodes / median of "
          "the pointwise relative error on them)")


def geometry_table(rows):
    print("\n=== 3. GEOMETRY — Spearman(relL2, parameter), 3 largest |rho| ===")
    for name, group in rows:
        rl = [x["relL2"] for x in group]
        rhos = []
        for p in PARAMS + ["moment_fraction"]:
            if p in group[0]:
                r = spearman([x[p] for x in group], rl)
                if not np.isnan(r):
                    rhos.append((p, r))
        rhos.sort(key=lambda t: -abs(t[1]))
        print(f"{name:5s} " + "   ".join(f"{p} {r:+.2f}" for p, r in rhos[:3]))
    print("(rho > 0: the error grows with the parameter; |rho| < 0.1 is "
          "negligible)")


def main():
    ap = argparse.ArgumentParser(
        description="Per-region, per-stress-level and per-geometry error "
                    "of a trained checkpoint on its validation split.")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", required=True,
                    help="dataset folder (with samples/ and concave_edges.pkl)")
    ap.add_argument("--n", type=int, default=2000,
                    help="validation cases drawn at random (0 = all)")
    ap.add_argument("--k", type=float, default=1.0,
                    help="multiplier of the region radii")
    ap.add_argument("--csv", default=None, help="per-case CSV output")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    model, ckpt, cfg = load_model(args.ckpt, args.device)
    sdir = os.path.join(os.path.expanduser(args.dataset), "samples")
    val = [os.path.join(sdir, os.path.basename(f))
           for f in checkpoint_split(ckpt)["val"]]
    if args.n and args.n < len(val):
        rng = np.random.default_rng(args.seed)
        val = [val[i] for i in sorted(rng.choice(len(val), args.n,
                                                 replace=False))]
    edge_map, missing = load_edge_map(val)
    if missing:
        print(f"[warning] {missing} cases without an edge: their edge "
              "region is empty")

    print(f"analysing {len(val)} validation cases "
          f"(regions with k = {args.k})...")
    res = []
    for i, f in enumerate(val):
        res.append(analyse_case(model, ckpt, cfg, f, edge_map, args.k,
                                args.device))
        if (i + 1) % 250 == 0:
            print(f"  {i+1}/{len(val)}")

    fams = sorted({x["family"] for x in res})
    rows = [(FAMILIES[f], [x for x in res if x["family"] == f])
            for f in fams] + [("ALL", res)]
    region_table(rows, f"k = {args.k}")
    stress_table(rows)
    geometry_table(rows)

    if args.csv:
        keys = list(res[0].keys())
        with open(args.csv, "w") as fh:
            fh.write(",".join(keys) + "\n")
            for x in res:
                fh.write(",".join(str(x[c]) if c != "family"
                                  else FAMILIES[x[c]] for c in keys) + "\n")
        print(f"\nper-case results in {args.csv}")


if __name__ == "__main__":
    main()
