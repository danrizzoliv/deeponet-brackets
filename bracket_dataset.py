#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DATASET — SIMPLE PARAMETRIC BRACKETS (L/Z/U).

Classical DeepONet, back to basics:
  BRANCH (14): [unit 6D load (F, M/l_ref)/s (6),
                standardized logE and nu (2),
                one-hot family (3),
                standardized dimensions A,B,C,W,T (5)] -> 16 numbers
  TRUNK (3): normalized xyz (GLOBAL c0/scale from training). No Fourier,
             no SDF — plain coordinates.
  TARGET: standardized log(vm/s + eps) (essential data conditioning —
          without the log the MSE stalls, as shown in phase 1 of the project).

Each sample has ITS OWN mesh (geometry varies): coords and targets are
preloaded per sample (target in float16 to save RAM).
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

N_FAM = 12


def list_samples(sample_dir):
    return sorted(glob.glob(os.path.join(sample_dir, "sample_*.npz")))


def split_files(files, val_frac=0.15, seed=0):
    rng = np.random.default_rng(seed)
    files = list(files)
    perm = rng.permutation(len(files))
    n_val = max(1, int(round(val_frac * len(files))))
    val_idx = set(perm[:n_val].tolist())
    return ([f for i, f in enumerate(files) if i not in val_idx],
            [f for i, f in enumerate(files) if i in val_idx])


def load_raw(path):
    d = np.load(path, allow_pickle=True)
    F = np.asarray(d["loads"], np.float64).reshape(-1)
    M = np.asarray(d["moments"], np.float64).reshape(-1)
    L = float(d["l_ref"])
    load6 = np.concatenate([F, M / L])
    s = float(np.linalg.norm(load6))
    return d, load6, s


def compute_stats(train_files, target="log"):
    """TRAINING set only. Target by subsampling (constant memory);
    global normalizations of material, dimensions and coords."""
    rng = np.random.default_rng(0)
    t_sub, gE, nus, params = [], [], [], []
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    for f in train_files:
        d, load6, s = load_raw(f)
        vm = np.asarray(d["von_mises"], np.float64)
        t = np.clip(vm / s, 0, None)
        m = min(t.size, 512)
        t_sub.append(rng.choice(t, size=m, replace=False))
        gE.append(np.log(float(d["E"])))
        nus.append(float(d["nu"]))
        params.append(np.asarray(d["geo_params"], np.float64))
        nodes = np.asarray(d["nodes"], np.float64)
        lo = np.minimum(lo, nodes.min(0))
        hi = np.maximum(hi, nodes.max(0))
    t_all = np.concatenate(t_sub)
    gE, nus = np.array(gE), np.array(nus)
    params = np.array(params)
    stats = {"c0": [float(v) for v in 0.5 * (lo + hi)],
             "scale": float(0.5 * (hi - lo).max() + 1e-30),
             "target_transform": target,
             "mat_mean": [float(gE.mean()), float(nus.mean())],
             "mat_std": [float(max(gE.std(), 1e-12)),
                         float(max(nus.std(), 1e-12))],
             "par_mean": params.mean(0).tolist(),
             "par_std": np.maximum(params.std(0), 1e-6).tolist(),
             "n_fam": N_FAM,
             "branch_dim": 6 + 2 + N_FAM + params.shape[1]}
    if target == "log":
        t_eps = max(0.01 * float(np.median(t_all)), 1e-12)
        g = np.log(t_all + t_eps)
        stats.update(t_eps=t_eps, t_mean=float(g.mean()),
                     t_std=float(max(g.std(), 1e-12)))
    else:
        stats.update(t_eps=0.0, t_mean=float(t_all.mean()),
                     t_std=float(max(t_all.std(), 1e-12)))
    return stats


def normalize_vm(vm, s, stats):
    t = np.clip(vm / s, 0.0, None)
    if stats.get("target_transform", "linear") == "log":
        t = np.log(t + stats["t_eps"])
    return (t - stats["t_mean"]) / stats["t_std"]


def denormalize_vm(y, s, stats):
    t = y * stats["t_std"] + stats["t_mean"]
    if stats.get("target_transform", "linear") == "log":
        t = np.exp(t) - stats["t_eps"]
    return np.clip(t, 0, None) * s


def case_branch(path, stats):
    """Branch (16,), s, vm(N,), nodes(N,3)."""
    d, load6, s = load_raw(path)
    vm = np.asarray(d["von_mises"], np.float32)
    nodes = np.asarray(d["nodes"], np.float64)
    b = [(load6 / s).astype(np.float32)]
    mm, ms = stats["mat_mean"], stats["mat_std"]
    b.append(np.array([(np.log(float(d["E"])) - mm[0]) / ms[0],
                       (float(d["nu"]) - mm[1]) / ms[1]], np.float32))
    n_fam = int(stats.get("n_fam", stats["branch_dim"] - 19))
    onehot = np.zeros(n_fam, np.float32)
    onehot[int(d["familia"])] = 1.0
    b.append(onehot)
    pm = np.asarray(stats["par_mean"])
    ps = np.asarray(stats["par_std"])
    b.append(((np.asarray(d["geo_params"], np.float64) - pm) / ps
              ).astype(np.float32))
    return np.concatenate(b), s, vm, nodes, d


def hole_distance(nodes, holes):
    """Smallest distance (>=0) from each node to the edge of any hole in
    the set. Holes are cylinders [cx,cy,cz, ax,ay,az, r, half]."""
    holes = np.asarray(holes, np.float64).reshape(-1, 8)
    dmin = np.full(nodes.shape[0], np.inf)
    for h in holes:
        c, axis, r, half = h[:3], h[3:6], float(h[6]), float(h[7])
        axis = axis / (np.linalg.norm(axis) + 1e-30)
        v = nodes - c
        proj = v @ axis
        radial = np.linalg.norm(v - np.outer(proj, axis), axis=1)
        dr = np.maximum(radial - r, 0.0)            # afastamento radial
        da = np.maximum(np.abs(proj) - half, 0.0)   # afastamento axial
        dmin = np.minimum(dmin, np.hypot(dr, da))
    return dmin


def trunk_feats(nodes, d, stats):
    """(N,5): normalized xyz + distances (on the global scale) to the nearest
    fixation hole and load hole. Stress concentration is a SMOOTH function
    of these distances — a physical feature in place of frequency."""
    xyz = norm_coords(nodes, stats)
    s = stats["scale"]
    df = hole_distance(nodes, d["furos_fix"]) / s
    dl = hole_distance(nodes, d["furos_carga"]) / s
    return np.concatenate([xyz, df[:, None].astype(np.float32),
                           dl[:, None].astype(np.float32)], axis=1)


def norm_coords(nodes, stats):
    c0 = np.asarray(stats["c0"], np.float64)
    return ((nodes - c0) / stats["scale"]).astype(np.float32)


class SimpleDataset(Dataset):
    """Preloads branch + coords + target (float16) for each sample;
    fresh node sampling on every visit; `repeats` for more steps/epoch.
    `hole_frac` > 0 oversamples the neighborhood of the holes (importance
    sampling): that fraction of the n_query points is drawn only among
    nodes within 0.12*scale of some hole — where the stress peaks live,
    which uniform sampling almost never visits."""

    NEAR_RADIUS = 0.12   # hole neighborhood, as a fraction of the global scale

    def __init__(self, files, stats, n_query=1024, repeats=1, dist=False,
                 hole_frac=0.0):
        self.n_query = n_query
        self.repeats = max(1, int(repeats))
        self.hole_frac = float(hole_frac)
        self.items = []
        for f in files:
            b, s, vm, nodes, d = case_branch(f, stats)
            ft = trunk_feats(nodes, d, stats)
            near = np.flatnonzero(ft[:, 3:5].min(axis=1) < self.NEAR_RADIUS)
            if not dist:
                ft = ft[:, :3]
            xyz = torch.from_numpy(np.ascontiguousarray(ft))
            y = torch.from_numpy(normalize_vm(vm, s, stats).astype(np.float16))
            self.items.append((torch.from_numpy(b), xyz, y,
                               torch.from_numpy(near.astype(np.int64))))

    def __len__(self):
        return len(self.items) * self.repeats

    def __getitem__(self, i):
        b, xyz, y, near = self.items[i % len(self.items)]
        n_near = int(round(self.hole_frac * self.n_query)) \
            if near.numel() > 0 else 0
        idx = torch.randint(0, xyz.shape[0], (self.n_query - n_near,))
        if n_near:
            idx_near = near[torch.randint(0, near.numel(), (n_near,))]
            idx = torch.cat([idx, idx_near])
        return b, xyz[idx], y[idx].float()
