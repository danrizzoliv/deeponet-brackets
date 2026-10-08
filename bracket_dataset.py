#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Input encodings and the PyTorch dataset of the bracket DeepONet.

BRANCH (31 numbers), one vector per case:
    unit load (F, M / l_ref) / s          6   s = |(F, M / l_ref)|
    standardized log E and nu             2
    one-hot family                       12   six used in the dataset
    standardized geometric parameters    11   [A, B, C, W, T, RF, RB, DF, DL, NF, NL]

TRUNK, one vector per query point:
    normalized coordinates                3   always
    distance to the nearest fixation
      hole and to the nearest load hole   2   --hole-distances
    distance to the nearest concave edge  1   --edge-distance
    proximity r / (r + d) to the nearest
      fixation hole and load hole         2   --hole-proximity

TARGET: standardized log(sigma_vm / s + eps). The load magnitude s is divided
out and restored analytically, which is exact in linear elasticity; the log
puts every stress level on the same footing.

Every case has its own mesh, so coordinates and targets are preloaded per
case (targets in float16 to save memory) and a fresh set of query points is
drawn on every visit.

Field names of the .npz files (`familia`, `furos_fix`, `furos_carga`) are
kept as written in the 55,803 archived cases; they are read through the
constants below.
"""

import os
import glob
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset

FAMILIES = ["L", "Z", "U", "T", "O", "G", "E", "F", "X", "J", "W", "S"]
NUM_FAMILIES = len(FAMILIES)

# .npz field names, fixed by the archived data (Portuguese in the files)
KEY_FAMILY = "familia"            # family index
KEY_FIXATION_HOLES = "furos_fix"  # one row per hole: [cx,cy,cz, ax,ay,az, r, half]
KEY_LOAD_HOLES = "furos_carga"

EDGE_FAR = 2.0                    # normalized distance used when a case has no edge
NO_EDGES = np.empty((0, 3), np.float32)
EDGE_FILES = ("concave_edges.pkl", "fillet_edges.pkl")   # new name, old name


# ----------------------------------------------------------------- files

def list_samples(sample_dir):
    return sorted(glob.glob(os.path.join(sample_dir, "sample_*.npz")))


def split_files(files, val_frac=0.15, seed=0):
    """Fixed random split; the same (files, val_frac, seed) always gives the
    same partition, which is what makes checkpoints resumable."""
    rng = np.random.default_rng(seed)
    files = list(files)
    perm = rng.permutation(len(files))
    n_val = max(1, int(round(val_frac * len(files))))
    val_idx = set(perm[:n_val].tolist())
    return ([f for i, f in enumerate(files) if i not in val_idx],
            [f for i, f in enumerate(files) if i in val_idx])


def load_case(path):
    """Returns (data, load6, s): the .npz, the 6-component load
    (F, M / l_ref) and its magnitude."""
    d = np.load(path, allow_pickle=True)
    F = np.asarray(d["loads"], np.float64).reshape(-1)
    M = np.asarray(d["moments"], np.float64).reshape(-1)
    load6 = np.concatenate([F, M / float(d["l_ref"])])
    return d, load6, float(np.linalg.norm(load6))


# ---------------------------------------------------- normalization stats

def compute_stats(train_files, target="log"):
    """Normalization statistics, from the TRAINING files only. The target
    statistics use a subsample of 512 points per case (constant memory)."""
    rng = np.random.default_rng(0)
    t_sub, log_e, nus, params = [], [], [], []
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    for f in train_files:
        d, _, s = load_case(f)
        t = np.clip(np.asarray(d["von_mises"], np.float64) / s, 0, None)
        t_sub.append(rng.choice(t, size=min(t.size, 512), replace=False))
        log_e.append(np.log(float(d["E"])))
        nus.append(float(d["nu"]))
        params.append(np.asarray(d["geo_params"], np.float64))
        nodes = np.asarray(d["nodes"], np.float64)
        lo = np.minimum(lo, nodes.min(0))
        hi = np.maximum(hi, nodes.max(0))
    t_all = np.concatenate(t_sub)
    log_e, nus, params = np.array(log_e), np.array(nus), np.array(params)
    stats = {"c0": [float(v) for v in 0.5 * (lo + hi)],
             "scale": float(0.5 * (hi - lo).max() + 1e-30),
             "target_transform": target,
             "mat_mean": [float(log_e.mean()), float(nus.mean())],
             "mat_std": [float(max(log_e.std(), 1e-12)),
                         float(max(nus.std(), 1e-12))],
             "par_mean": params.mean(0).tolist(),
             "par_std": np.maximum(params.std(0), 1e-6).tolist(),
             "n_fam": NUM_FAMILIES,
             "branch_dim": 6 + 2 + NUM_FAMILIES + params.shape[1]}
    if target == "log":
        t_eps = max(0.01 * float(np.median(t_all)), 1e-12)
        g = np.log(t_all + t_eps)
        stats.update(t_eps=t_eps, t_mean=float(g.mean()),
                     t_std=float(max(g.std(), 1e-12)))
    else:
        stats.update(t_eps=0.0, t_mean=float(t_all.mean()),
                     t_std=float(max(t_all.std(), 1e-12)))
    return stats


def uses_hole_proximity(stats):
    """The proximity features travel inside the stats saved in the
    checkpoint (key `dist_rel` in checkpoints written before the
    translation)."""
    return bool(stats.get("hole_proximity", stats.get("dist_rel", False)))


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


def norm_coords(nodes, stats):
    c0 = np.asarray(stats["c0"], np.float64)
    return ((nodes - c0) / stats["scale"]).astype(np.float32)


# ------------------------------------------------------------ the branch

def case_branch(path, stats):
    """Returns (branch (31,), s, vm (N,), nodes (N, 3), data)."""
    d, load6, s = load_case(path)
    vm = np.asarray(d["von_mises"], np.float32)
    nodes = np.asarray(d["nodes"], np.float64)
    b = [(load6 / s).astype(np.float32)]
    mm, ms = stats["mat_mean"], stats["mat_std"]
    b.append(np.array([(np.log(float(d["E"])) - mm[0]) / ms[0],
                       (float(d["nu"]) - mm[1]) / ms[1]], np.float32))
    n_fam = int(stats.get("n_fam", stats["branch_dim"] - 19))
    onehot = np.zeros(n_fam, np.float32)
    onehot[int(d[KEY_FAMILY])] = 1.0
    b.append(onehot)
    pm, ps = np.asarray(stats["par_mean"]), np.asarray(stats["par_std"])
    b.append(((np.asarray(d["geo_params"], np.float64) - pm) / ps
              ).astype(np.float32))
    return np.concatenate(b), s, vm, nodes, d


# ------------------------------------------------------ trunk features

def _hole_offsets(nodes, h):
    """Radius of a cylindrical hole and the distance of each node to its
    surface (0 on the wall). A hole is [cx,cy,cz, ax,ay,az, r, half]."""
    c, axis, r, half = h[:3], h[3:6], float(h[6]), float(h[7])
    axis = axis / (np.linalg.norm(axis) + 1e-30)
    v = nodes - c
    along = v @ axis
    radial = np.linalg.norm(v - np.outer(along, axis), axis=1)
    return r, np.hypot(np.maximum(radial - r, 0.0),
                       np.maximum(np.abs(along) - half, 0.0))


def hole_distance(nodes, holes):
    """Distance (>= 0) from each node to the surface of the nearest hole."""
    dmin = np.full(nodes.shape[0], np.inf)
    for h in np.asarray(holes, np.float64).reshape(-1, 8):
        dmin = np.minimum(dmin, _hole_offsets(nodes, h)[1])
    return dmin


def hole_proximity(nodes, holes):
    """Proximity to the nearest hole, measured in that hole's own radius:
    max over holes of r / (r + d). Equal to 1 on the wall, 1/2 at one radius
    away. Stress around a hole decays with the distance in radii
    (Kirsch: sigma ~ (r / rho)^2, rho = r + d), which a distance divided by
    the global scale of the part cannot express. Bounded in (0, 1]."""
    q = np.zeros(nodes.shape[0])
    for h in np.asarray(holes, np.float64).reshape(-1, 8):
        r, dist = _hole_offsets(nodes, h)
        q = np.maximum(q, r / (r + dist + 1e-30))
    return q


def edge_distance(nodes, edges):
    """Distance from each node to the nearest concave edge (where the fillet
    lies). `edges` is (n, 3), one line per row (axis, c1, c2):
        axis 0: line along x, with y = c1, z = c2
        axis 1: line along y, with x = c1, z = c2
        axis 2: line along z, with x = c1, y = c2
    as written by concave_edges.py. Each edge is taken as an infinite line:
    exact in L, Z, U, T and O, whose edges cross the whole width; an
    approximation in G, whose rib edges are short."""
    nodes = np.asarray(nodes, np.float64)
    d = np.full(nodes.shape[0], np.inf)
    across = {0: (1, 2), 1: (0, 2), 2: (0, 1)}
    for axis, c1, c2 in np.asarray(edges, np.float64).reshape(-1, 3):
        a, b = across[int(axis)]
        d = np.minimum(d, np.hypot(nodes[:, a] - c1, nodes[:, b] - c2))
    return d


def trunk_features(nodes, d, stats, edges=None):
    """Trunk inputs of every node, in this column order:
        0-2  normalized xyz
        3-4  distances to the nearest fixation and load hole (global scale)
        5    distance to the nearest concave edge, only if `edges` is given
        last two: proximities to the nearest fixation and load hole, only
             if the stats say so (uses_hole_proximity)
    Callers keep only the first three columns for a model trained without
    hole distances. New columns are always appended, so models trained with
    fewer features remain valid."""
    xyz = norm_coords(nodes, stats)
    s = stats["scale"]
    cols = [xyz,
            (hole_distance(nodes, d[KEY_FIXATION_HOLES]) / s)[:, None],
            (hole_distance(nodes, d[KEY_LOAD_HOLES]) / s)[:, None]]
    if edges is not None:
        de = np.minimum(edge_distance(nodes, edges) / s, EDGE_FAR)
        cols.append(de[:, None])
    if uses_hole_proximity(stats):
        cols += [hole_proximity(nodes, d[KEY_FIXATION_HOLES])[:, None],
                 hole_proximity(nodes, d[KEY_LOAD_HOLES])[:, None]]
    return np.concatenate([c.astype(np.float32) for c in cols], axis=1)


def load_edge_map(files):
    """Reads the concave edges of the dataset that holds `files`, written by
    concave_edges.py. Returns (map, missing): file name -> edges, and how
    many of the files have no reconstructed edge."""
    if not files:
        return {}, 0
    base = os.path.dirname(os.path.dirname(os.path.abspath(files[0])))
    for name in EDGE_FILES:
        p = os.path.join(base, name)
        if os.path.exists(p):
            with open(p, "rb") as fh:
                edge_map = pickle.load(fh)
            missing = sum(1 for f in files
                          if os.path.basename(f) not in edge_map)
            return edge_map, missing
    raise SystemExit(f"no {EDGE_FILES[0]} in {base}; run first: "
                     f"python concave_edges.py --dataset {base}")


# ---------------------------------------------------------- the dataset

class BracketDataset(Dataset):
    """Preloads branch, trunk features and target of every case and draws
    `n_query` nodes afresh on every visit; `repeats` visits each case
    several times per epoch.

    `hole_fraction` > 0 oversamples the neighbourhood of the holes
    (importance sampling): that fraction of the query points is drawn only
    among nodes within NEAR_RADIUS of a hole, where the peaks are and where
    uniform sampling rarely lands."""

    NEAR_RADIUS = 0.12   # hole neighbourhood, as a fraction of the global scale

    def __init__(self, files, stats, n_query=1024, repeats=1,
                 hole_distances=False, hole_fraction=0.0, edge_map=None):
        self.n_query = n_query
        self.repeats = max(1, int(repeats))
        self.hole_fraction = float(hole_fraction)
        self.items = []
        for f in files:
            b, s, vm, nodes, d = case_branch(f, stats)
            edges = (None if edge_map is None
                     else edge_map.get(os.path.basename(f), NO_EDGES))
            ft = trunk_features(nodes, d, stats, edges=edges)
            near = np.flatnonzero(ft[:, 3:5].min(axis=1) < self.NEAR_RADIUS)
            if not hole_distances:
                ft = ft[:, :3]
            y = normalize_vm(vm, s, stats).astype(np.float16)
            self.items.append((torch.from_numpy(b),
                               torch.from_numpy(np.ascontiguousarray(ft)),
                               torch.from_numpy(y),
                               torch.from_numpy(near.astype(np.int64))))

    def __len__(self):
        return len(self.items) * self.repeats

    def __getitem__(self, i):
        b, x, y, near = self.items[i % len(self.items)]
        n_near = (int(round(self.hole_fraction * self.n_query))
                  if near.numel() > 0 else 0)
        idx = torch.randint(0, x.shape[0], (self.n_query - n_near,))
        if n_near:
            idx = torch.cat([idx, near[torch.randint(0, near.numel(),
                                                     (n_near,))]])
        return b, x[idx], y[idx].float()
