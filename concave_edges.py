#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reconstructs, for every case of a dataset, the concave edges where the
fillet lies, and writes them next to the dataset as concave_edges.pkl.

The edges are not stored in the .npz (only the fillet radius, in
geo_params[5]), but they are produced by sample_geometry() of the
generator, which depends only on the random generator --- and the generator
is seeded by the case index. Drawing again recovers the edges without
meshing or solving anything.

The function is extracted from the source of generate_dataset.py without
importing it, since the generator imports FEniCSx at the top while the
sampling is pure NumPy; the script therefore runs anywhere, Colab included.

The reconstruction is VERIFIED: the draw also returns the eleven geometric
parameters, which are stored in the .npz; if they match, the seed and the
edges are right. RF (geo_params[5]) and RB (geo_params[6]) are left out of
the comparison, because the generator reduces them after the draw when the
OCC fillet fails (transactional retries RB, RB/2, RB/4...). Neither moves the
concave edges: RF is the fillet radius on them and RB rounds the convex
outer edges.

    python concave_edges.py --dataset dataset_v4
    python concave_edges.py --dataset fixed_L400 --search-fixed-geometry
    python concave_edges.py --dataset fixed_L400 --fixed-geometry 1234

Writes {dataset}/concave_edges.pkl: {file name: array (n, 3)}, one row per
edge (axis, c1, c2), axis 0 = x, 1 = y, 2 = z (see
bracket_dataset.edge_distance).
"""

import os
import re
import ast
import glob
import pickle
import argparse
import numpy as np

# kept local (not imported from bracket_dataset) so that this script needs
# NumPy only
FAMILIES = ["L", "Z", "U", "T", "O", "G", "E", "F", "X", "J", "W", "S"]
KEY_FAMILY = "familia"            # .npz field name of the archived data

GENERATOR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "generate_dataset.py")
SAMPLER_NAMES = ("sample_geometry", "sortear_geometria")   # current, original
AXIS = {"x": 0, "y": 1, "z": 2}

# Relaxed check, by family: indices of geo_params on which the concave
# edges depend, used only when the full check fails. Z: edges (T, T) and
# (0, B - T) depend on B and T alone; A and W, drawn before them, are also
# required as proof that the seed is right. Reason: the block
# sample_004000..005999 of the dataset came from an earlier revision of the
# generator, which drew C, DF and DL of the Z differently but A, B, W and T
# identically.
RELAXED = {1: [0, 1, 3, 4]}


def load_sampler(path=GENERATOR):
    """Extracts the geometry sampler from the generator without running its
    imports."""
    src = open(path).read()
    fn = next(n for n in ast.parse(src).body
              if isinstance(n, ast.FunctionDef) and n.name in SAMPLER_NAMES)
    ns = {"np": np, "FAMILIES": FAMILIES, "FAMILIAS": FAMILIES}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), path, "exec"), ns)
    return ns[fn.name]


def normalize_edges(edges, W):
    """Converts the two edge formats of the generator into rows
    (axis, c1, c2), with the convention of build_mesh:
        'y': line along y, with x = c1 and z = c2
        'x': line along x, with y = c1 and z = c2
        'z': line along z, with x = c1 and y = c2
    The short format (x0, z0) is a 'y' line."""
    out = []
    for e in edges:
        if len(e) == 2:
            e = ("y", e[0], e[1], 0.5 * W)
        axis, c1, c2, _ = e
        out.append((AXIS[axis], float(c1), float(c2)))
    return np.array(out, np.float32).reshape(-1, 3)


def case_index(name):
    return int(re.search(r"(\d+)", os.path.basename(name)).group(1))


def check(sampler, d, rng, fam):
    """Draws again and compares the parameters with the stored ones.
    Returns (ok, edges, W, relaxed)."""
    _, params, _, edges, _, _ = sampler(rng, fam)
    g = np.asarray(d["geo_params"], np.float64)
    p = np.asarray(params, np.float64)
    mask = np.ones(len(g), bool)
    mask[[5, 6]] = False                 # RF and RB may have been reduced
    ok = np.allclose(p[mask], g[mask], rtol=1e-4, atol=1e-3)
    relaxed = False
    if not ok and fam in RELAXED:
        idx = RELAXED[fam]
        ok = relaxed = bool(np.allclose(p[idx], g[idx], rtol=1e-4, atol=1e-3))
    return ok, edges, float(p[3]), relaxed            # p[3] = W


def search_fixed_geometry(sampler, d, fam, limit):
    """The fixed-geometry datasets do not record the seed of the geometry,
    but all their cases share it: find the seed that reproduces one."""
    for s in range(limit):
        if check(sampler, d, np.random.default_rng(s), fam)[0]:
            return s
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seed", type=int, default=0,
                    help="base seed of the generation (varying geometry)")
    ap.add_argument("--fixed-geometry", "--geo-fixa", type=int, default=None,
                    dest="fixed_geometry", metavar="SEED",
                    help="geometry seed of a fixed-geometry dataset")
    ap.add_argument("--search-fixed-geometry", "--buscar-geo-fixa",
                    action="store_true", dest="search",
                    help="find the geometry seed automatically")
    ap.add_argument("--search-limit", type=int, default=200000)
    ap.add_argument("--generator", default=GENERATOR)
    args = ap.parse_args()

    sampler = load_sampler(args.generator)
    src = os.path.join(os.path.expanduser(args.dataset), "samples")
    files = sorted(glob.glob(os.path.join(src, "*.npz")))
    files = [f for f in files if os.path.basename(f).startswith("sample_")]
    if not files:
        raise SystemExit(f"no sample_*.npz in {src}")
    print(f"{len(files)} cases in {src}")

    fixed = args.fixed_geometry
    if args.search:
        d0 = np.load(files[0], allow_pickle=True)
        fam0 = int(d0[KEY_FAMILY])
        print(f"searching the fixed-geometry seed (family {FAMILIES[fam0]})...")
        fixed = search_fixed_geometry(sampler, d0, fam0, args.search_limit)
        if fixed is None:
            raise SystemExit("seed not found; raise --search-limit or give "
                             "--fixed-geometry")
        print(f"fixed geometry seed = {fixed}")

    out, bad, relaxed = {}, [], []
    for k, f in enumerate(files):
        d = np.load(f, allow_pickle=True)
        fam = int(d[KEY_FAMILY])
        seed = fixed if fixed is not None else args.seed + case_index(f)
        ok, edges, W, rel = check(sampler, d, np.random.default_rng(seed),
                                  fam)
        name = os.path.basename(f)
        if not ok:
            bad.append(name)
            continue
        if rel:
            relaxed.append(name)
        out[name] = normalize_edges(edges, W)
        if (k + 1) % 5000 == 0:
            print(f"  {k + 1}/{len(files)}")

    print(f"\nreconstructed and verified: {len(out)} of {len(files)} "
          f"({100 * len(out) / len(files):.2f}%)")
    if relaxed:
        print(f"  of which {len(relaxed)} by the relaxed check (only the "
              f"parameters that define the edges), e.g. {relaxed[:3]}")
    if bad:
        print(f"NOT matched: {len(bad)}, e.g. {bad[:5]}")
        print("If many, those cases were generated with another base seed; "
              "if few, they are isolated and can be left out.")
    counts = {}
    for v in out.values():
        counts[len(v)] = counts.get(len(v), 0) + 1
    print("edges per case:", dict(sorted(counts.items())))

    path = os.path.join(os.path.expanduser(args.dataset), "concave_edges.pkl")
    with open(path, "wb") as fh:
        pickle.dump(out, fh)
    print(f"written to {path}")


if __name__ == "__main__":
    main()
