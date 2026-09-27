#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COMPACT — version of the dataset containing only what TRAINING uses.

Each .npz stores the tetrahedron connectivity (`cells`), which is the largest
field in the file and serves only to render figures: training uses nodes,
stress and the case metadata. By removing `cells` and writing in float32, the
set typically shrinks by a factor of 3 to 5, which makes the upload to
Colab feasible.

The compact dataset does NOT work with inspect_cases.py nor for the
figures — for those, keep using the original.

    python compact_dataset.py --src dataset_v4_p2 --dst dataset_v4_treino
    python compact_dataset.py --src dataset_v4_p2 --dst d4 --jobs 4
"""

import os
import glob
import argparse
import numpy as np
from concurrent.futures import ProcessPoolExecutor

# everything case_branch and trunk_feats read
KEEP = ("nodes", "von_mises", "familia", "geo_params", "loads", "moments",
          "E", "nu", "furos_fix", "furos_carga", "load_centers", "n_nodes",
          "l_ref", "escala")


def compact(pair):
    src, dst = pair
    try:
        d = np.load(src, allow_pickle=True)
        out = {}
        for k in d.files:
            if k not in KEEP:
                continue
            v = d[k]
            if v.dtype == np.float64:
                v = v.astype(np.float32)
            elif v.dtype == np.int64 and k != "familia":
                v = v.astype(np.int32)
            out[k] = v
        np.savez_compressed(dst, **out)
        return os.path.getsize(src), os.path.getsize(dst)
    except Exception as e:
        print(f"  failed {os.path.basename(src)}: {e}")
        return 0, 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="source dataset")
    ap.add_argument("--dst", required=True, help="compact dataset to create")
    ap.add_argument("--jobs", type=int, default=2)
    args = ap.parse_args()

    src = os.path.join(os.path.expanduser(args.src), "samples")
    dst = os.path.join(os.path.expanduser(args.dst), "samples")
    os.makedirs(dst, exist_ok=True)
    files = sorted(glob.glob(os.path.join(src, "*.npz")))
    if not files:
        raise SystemExit(f"no .npz in {src}")

    # report of the fields of the first file, for checking
    d0 = np.load(files[0], allow_pickle=True)
    print(f"{len(files)} files\nfields in the original:")
    for k in d0.files:
        v = d0[k]
        tag = "kept" if k in KEEP else "DISCARDED"
        print(f"  {k:16s} {str(v.dtype):9s} {str(v.shape):16s} {tag}")
    missing = [k for k in KEEP if k not in d0.files]
    if missing:
        print(f"\n[warning] not present in the original: {missing}")

    pairs = [(a, os.path.join(dst, os.path.basename(a))) for a in files]
    tot_s = tot_d = 0
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        for i, (s, t) in enumerate(ex.map(compact, pairs, chunksize=32), 1):
            tot_s += s
            tot_d += t
            if i % 2000 == 0:
                print(f"  {i}/{len(pairs)}  ({tot_d/1e9:.1f} GB so far)")

    print(f"\nsource  {tot_s/1e9:6.2f} GB")
    print(f"target  {tot_d/1e9:6.2f} GB")
    if tot_d:
        print(f"reduction {tot_s/tot_d:.1f}x")
    print(f"\n{dst}")


if __name__ == "__main__":
    main()
