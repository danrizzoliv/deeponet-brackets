#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ABLATION ON P2 — reruns the architectural ladder on quadratic data.

The original ladder ran on linear elements (P1), before the convergence
study that rejected them. This script repeats the same variants on
300 cases of the U family in P2, with the same split and the same budget, so
that the conclusions do not depend on a reference with flattened peaks.

It also adds the SIREN trunk, whose original run was interrupted at
epoch 119 of 300. The current train_simples.py already implements the
initialization of Sitzmann et al. (2020).

    python ablacao_p2.py --src dataset_v4_treino --device cuda
    python ablacao_p2.py --src dataset_v4_treino --so base prod siren

Writes ablacao_p2_{variant}.log and, at the end, ablacao_p2_resumo.txt.
"""

import os
import re
import glob
import argparse
import subprocess
import numpy as np

# same variants as the original ladder (ablation table), plus the SIREN.
# All other options stay at the train_simples.py defaults, as in the
# original ladder: batch 16, repeats 4, n-query 1024, lr 1e-3.
VARIANTS = [
    ("base",    "Baseline (inner product)",      []),
    ("dist",    "+ hole distances",              ["--dist-furos"]),
    ("frac",    "+ importance sampling",         ["--dist-furos",
                                                  "--frac-furos", "0.3"]),
    ("fourier", "+ Fourier features (6)",        ["--fourier", "6"]),
    ("w256",    "Width 256, 500 epochs",         ["--hidden", "256",
                                                  "--epochs", "500"]),
    ("prod",    "+ fusion decoder",              ["--dist-furos",
                                                  "--decoder", "prod"]),
    ("siren",   "+ fusion decoder, SIREN trunk", ["--dist-furos",
                                                  "--decoder", "prod",
                                                  "--trunk", "siren"]),
]


def build_subset(src, dst, n, family_idx=(0, 2000)):
    """The U cases in P2 are indices 0-1999 of the v3 dataset (the block that
    came in by symlink from dataset_U2k_p2). Takes the first n and checks that
    they all belong to the same family --- if they do not, the assumption about
    the indices is wrong and the script stops instead of running on mixed
    data."""
    s = os.path.join(src, "samples")
    d = os.path.join(dst, "samples")
    os.makedirs(d, exist_ok=True)
    files = [os.path.join(s, f"sample_{i:06d}.npz")
            for i in range(family_idx[0], family_idx[1])]
    files = [a for a in files if os.path.exists(a)][:n]
    if len(files) < n:
        raise SystemExit(f"only {len(files)} cases found in {s}")
    fams = {int(np.load(a, allow_pickle=True)["familia"]) for a in files}
    if len(fams) != 1:
        raise SystemExit(f"the cases belong to different families: {fams}; "
                         f"the indices do not correspond to a single block")
    for a in files:
        link = os.path.join(d, os.path.basename(a))
        if not os.path.exists(link):
            os.symlink(os.path.abspath(a), link)
    print(f"{n} cases of family {fams.pop()} in {d}")


def result(log):
    """Extracts the final mean relL2 of training and validation."""
    tr = va = None
    for ln in open(log, errors="ignore"):
        m = re.search(r"TREINO:\s*méd=([\d.]+)", ln)
        if m:
            tr = float(m.group(1))
        m = re.search(r"VAL\s*:\s*méd=([\d.]+)", ln)
        if m:
            va = float(m.group(1))
    return tr, va


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="dataset_v4_treino",
                    help="P2 dataset from which to take the cases")
    ap.add_argument("--dst", default="ablacao_p2_U300")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--so", nargs="*", default=None,
                    help="runs only these variants (e.g.: base prod siren)")
    args = ap.parse_args()

    build_subset(args.src, args.dst, args.n)

    chosen = [v for v in VARIANTS if not args.so or v[0] in args.so]
    for name, label, extra in chosen:
        log = f"ablacao_p2_{name}.log"
        if os.path.exists(log) and result(log)[1] is not None:
            print(f"[{name}] already done, skipping")
            continue
        # --resume: if the machine restarts in the middle of a variant, it
        # continues from the last saved epoch instead of starting from scratch
        cmd = ["python", "-u", "train_simples.py",
               "--dataset", args.dst, "--device", args.device,
               "--epochs", str(args.epochs), "--final-eval", "0",
               "--save-every", "10", "--resume",
               "--out", f"ablacao_p2_{name}.pt"] + extra
        # the variant's --epochs (w256) overrides the default by coming later
        print(f"\n[{name}] {label}\n  {' '.join(cmd)}")
        with open(log, "a") as fh:
            subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT)
        tr, va = result(log)
        print(f"  training {tr}  validation {va}")

    rows = [f"{'variant':34s} {'training':>8s} {'validation':>10s}"]
    for name, label, _ in VARIANTS:
        log = f"ablacao_p2_{name}.log"
        if os.path.exists(log):
            tr, va = result(log)
            rows.append(f"{label:34s} {str(tr):>8s} {str(va):>10s}")
    txt = "\n".join(rows)
    print("\n" + txt)
    open("ablacao_p2_resumo.txt", "w").write(txt + "\n")


if __name__ == "__main__":
    main()
