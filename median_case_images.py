#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figures of the median case: for each family, picks the validation case whose
relL2 is closest to the median of that family and renders the two
inspect_cases.py figures of it (setup and result, three panels each).

    python median_case_images.py --mode var     # varying geometry, one model
    python median_case_images.py --mode fixed   # fixed geometry, one model per family
    python median_case_images.py --mode var --dry-run   # only print the choice

The case is taken from the per-case validation CSV that train_deeponet.py
writes at the end of training (same relL2 as inspect_cases.py), so nothing is
re-evaluated to choose: only the chosen cases are rendered. Both CSV formats
are read: "<out>_val_errors.csv" with header file,family,relL2 (current) and
"<out>_erros_val.csv" with header arquivo,familia,relL2 (written before the
translation). Given either name, the other is tried when it does not exist.

Defaults match the author's existing files (adjust PROJECT_DIR / DOWNLOADS,
or override the options):
    var:   checkpoint DOWNLOADS/final.pt (--var-ckpt), CSV
           resultados_fillet/var55k/<ckpt name>_val_errors.csv or
           _erros_val.csv (--var-csv), dataset PROJECT_DIR/dataset_v4_treino
           (--var-dataset; compact, without mesh, hence --meshes)
    fixed: checkpoint DOWNLOADS/novo_{F}_s0.pt (--fixed-ckpt, a pattern where
           {F} is the family letter), CSV next to it, dataset
           PROJECT_DIR/fixo_{F} (--fixed-dataset)
Relative paths are taken from the current folder, as in the original script
(run it from PROJECT_DIR). Old option names (--modo with var|fixo, --saida,
--malhas) are still accepted.

Output: {out}/{mode}_{F}_{case}_setup.png and _result.png
"""

import os
import sys
import csv
import argparse
import subprocess
import numpy as np
import torch

from bracket_dataset import FAMILIES
from train_deeponet import checkpoint_split

PROJECT_DIR = os.path.expanduser("~/projetos/tcc_brackets")  # author's project folder; adjust
DOWNLOADS = "/mnt/c/Users/User/Downloads"   # Windows Downloads folder seen from WSL; adjust
RESULTS_DIR = "resultados_fillet"           # author's existing results folder (relative)
DATASET_FAMILIES = FAMILIES[:6]             # the six families present in the dataset
INSPECT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "inspect_cases.py")

MODE_ALIASES = {"fixo": "fixed"}            # value used before the translation
CSV_SUFFIXES = ("_val_errors.csv", "_erros_val.csv")   # current, old
# CSV column names: current, and as written before the translation
CSV_COLUMNS = (("file", "family", "relL2"), ("arquivo", "familia", "relL2"))


def csv_for(path):
    """`path`, or the same CSV under the other naming scheme if `path` does
    not exist (<out>_val_errors.csv <-> <out>_erros_val.csv)."""
    if os.path.exists(path):
        return path
    for a, b in (CSV_SUFFIXES, CSV_SUFFIXES[::-1]):
        if path.endswith(a):
            alt = path[: -len(a)] + b
            if os.path.exists(alt):
                return alt
    raise SystemExit(f"no per-case CSV {path} (nor its "
                     f"{'/'.join(CSV_SUFFIXES)} counterpart)")


def ckpt_csv(folder, ckpt):
    """Per-case CSV of checkpoint `ckpt` in `folder`, in either format."""
    stem = os.path.splitext(os.path.basename(ckpt))[0]
    return csv_for(os.path.join(folder, stem + CSV_SUFFIXES[0]))


def read_cases(path):
    """[(file name, family letter, relL2), ...] from a per-case CSV in the
    current or in the old format."""
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        for cols in CSV_COLUMNS:
            if set(cols) <= set(reader.fieldnames or ()):
                f_col, fam_col, e_col = cols
                return [(r[f_col], r[fam_col], float(r[e_col]))
                        for r in reader]
    raise SystemExit(f"{path}: unknown header {reader.fieldnames}")


def closest_to_median(cases, fam):
    """(file name, relL2, family median, n) of the case of family `fam`
    whose relL2 is closest to the median of the family."""
    sub = [c for c in cases if c[1] == fam]
    if not sub:
        raise SystemExit(f"family {fam} has no cases in the CSV")
    e = np.array([c[2] for c in sub])
    med = float(np.median(e))
    k = int(np.argmin(np.abs(e - med)))
    return sub[k][0], sub[k][2], med, len(sub)


def validation_index(ckpt, name):
    """Index of case `name` in the validation split of `ckpt` (what
    inspect_cases.py --case expects)."""
    ck = torch.load(os.path.expanduser(ckpt), map_location="cpu",
                    weights_only=False)
    val = [os.path.basename(f) for f in checkpoint_split(ck)["val"]]
    if name not in val:
        raise SystemExit(f"{name} is not in the validation split of {ckpt}: "
                         "the CSV and the checkpoint are not from the same "
                         "training run")
    return val.index(name)


def render_command(ckpt, dataset, idx, out, meshes=None):
    cmd = [sys.executable, INSPECT, "--ckpt", ckpt, "--dataset", dataset,
           "--split", "val", "--case", str(idx), "--save", out]
    if meshes:
        cmd += ["--meshes", meshes]
    return cmd


def mode_name(value):
    value = MODE_ALIASES.get(value, value)
    if value not in ("var", "fixed"):
        raise argparse.ArgumentTypeError(
            f"invalid mode {value!r} (choose var or fixed)")
    return value


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Render the median validation case of each family.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--mode", type=mode_name, metavar="{var,fixed}",
                    help="var: varying geometry (one model); fixed: fixed "
                         "geometry (one model per family)")
    ap.add_argument("--dir", default=DOWNLOADS,
                    help="folder of the .pt and CSV files downloaded from "
                         "Drive")
    ap.add_argument("--out", default=os.path.join(RESULTS_DIR, "images"),
                    help="output folder of the figures")
    ap.add_argument("--meshes",
                    default=os.path.join(PROJECT_DIR, "dataset_tcc_55803"),
                    help="full dataset (with mesh) of the varying geometry; "
                         "the training dataset is compact, without "
                         "connectivity")
    ap.add_argument("--families", nargs="+", default=DATASET_FAMILIES,
                    choices=FAMILIES, metavar="F")
    ap.add_argument("--var-ckpt", default="final.pt",
                    help="var mode: checkpoint, relative to --dir")
    ap.add_argument("--var-csv", default=None,
                    help="var mode: per-case CSV (default: "
                         f"{RESULTS_DIR}/var55k/<ckpt name>{CSV_SUFFIXES[0]} "
                         f"or {CSV_SUFFIXES[1]})")
    ap.add_argument("--var-dataset",
                    default=os.path.join(PROJECT_DIR, "dataset_v4_treino"),
                    help="var mode: dataset the model was trained on")
    ap.add_argument("--fixed-ckpt", default="novo_{F}_s0.pt",
                    help="fixed mode: checkpoint pattern, relative to --dir; "
                         "{F} is the family; its CSV is read next to it")
    ap.add_argument("--fixed-dataset",
                    default=os.path.join(PROJECT_DIR, "fixo_{F}"),
                    help="fixed mode: dataset pattern; {F} is the family")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the chosen cases and commands, render nothing")
    # names used before the translation
    ap.add_argument("--modo", dest="mode", type=mode_name,
                    help=argparse.SUPPRESS)
    ap.add_argument("--saida", dest="out", help=argparse.SUPPRESS)
    ap.add_argument("--malhas", dest="meshes", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if args.mode is None:
        ap.error("--mode is required (var or fixed)")
    return args


def make_plan(args):
    """[(family, ckpt, dataset, case file, relL2, median, n), ...]"""
    plan = []
    if args.mode == "var":
        ckpt = os.path.join(args.dir, args.var_ckpt)
        csv_path = (csv_for(args.var_csv) if args.var_csv
                    else ckpt_csv(os.path.join(RESULTS_DIR, "var55k"), ckpt))
        cases = read_cases(csv_path)
        for F in args.families:
            plan.append((F, ckpt, args.var_dataset,
                         *closest_to_median(cases, F)))
    else:
        for F in args.families:
            ckpt = os.path.join(args.dir, args.fixed_ckpt.format(F=F))
            cases = read_cases(ckpt_csv(args.dir, ckpt))
            plan.append((F, ckpt, args.fixed_dataset.format(F=F),
                         *closest_to_median(cases, F)))
    return plan


def main():
    args = parse_args()
    plan = make_plan(args)

    print(f"\n{'fam':4s} {'chosen case':22s} {'relL2':>7s} "
          f"{'median':>8s} {'n':>5s}")
    for F, _, _, name, e, med, n in plan:
        print(f"{F:4s} {name:22s} {e:7.3f} {med:8.3f} {n:5d}")
    print()

    if not args.dry_run:
        os.makedirs(args.out, exist_ok=True)
    for F, ckpt, ds, name, e, med, n in plan:
        idx = validation_index(ckpt, name)
        base = os.path.join(args.out,
                            f"{args.mode}_{F}_{os.path.splitext(name)[0]}.png")
        print(f"==> {F}: {name} (val #{idx}, relL2 {e:.3f})")
        cmd = render_command(ckpt, ds, idx, base,
                             args.meshes if args.mode == "var" else None)
        if args.dry_run:
            print("    " + " ".join(cmd))
        else:
            subprocess.run(cmd, check=True)

    if not args.dry_run:
        print(f"\n{2 * len(plan)} figures in {args.out}/ "
              f"({args.mode}_*_setup.png and {args.mode}_*_result.png)")


if __name__ == "__main__":
    main()
