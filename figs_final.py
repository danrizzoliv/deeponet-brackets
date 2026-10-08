#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FIGURES OF THE MULTI-FAMILY MODEL: one per family.

Same criterion as figs_fixed.py: runs inspect_cases.py in listing mode
(cached in PROJECT_DIR/listing_<prefix>.txt), picks the TYPICAL validation
case of each family (relL2 and peak error close to the median of that
family, true peak >= 50 MPa) and has inspect_cases.py save its two
three-panel figures, <prefix>_<family>_setup.png and _result.png, in
PROJECT_DIR/figs. Families without validation cases are skipped.

    python figs_final.py
    python figs_final.py --ckpt deeponet_6fam_55k.pt --dataset dataset_v4_p2 --prefix final55k
"""

import os
import argparse
import sys
import subprocess
import numpy as np

from bracket_dataset import FAMILIES

PROJECT_DIR = os.path.expanduser("~/projetos/tcc_brackets")  # author's project folder; adjust
# the inspection script of this repository, next to this file
INSPECT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "inspect_cases.py")


def inspect_options():
    """Option names of inspect_cases.py."""
    return {"list": "--listing", "case": "--case", "save": "--save"}


def listing_rows(log):
    """Validation rows of a listing, in file order:
    (family, relL2, case, true peak, peak error %). Columns of a listing
    line: split, case, family, |F|, |M|, nu, relL2, true peak, predicted
    peak, peak error %."""
    rows = []
    with open(log, encoding="utf-8", errors="replace") as fh:
        for ln in fh:
            p = ln.split()
            if len(p) >= 10 and p[0] == "val":
                try:
                    rows.append((p[2], float(p[6]), int(p[1]), float(p[7]),
                                 float(p[9])))
                except ValueError:
                    pass
    return rows


def parse_listing(log, families=FAMILIES):
    """Validation rows of a listing, by family: (relL2, case, peak, error%)."""
    by_fam = {f: [] for f in families}
    for fam, *row in listing_rows(log):
        if fam in by_fam:
            by_fam[fam].append(tuple(row))
    return by_fam


def run_listing(ckpt, dataset, prefix):
    """Path of the listing of `ckpt` on `dataset`; runs it only if no
    listing of an earlier run exists (old name: lista_<prefix>.txt)."""
    log = os.path.join(PROJECT_DIR, f"listing_{prefix}.txt")
    legacy = os.path.join(PROJECT_DIR, f"lista_{prefix}.txt")
    if not os.path.exists(log) and os.path.exists(legacy):
        return legacy
    if not os.path.exists(log):
        with open(log, "w") as fh:
            subprocess.run(
                [sys.executable, "-u", INSPECT,
                 "--ckpt", ckpt, "--dataset", dataset,
                 inspect_options()["list"]],
                stdout=fh, stderr=subprocess.DEVNULL, cwd=PROJECT_DIR,
                check=True)
    return log


def choose_typical(rows):
    """The typical case: relL2 within 0.04 of the median, true peak
    >= 50 MPa, and the absolute peak error closest to its median."""
    med_rel = np.median([r[0] for r in rows])
    med_peak = np.median([abs(r[3]) for r in rows])
    cand = [r for r in rows if abs(r[0] - med_rel) < 0.04 and r[2] >= 50.0]
    if not cand:
        cand = [r for r in rows if abs(r[0] - med_rel) < 0.04] or rows
    cand.sort(key=lambda r: abs(abs(r[3]) - med_peak))
    return cand[0], med_rel


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ckpt", default="deeponet_6fam_e2000.pt",
                    help="checkpoint, relative to PROJECT_DIR")
    ap.add_argument("--dataset", default="dataset_v3_p2",
                    help="dataset, relative to PROJECT_DIR")
    ap.add_argument("--prefix", default="final",
                    help="name of the listing and of the figures")
    ap.add_argument("--prefixo", dest="prefix", help=argparse.SUPPRESS)
    args = ap.parse_args()

    ckpt = os.path.join(PROJECT_DIR, os.path.expanduser(args.ckpt))
    ds = os.path.join(PROJECT_DIR, os.path.expanduser(args.dataset))
    os.makedirs(os.path.join(PROJECT_DIR, "figs"), exist_ok=True)
    opt = inspect_options()

    by_fam = parse_listing(run_listing(ckpt, ds, args.prefix))
    for fam in FAMILIES:
        v = by_fam[fam]
        if not v:
            continue                  # family absent from this dataset
        (rel, case_i, peak, err), med_rel = choose_typical(v)

        out_path = os.path.join(PROJECT_DIR, "figs",
                                f"{args.prefix}_{fam}.png")
        subprocess.run(
            [sys.executable, INSPECT, "--ckpt", ckpt, "--dataset", ds,
             "--split", "val", opt["case"], str(case_i),
             opt["save"], out_path],
            stdout=subprocess.DEVNULL, cwd=PROJECT_DIR, check=True)
        print(f"{fam}: case {case_i:5d} | relL2 {rel:.3f} (med {med_rel:.3f}) | "
              f"peak {peak:7.1f} MPa | error {err:+6.1f}%")
        print(f"     {out_path[:-4]}_setup.png and {out_path[:-4]}_result.png")


if __name__ == "__main__":
    main()
