#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FIGURES OF THE FIXED-GEOMETRY MODELS: one per family, 800-case level.

For each family, runs inspect_cases.py in listing mode for that family's
fixed-geometry model (cached in PROJECT_DIR/listing_<prefix>_<family>.txt),
picks the TYPICAL validation case (relL2 and peak error both close to the
median, true peak >= 50 MPa when possible) and has inspect_cases.py save its
figures as PROJECT_DIR/figs/<prefix>_<family>_case<i>.png. Same criterion as
figs_final.py. The checkpoint and dataset of each family are given by
patterns in which {fam} is replaced by the family letter.

    python figs_fixed.py
    python figs_fixed.py --families L,U --ckpt-pattern "deeponet_fixo800_{fam}_e600.pt"
"""

import os
import argparse
import sys
import subprocess

from figs_final import (PROJECT_DIR, INSPECT, inspect_options, listing_rows,
                        choose_typical)

DEFAULT_FAMILIES = "L,Z,U,T,O,G"
DEFAULT_CKPT_PATTERN = "deeponet_fixo800_{fam}_e600.pt"
DEFAULT_DATASET_PATTERN = "fixo_{fam}"
DEFAULT_PREFIX = "fixed800"
LEGACY_PREFIX = "fixo800"     # listing name of earlier runs: lista_fixo800_<family>.txt


def run_listing(fam, ckpt, dataset, prefix):
    """Path of the listing of one family; runs it only if no listing of an
    earlier run exists."""
    log = os.path.join(PROJECT_DIR, f"listing_{prefix}_{fam}.txt")
    legacy = os.path.join(PROJECT_DIR, f"lista_{LEGACY_PREFIX}_{fam}.txt")
    if (not os.path.exists(log) and prefix == DEFAULT_PREFIX
            and os.path.exists(legacy)):
        return legacy
    if not os.path.exists(log):
        with open(log, "w") as fh:
            subprocess.run(
                [sys.executable, "-u", INSPECT, "--ckpt", ckpt,
                 "--dataset", dataset, inspect_options()["list"]],
                stdout=fh, stderr=subprocess.DEVNULL, cwd=PROJECT_DIR,
                check=True)
    return log


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--families", default=DEFAULT_FAMILIES,
                    help="comma-separated family letters")
    ap.add_argument("--ckpt-pattern", default=DEFAULT_CKPT_PATTERN,
                    help="checkpoint of each family, relative to PROJECT_DIR")
    ap.add_argument("--dataset-pattern", default=DEFAULT_DATASET_PATTERN,
                    help="dataset of each family, relative to PROJECT_DIR")
    ap.add_argument("--prefix", default=DEFAULT_PREFIX,
                    help="name of the listings and of the figures")
    args = ap.parse_args()

    os.makedirs(os.path.join(PROJECT_DIR, "figs"), exist_ok=True)
    opt = inspect_options()
    for fam in filter(None, (f.strip().upper()
                             for f in args.families.split(","))):
        ckpt = os.path.join(PROJECT_DIR, os.path.expanduser(
            args.ckpt_pattern.format(fam=fam)))
        ds = os.path.join(PROJECT_DIR, os.path.expanduser(
            args.dataset_pattern.format(fam=fam)))
        v = [r[1:] for r in listing_rows(run_listing(fam, ckpt, ds,
                                                     args.prefix))]
        if not v:
            print(f"{fam}: no validation rows — skipping")
            continue
        # the fixed geometry may have low stress: choose_typical then drops
        # the 50 MPa threshold
        (rel, case_i, peak, err), med_rel = choose_typical(v)

        out_path = os.path.join(PROJECT_DIR, "figs",
                                f"{args.prefix}_{fam}_case{case_i}.png")
        subprocess.run(
            [sys.executable, INSPECT, "--ckpt", ckpt, "--dataset", ds,
             "--split", "val", opt["case"], str(case_i),
             opt["save"], out_path],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            cwd=PROJECT_DIR, check=True)
        print(f"{fam}: case {case_i:4d} | relL2 {rel:.3f} (med {med_rel:.3f}) | "
              f"peak {peak:7.1f} MPa | error {err:+6.1f}% -> {out_path}")


if __name__ == "__main__":
    main()
