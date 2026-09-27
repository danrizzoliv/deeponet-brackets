#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FIGURES OF THE MULTI-FAMILY MODEL — one per family.

Same criterion as figs_fixed.py: runs --lista, picks the TYPICAL case of each
family (relL2 and peak error close to the median of that family, peak
>= 50 MPa) and saves the two three-panel figures.

    python figs_final.py
    python figs_final.py --ckpt deeponet_6fam_e2000.pt
"""

import os
import argparse
import subprocess
import numpy as np

FAMILIES = ["L", "Z", "U", "T", "O", "G"]
BASE = os.path.expanduser("~/projetos/tcc_brackets")


def run_listing(ckpt, dataset, log):
    if not os.path.exists(log):
        with open(log, "w") as fh:
            subprocess.run(
                ["python", "-u", os.path.join(BASE, "inspect_cases.py"),
                 "--ckpt", ckpt, "--dataset", dataset, "--lista"],
                stdout=fh, stderr=subprocess.DEVNULL, cwd=BASE, check=True)
    by_fam = {f: [] for f in FAMILIES}
    for ln in open(log):
        p = ln.split()
        if len(p) >= 10 and p[0] == "val" and p[2] in by_fam:
            try:
                by_fam[p[2]].append(
                    (float(p[6]), int(p[1]), float(p[7]), float(p[9])))
            except ValueError:
                pass
    return by_fam


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="deeponet_6fam_e2000.pt")
    ap.add_argument("--dataset", default="dataset_v3_p2")
    ap.add_argument("--prefixo", default="final")
    args = ap.parse_args()

    ckpt = os.path.join(BASE, args.ckpt)
    ds = os.path.join(BASE, args.dataset)
    log = os.path.join(BASE, f"lista_{args.prefixo}.txt")
    os.makedirs(os.path.join(BASE, "figs"), exist_ok=True)

    by_fam = run_listing(ckpt, ds, log)
    for fam in FAMILIES:
        v = by_fam[fam]
        if not v:
            print(f"{fam}: no validation cases — skipping")
            continue
        med_rel = np.median([r[0] for r in v])
        med_peak = np.median([abs(r[3]) for r in v])
        cand = [r for r in v if abs(r[0] - med_rel) < 0.04 and r[2] >= 50.0]
        if not cand:
            cand = [r for r in v if abs(r[0] - med_rel) < 0.04] or v
        cand.sort(key=lambda r: abs(abs(r[3]) - med_peak))
        rel, case_i, peak, err = cand[0]

        out_path = os.path.join(BASE, "figs", f"{args.prefixo}_{fam}.png")
        subprocess.run(
            ["python", os.path.join(BASE, "inspect_cases.py"),
             "--ckpt", ckpt, "--dataset", ds,
             "--split", "val", "--caso", str(case_i), "--salvar", out_path],
            stdout=subprocess.DEVNULL, cwd=BASE, check=True)
        print(f"{fam}: case {case_i:5d} | relL2 {rel:.3f} (med {med_rel:.3f}) | "
              f"peak {peak:7.1f} MPa | error {err:+6.1f}%")
        print(f"     {out_path[:-4]}_setup.png and {out_path[:-4]}_result.png")


if __name__ == "__main__":
    main()
