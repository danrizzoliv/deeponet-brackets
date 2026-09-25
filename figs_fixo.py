#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FIGURES OF THE LOCKED GEOMETRY — one per family, 800 level.

For each family: runs --lista for the fixo_800 model, picks the TYPICAL case
(relL2 and peak error both close to the median, peak >= 50 MPa) and saves the
6-panel figure. Same criterion as the figures of the final model.

    python figs_fixo.py
"""

import os
import subprocess
import numpy as np

FAMILIES = ["L", "Z", "U", "T", "O", "G"]
BASE = os.path.expanduser("~/projetos/tcc_brackets")


def run_listing(fam):
    """Runs --lista and returns the validation rows: (relL2, case, peak, error%)."""
    log = os.path.join(BASE, f"lista_fixo800_{fam}.txt")
    if not os.path.exists(log):
        with open(log, "w") as fh:
            subprocess.run(
                ["python", "-u", os.path.join(BASE, "inspecionar_simples.py"),
                 "--ckpt", os.path.join(BASE, f"deeponet_fixo800_{fam}_e600.pt"),
                 "--dataset", os.path.join(BASE, f"fixo_{fam}"), "--lista"],
                stdout=fh, stderr=subprocess.DEVNULL, cwd=BASE, check=True)
    rows = []
    for ln in open(log):
        p = ln.split()
        if len(p) >= 10 and p[0] == "val":
            try:
                rows.append((float(p[6]), int(p[1]), float(p[7]), float(p[9])))
            except ValueError:
                pass
    return rows


def main():
    os.makedirs(os.path.join(BASE, "figs"), exist_ok=True)
    for fam in FAMILIES:
        v = run_listing(fam)
        if not v:
            print(f"{fam}: no validation rows — skipping")
            continue
        med_rel = np.median([r[0] for r in v])
        med_peak = np.median([abs(r[3]) for r in v])
        cand = [r for r in v if abs(r[0] - med_rel) < 0.04 and r[2] >= 50.0]
        if not cand:                       # fixed geometry may have low stress
            cand = [r for r in v if abs(r[0] - med_rel) < 0.04] or v
        cand.sort(key=lambda r: abs(abs(r[3]) - med_peak))
        rel, case_i, peak, err = cand[0]

        out_path = os.path.join(BASE, "figs", f"fixo800_{fam}_caso{case_i}.png")
        subprocess.run(
            ["python", os.path.join(BASE, "inspecionar_simples.py"),
             "--ckpt", os.path.join(BASE, f"deeponet_fixo800_{fam}_e600.pt"),
             "--dataset", os.path.join(BASE, f"fixo_{fam}"),
             "--split", "val", "--caso", str(case_i), "--salvar", out_path],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            cwd=BASE, check=True)
        print(f"{fam}: case {case_i:4d} | relL2 {rel:.3f} (med {med_rel:.3f}) | "
              f"peak {peak:7.1f} MPa | error {err:+6.1f}% -> {out_path}")


if __name__ == "__main__":
    main()
