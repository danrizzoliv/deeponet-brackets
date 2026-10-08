#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Paired comparison of two models, case by case, on the same validation cases.

    python compare_models.py delivered_val_errors.csv revised_val_errors.csv

The per-case CSVs are written by train_deeponet.py at the end of training
("<out>_val_errors.csv", columns file,family,relL2) or by
evaluate_checkpoint.py; CSVs written before the translation
("<out>_erros_val.csv", columns arquivo,familia,relL2) are read as well.
Only cases present in both files enter (same validation split). By family:
mean relL2 of A and B, relative gain, fraction of cases in which B is better
and the p-value of the paired Wilcoxon signed-rank test (needs scipy;
otherwise nan).
"""

import sys
import csv
import numpy as np

# column names: current, and those written before the translation
FILE_COLS = ("file", "arquivo")
FAMILY_COLS = ("family", "familia")


def read_errors(path):
    """{file name: (family, relL2)} from a per-case CSV in either format."""
    with open(path) as fh:
        reader = csv.DictReader(fh)
        cols = reader.fieldnames or []
        fcol = next((c for c in FILE_COLS if c in cols), None)
        famcol = next((c for c in FAMILY_COLS if c in cols), None)
        if fcol is None or famcol is None or "relL2" not in cols:
            raise SystemExit(f"{path}: expected columns file,family,relL2 "
                             f"(or arquivo,familia,relL2), found {cols}")
        return {r[fcol]: (r[famcol], float(r["relL2"])) for r in reader}


def p_value(diff):
    try:
        from scipy.stats import wilcoxon
        return float(wilcoxon(diff).pvalue) if np.any(diff != 0) else 1.0
    except ImportError:
        return float("nan")


def main():
    if any(a in ("-h", "--help") for a in sys.argv[1:]):
        print(__doc__.strip())
        return
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    a, b = read_errors(sys.argv[1]), read_errors(sys.argv[2])
    common = sorted(set(a) & set(b))
    if not common:
        raise SystemExit("no case in common: the splits are different")
    if len(common) < max(len(a), len(b)):
        print(f"[warning] {len(common)} cases in common out of {len(a)} "
              f"and {len(b)}")
    fam = np.array([a[k][0] for k in common])
    ea = np.array([a[k][1] for k in common])
    eb = np.array([b[k][1] for k in common])

    print(f"A = {sys.argv[1]}\nB = {sys.argv[2]}\n")
    print(f"{'fam':5s} {'n':>5s} {'A':>7s} {'B':>7s} {'gain':>8s} "
          f"{'B better':>9s} {'p':>8s}")
    groups = [(f, fam == f) for f in dict.fromkeys(fam)] + \
             [("ALL", np.ones(len(fam), bool))]
    for name, m in groups:
        xa, xb = ea[m], eb[m]
        gain = 100 * (xa.mean() - xb.mean()) / xa.mean()
        print(f"{name:5s} {m.sum():5d} {xa.mean():7.3f} {xb.mean():7.3f} "
              f"{gain:+7.1f}% {100*np.mean(xb < xa):8.0f}% "
              f"{p_value(xa - xb):8.2g}")
    print("\ngain > 0: model B errs less")


if __name__ == "__main__":
    main()
