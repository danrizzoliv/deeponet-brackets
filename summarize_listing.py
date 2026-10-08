#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Per-family table of the validation cases of an inspect_cases.py --listing
output: mean relL2, median |peak error|, median signed peak error (bias) and
number of cases, per family and over all; then, for each family, the TYPICAL
case (relL2 within 0.03 of the family median, true peak >= 50 MPa, |peak
error| closest to the family median) with its --case index.

Reads listings of the current inspect_cases.py and of the old
inspecionar_simples.py alike: the case lines have the same columns
(split case fam |F| |M| nu relL2 true_peak pred_peak err%), only the
training rows are labelled "train" now and "treino" before. Header, summary
and any other line are ignored.

    python inspect_cases.py --ckpt ... --dataset ... --listing > listing.txt
    python summarize_listing.py listing.txt
"""

import sys
import collections
import numpy as np

from bracket_dataset import FAMILIES

SPLIT_ALIASES = {"treino": "train"}   # split name in listings written before the translation


def read_listing(path):
    """{split: {family: [(relL2, case, true_peak, peak_err%), ...]}} from a
    listing in the current or in the old format."""
    rows = collections.defaultdict(lambda: collections.defaultdict(list))
    with open(path, encoding="utf-8", errors="replace") as fh:
        for ln in fh:
            p = ln.split()
            if len(p) < 10:
                continue
            split = SPLIT_ALIASES.get(p[0], p[0])
            if split not in ("train", "val"):
                continue
            try:
                rows[split][p[2]].append(
                    (float(p[6]), int(p[1]), float(p[7]), float(p[9])))
            except ValueError:
                pass
    return rows


def main():
    if any(a in ("-h", "--help") for a in sys.argv[1:]):
        print(__doc__.strip())
        return
    if len(sys.argv) != 2:
        raise SystemExit("usage: python summarize_listing.py LISTING.txt")
    rows = read_listing(sys.argv[1])["val"]
    fams = [f for f in FAMILIES if rows[f]]

    print(f"{'fam':4s} {'mean relL2':>10s} {'med |peak|':>11s} {'bias':>8s} "
          f"{'n':>6s}")
    all_rows = []
    for fam in fams:
        v = rows[fam]
        all_rows += v
        r = np.array([x[0] for x in v]); q = np.array([x[3] for x in v])
        print(f"{fam:4s} {r.mean():10.3f} {np.median(np.abs(q)):10.1f}% "
              f"{np.median(q):+7.1f}% {len(v):6d}")
    if all_rows:
        r = np.array([x[0] for x in all_rows])
        q = np.array([x[3] for x in all_rows])
        print(f"{'ALL':4s} {r.mean():10.3f} {np.median(np.abs(q)):10.1f}% "
              f"{np.median(q):+7.1f}% {len(all_rows):6d}\n")
    for fam in fams:
        v = rows[fam]
        mr = np.median([x[0] for x in v]); mp = np.median([abs(x[3]) for x in v])
        c = [x for x in v if abs(x[0] - mr) < 0.03 and x[2] >= 50.0]
        c.sort(key=lambda x: abs(abs(x[3]) - mp))
        if c:
            print(f"# {fam}: --case {c[0][1]}  relL2 {c[0][0]:.3f}  "
                  f"peak {c[0][2]:.1f} MPa  error {c[0][3]:+.1f}%")


if __name__ == "__main__":
    main()
