#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Read-only audit of a generated dataset: the p-refinement check, the
generation time and the number of generation processes that ran in
parallel. It only READS files: it writes, moves and deletes nothing.

    python verification_report.py p-refinement --root ~/projetos
    python verification_report.py hours --dataset DATASET [--logs 'logs/*.txt']
    python verification_report.py parallel --dataset DATASET

(The names used before the translation, tabela3, horas and paralelo, are
accepted as aliases, as are the options --raiz, --limite and --pausa.)

p-refinement
    generate_dataset.py, run with --check-degree G (old --conv-grau), solves
    each verification case a second time at the higher degree G and writes
    samples/conv_NNNNNN_gG.npz with the peak deviation (dpico, in %) and the
    relL2 (rel_l2) between the two solutions on the same mesh. This command
    finds every such file under --root (recursively), groups them by
    (degree, reference degree) and prints, per family: number of cases,
    max |peak deviation|, cases accepted under --threshold (%), the relL2
    range and the reference-degree solve time. That time is the difference
    between the timestamps of conv_NNNNNN and sample_NNNNNN, since the
    sample file is written just before the higher-degree solve and the conv
    file just after it.

hours
    The time per case is not stored in the .npz, but the modification time
    of each sample_NNNNNN.npz is the instant the case finished. Between two
    consecutive indices (i-1 and i) of the same process the difference is
    the time of case i; summing gives the total computation, even when
    several processes ran in parallel. Long intervals (pauses, machine
    switched off), longer than --pause seconds, are discarded. If terminal
    logs were saved, the time printed on each "OK ... (12.3s)" line is
    also summed (--logs).

parallel
    Compares the wall-clock time during which generation was active (union
    of the intervals, all files in timestamp order) with the sum of the
    per-case times of each process. A ratio of ~1 means one process at a
    time; ~2, two in parallel; and so on. Shown per week, because it may
    have changed along the way.

Caveats: every timing comes from file modification times, so it is only
meaningful in the ORIGINAL generation folder. In a copy (cp, tar, unzip,
download) the timestamps are those of the copy; `hours` warns when the
intervals look like it. Index-consecutive cases are assumed to come from
the same process, which holds when each process generated a contiguous
index block; at block boundaries one interval may be wrong, and it is
usually removed by --pause. The P3 time likewise needs both files
untouched since generation.
"""

import os
import re
import sys
import glob
import argparse
import numpy as np

from bracket_dataset import FAMILIES

# author's project folder; adjust
PROJECT_DIR = os.path.expanduser("~/projetos/tcc_brackets")

# field names of the archived conv_*.npz files
KEY_DEGREE = "grau"
KEY_REF_DEGREE = "grau_conv"
KEY_FAMILY = "familia"
KEY_PEAK_DEV = "dpico"
KEY_REL_L2 = "rel_l2"

# subcommand names used before the translation
LEGACY_COMMANDS = {"tabela3": "p-refinement", "horas": "hours",
                   "paralelo": "parallel"}


def p_refinement(root, threshold):
    files = sorted(glob.glob(os.path.join(root, "**", "conv_*_g*.npz"),
                             recursive=True))
    if not files:
        raise SystemExit(f"no conv_*.npz under {root}; try --root with the "
                         "folder where the verification was run")
    groups = {}
    for f in files:
        d = np.load(f, allow_pickle=True)
        key = (int(d[KEY_DEGREE]), int(d[KEY_REF_DEGREE]))
        groups.setdefault(key, []).append(
            (os.path.dirname(f), os.path.basename(f), int(d[KEY_FAMILY]),
             float(d[KEY_PEAK_DEV]), float(d[KEY_REL_L2])))
    for (g, gc), cases in sorted(groups.items()):
        print(f"\n=== P{g} against P{gc}: {len(cases)} cases ===")
        for p in sorted({c[0] for c in cases}):
            print(f"  folder: {p} ({sum(c[0] == p for c in cases)} cases)")
        print(f"\n{'fam':4s} {'cases':>5s} {'max|dpeak|':>11s} "
              f"{'accepted':>8s}   dpeak of each case (%)")
        tot = acc = 0
        for f in sorted({c[2] for c in cases}):
            sub = [c for c in cases if c[2] == f]
            dp = np.array([c[3] for c in sub])
            n_acc = int((np.abs(dp) <= threshold).sum())
            tot += len(sub)
            acc += n_acc
            print(f"{FAMILIES[f]:4s} {len(sub):5d} {np.abs(dp).max():10.1f}% "
                  f"{n_acc:4d}/{len(sub):<3d}   "
                  + " ".join(f"{x:+.1f}" for x in dp))
        print(f"\n{'fam':4s} {'relL2 min':>9s} {'median':>8s} {'max':>7s} "
              f"{f'P{gc} time (s)':>13s}")
        for f in sorted({c[2] for c in cases}):
            sub = [c for c in cases if c[2] == f]
            r = np.array([c[4] for c in sub])
            # reference-degree time: sample_ is written before that solve
            # and conv_ right after, so the timestamp difference is its time
            tp = []
            for c in sub:
                i = int(re.search(r"conv_(\d+)_", c[1]).group(1))
                sa = os.path.join(c[0], f"sample_{i:06d}.npz")
                if os.path.exists(sa):
                    tp.append(os.path.getmtime(os.path.join(c[0], c[1]))
                              - os.path.getmtime(sa))
            ttxt = (f"{min(tp):5.0f} to {max(tp):3.0f}" if tp else "   ---")
            print(f"{FAMILIES[f]:4s} {r.min():9.3f} {np.median(r):8.3f} "
                  f"{r.max():7.3f} {ttxt:>13s}")
        rl = np.array([c[4] for c in cases])
        print(f"{'TOT':4s} {tot:5d} {'':>11s} {acc:4d}/{tot:<3d}")
        print(f"relL2 P{g} x P{gc}: from {rl.min():.3f} to {rl.max():.3f} "
              f"(median {np.median(rl):.3f})")
        above = [c for c in cases if abs(c[3]) > threshold]
        if above:
            print(f"above {threshold}%: " + ", ".join(
                f"{FAMILIES[c[2]]} {c[3]:+.1f}%" for c in above))


def _sample_times(dataset):
    """Indices and modification times of the sample files, by index."""
    sdir = os.path.join(os.path.expanduser(dataset), "samples")
    files = glob.glob(os.path.join(sdir, "sample_*.npz"))
    if not files:
        raise SystemExit(f"no sample_*.npz in {sdir}")
    idx = np.array([int(re.search(r"(\d+)", os.path.basename(f)).group(1))
                    for f in files])
    mt = np.array([os.path.getmtime(f) for f in files])
    o = np.argsort(idx)
    return sdir, len(files), idx[o], mt[o]


def hours(dataset, pause, logs):
    sdir, n, idx, mt = _sample_times(dataset)
    dt = np.diff(mt)
    consecutive = (np.diff(idx) == 1) & (dt > 0) & (dt < pause)
    per_case = dt[consecutive]
    span = (mt.max() - mt.min()) / 3600
    print(f"\n{n} cases in {sdir}")
    print(f"first to last file: {span:.1f} h of calendar time "
          f"({span/24:.1f} days)")
    if len(per_case) < 0.5 * n or np.median(per_case) < 1.0:
        print("WARNING: few valid intervals, or intervals under 1 s. This "
              "folder looks like a COPY (the timestamps are those of the "
              "copy). Run on the original generation folder, or use --logs.")
    if len(per_case):
        med = float(np.median(per_case))
        est = med * n / 3600
        print(f"time per case (from timestamps): median {med:.1f} s, mean "
              f"{per_case.mean():.1f} s, over {len(per_case)} intervals")
        print(f"-> estimated total computation: {est:.0f} h "
              f"(median x {n} cases); sum of the valid intervals: "
              f"{per_case.sum()/3600:.0f} h")

    if logs:
        times = []
        for lg in glob.glob(os.path.expanduser(logs), recursive=True):
            with open(lg, errors="ignore") as fh:
                for ln in fh:
                    m = re.search(r"\] OK .*\((\d+(?:\.\d+)?)s\)\s*$", ln)
                    if m:
                        times.append(float(m.group(1)))
        if times:
            t = np.array(times)
            print(f"\nlogs: {len(t)} OK cases with a printed time; median "
                  f"{np.median(t):.1f} s; sum {t.sum()/3600:.0f} h")
        else:
            print(f"\nno 'OK ... (Xs)' line in the logs {logs}")


def parallel(dataset, pause):
    """Number of generation processes running at the same time (see the
    module docstring)."""
    _, _, idx, mt = _sample_times(dataset)
    dt = np.diff(mt)
    consecutive = (np.diff(idx) == 1) & (dt > 0) & (dt < pause)
    t_case, when = dt[consecutive], mt[1:][consecutive]   # time of each case
    ts = np.sort(mt)
    g = np.diff(ts)
    active, q_act = np.where(g < pause, g, 0.0), ts[1:]   # clock with generation
    print(f"\nwall clock with active generation: {active.sum()/3600:.0f} h")
    print(f"sum of the per-case times:         {t_case.sum()/3600:.0f} h")
    print(f"-> simultaneous processes on average: "
          f"{t_case.sum()/max(active.sum(), 1):.2f}")
    print(f"\n{'week':>6s} {'cases':>6s} {'s/case':>7s} {'clock h':>10s} "
          f"{'processes':>10s}")
    t0 = ts.min()
    week = 7 * 86400
    for k in range(int((ts.max() - t0) // week) + 1):
        a, b = t0 + k * week, t0 + (k + 1) * week
        mc, ma = (when >= a) & (when < b), (q_act >= a) & (q_act < b)
        nc = int(((mt >= a) & (mt < b)).sum())
        if nc == 0:
            continue
        clock = active[ma].sum()
        med = np.median(t_case[mc]) if mc.any() else float("nan")
        print(f"{k+1:6d} {nc:6d} {med:7.1f} {clock/3600:10.1f} "
              f"{t_case[mc].sum()/max(clock, 1):10.2f}")
    print("\n~1 process: the cases ran one at a time, so the time per case is "
          "the true solve time.\n>=2 processes: the processes shared the "
          "machine, which inflates the time per case.")


def main():
    argv = sys.argv[1:]
    for i, a in enumerate(argv):          # map an old subcommand name
        if not a.startswith("-"):
            argv[i] = LEGACY_COMMANDS.get(a, a)
            break
    ap = argparse.ArgumentParser(
        description="Read-only audit of a generated dataset.",
        epilog="See the module docstring for what each command measures.")
    sp = ap.add_subparsers(dest="cmd", required=True)
    a = sp.add_parser("p-refinement",
                      help="table of the --check-degree verification")
    a.add_argument("--root", default=os.path.dirname(PROJECT_DIR),
                   help="folder searched recursively for conv_*_g*.npz")
    a.add_argument("--threshold", type=float, default=5.0,
                   help="accepted |peak deviation| (%%)")
    a.add_argument("--raiz", dest="root", help=argparse.SUPPRESS)
    a.add_argument("--limite", dest="threshold", type=float,
                   help=argparse.SUPPRESS)
    default_ds = os.path.join(PROJECT_DIR, "dataset_tcc_55803")
    b = sp.add_parser("hours", help="generation time from file timestamps")
    b.add_argument("--dataset", default=default_ds)
    b.add_argument("--pause", type=float, default=600,
                   help="interval (s) above which it is a pause, not "
                        "computation")
    b.add_argument("--logs", default=None,
                   help="glob of saved terminal logs, e.g. '~/logs/*.txt'")
    b.add_argument("--pausa", dest="pause", type=float,
                   help=argparse.SUPPRESS)
    c = sp.add_parser("parallel",
                      help="generation processes running in parallel, "
                           "per week")
    c.add_argument("--dataset", default=default_ds)
    c.add_argument("--pause", type=float, default=600)
    c.add_argument("--pausa", dest="pause", type=float,
                   help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if args.cmd == "parallel":
        parallel(args.dataset, args.pause)
    elif args.cmd == "p-refinement":
        p_refinement(os.path.expanduser(args.root), args.threshold)
    else:
        hours(args.dataset, args.pause, args.logs)


if __name__ == "__main__":
    main()
