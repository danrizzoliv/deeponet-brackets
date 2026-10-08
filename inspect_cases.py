#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Per-case inspection of a trained bracket DeepONet: a listing of the error of
every case, or the two three-panel figures of one case.

With --listing, evaluates every training and validation case of the
checkpoint's split and prints one line per case (format below), followed by a
summary per split; summarize_listing.py turns it into the per-family table.
Without it, renders one case (pyvista) as two 1x3 figures:
    setup:  boundary conditions (clamped hole surfaces blue, loaded hole
            surfaces orange, red arrow F, double purple arrow M) | mesh, with
            load, material and size | signed error (DeepONet - FEA)
    result: DeepONet prediction | finite element reference | absolute error
Works with checkpoints of the old and of the current training code, including
models trained with --edge-distance (reads the concave-edge file of the
dataset) and --hole-proximity.

    python inspect_cases.py --ckpt deeponet_6fam_55k.pt \\
        --dataset ~/projetos/tcc_brackets/dataset_v4_p2 --listing > listing.txt
    python inspect_cases.py --ckpt ... --dataset ... --split val --case 3 \\
        [--save val3.png | --html val3.html] [--meshes FULL_DATASET]

--save val3.png writes val3_setup.png and val3_result.png (--html likewise).
The compact training dataset (compact_dataset.py) has no mesh connectivity:
to render one of its cases, pass the full dataset with --meshes; its nodes
are checked against those of the compact case.

Listing line (whitespace-separated; parsers read columns 0, 1, 2, 6, 7, 9):
    split case fam |F| (N) |M| (N.mm) nu relL2 true_peak pred_peak err%
    f"{split:6s} {case:4d} {fam:3s} {F:9.1f} {M:11.1f} {nu:6.3f} {relL2:7.3f}"
    f" {true_peak:10.1f} {pred_peak:8.1f} {peak_err:+7.1f}"
with split = "train" or "val" (listings written before the translation say
"treino" for the training rows; the columns are otherwise the same).
Old option names (--caso, --lista, --salvar, --malhas) are still accepted.
"""

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, os.environ.get("TORCH_THREADS", "2"))

import argparse
import glob
import numpy as np
import torch

torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "2")))

from bracket_dataset import (FAMILIES, KEY_FAMILY, KEY_FIXATION_HOLES,
                             KEY_LOAD_HOLES, load_edge_map)
from train_deeponet import (load_model, checkpoint_split, predict_case,
                            relative_l2)

SPLIT_ALIASES = {"treino": "train"}   # split name used before the translation


def build_grid(nodes, cells, scalars, name):
    import pyvista as pv
    nodes = np.ascontiguousarray(nodes, dtype=np.float64)
    cells = np.asarray(cells, dtype=np.int64)
    nC = cells.shape[0]
    vtk = np.empty((nC, 5), dtype=np.int64)
    vtk[:, 0] = 4
    vtk[:, 1:] = cells
    ctypes = np.full(nC, int(pv.CellType.TETRA), dtype=np.uint8)
    grid = pv.UnstructuredGrid(vtk.ravel(), ctypes, nodes)
    grid[name] = np.asarray(scalars, dtype=np.float64)
    return grid


def mark_cylinder_np(nodes, h):
    """h = [cx,cy,cz, ax,ay,az, r, half] -> mask of the nodes on the shell."""
    c = np.array(h[:3], float); axis = np.array(h[3:6], float)
    axis = axis / (np.linalg.norm(axis) + 1e-30)
    radius, half = float(h[6]), float(h[7])
    d = nodes - c
    proj = d @ axis
    radial = np.linalg.norm(d - np.outer(proj, axis), axis=1)
    return (np.abs(radial - radius) < 0.35 * radius + 0.5) & (np.abs(proj) < half + 1.0)


def _arrow(pl, origin, direction, length, color, double=False):
    import pyvista as pv
    direction = np.asarray(direction, float)
    direction = direction / (np.linalg.norm(direction) + 1e-30)
    pl.add_mesh(pv.Arrow(start=origin, direction=direction, scale=length,
                         tip_length=0.22, tip_radius=0.055,
                         shaft_radius=0.018), color=color)
    if double:
        pl.add_mesh(pv.Arrow(start=origin, direction=direction,
                             scale=0.84 * length, tip_length=0.26,
                             tip_radius=0.066, shaft_radius=0.0), color=color)


def in_box(nodes, box):
    lo, hi = np.asarray(box[0], float), np.asarray(box[1], float)
    tol = 1e-6 + 0.0
    return np.all((nodes >= lo - tol) & (nodes <= hi + tol), axis=1)


def mono_text(pl, txt, **kw):
    """add_text with a monospaced font, tolerant to the PyVista version.

    The argument name changed between versions ('font' / 'font_family') and
    in some it does not exist; without monospace the left alignment is lost,
    but the figure is still correct."""
    for extra in ({"font": "courier"}, {"font_family": "courier"}, {}):
        try:
            return pl.add_text(txt, **kw, **extra)
        except TypeError:
            continue
    return pl.add_text(txt, **kw)


def split_name(value):
    """--split value: 'train' or 'val' ('treino' accepted for 'train')."""
    value = SPLIT_ALIASES.get(value, value)
    if value not in ("train", "val"):
        raise argparse.ArgumentTypeError(
            f"invalid split {value!r} (choose train or val)")
    return value


def parse_args():
    ap = argparse.ArgumentParser(
        description="Per-case listing or figures of a trained bracket "
                    "DeepONet.")
    ap.add_argument("--ckpt", default="deeponet.pt")
    ap.add_argument("--dataset", required=True,
                    help="dataset folder (with samples/) the model was "
                         "trained on")
    ap.add_argument("--split", type=split_name, default="val",
                    metavar="{train,val}")
    ap.add_argument("--case", type=int, default=0,
                    help="index of the case in the checkpoint's split")
    ap.add_argument("--listing", action="store_true",
                    help="print one line per case of both splits and exit")
    ap.add_argument("--save", default=None, metavar="FILE.png",
                    help="write FILE_setup.png and FILE_result.png")
    ap.add_argument("--meshes", default=None, metavar="DATASET",
                    help="full dataset to read the connectivity (cells) "
                         "from, when --dataset is the compact version, "
                         "without mesh; the nodes are checked")
    ap.add_argument("--html", default=None, metavar="FILE.html",
                    help="exports an INTERACTIVE scene for the Windows "
                         "browser")
    # names used before the translation
    ap.add_argument("--caso", dest="case", type=int, help=argparse.SUPPRESS)
    ap.add_argument("--lista", dest="listing", action="store_true",
                    help=argparse.SUPPRESS)
    ap.add_argument("--salvar", dest="save", help=argparse.SUPPRESS)
    ap.add_argument("--malhas", dest="meshes", help=argparse.SUPPRESS)
    return ap.parse_args()


def mesh_cells(path, d, nodes, meshes):
    """Connectivity of the case: from the case itself, or from the same file
    of the full dataset `meshes` (the compact dataset has no cells)."""
    if "cells" in d.files:
        return d["cells"]
    if meshes:
        pm = os.path.join(os.path.expanduser(meshes), "samples",
                          os.path.basename(path))
        dm = np.load(pm, allow_pickle=True)
        nm = np.asarray(dm["nodes"], np.float64)
        if nm.shape != nodes.shape or not np.allclose(nm, nodes, atol=1e-6):
            raise SystemExit(f"the nodes of {pm} do not match those of "
                             f"{path}")
        return dm["cells"]
    raise SystemExit(f"{os.path.basename(path)} has no mesh (cells): it is "
                     "the compact version of the dataset; pass --meshes "
                     "with the full dataset")


def main():
    args = parse_args()

    model, ckpt, cfg = load_model(args.ckpt, "cpu")
    stats = ckpt["stats"]
    split = checkpoint_split(ckpt)
    samples_dir = os.path.join(args.dataset, "samples")

    # a model trained with the edge distance needs the edges of the dataset;
    # the others are evaluated without them, exactly as before
    edge_map = None
    if cfg.get("edge_distance"):
        edge_map, missing = load_edge_map(
            sorted(glob.glob(os.path.join(samples_dir, "*.npz"))))
        print(f"[edge-distance] {missing} cases without a reconstructed edge")

    def evaluate(path):
        vm, vm_p, nodes, d = predict_case(model, path, stats, cfg, "cpu",
                                          edge_map)
        vm_p = np.clip(vm_p, 0, None)
        return vm, vm_p, relative_l2(vm_p, vm), nodes, d

    if args.listing:
        print(f"{'split':6s} {'case':>4s} {'fam':3s} {'|F| (N)':>9s} "
              f"{'|M| (N·mm)':>11s} {'nu':>6s} {'relL2':>7s} "
              f"{'true peak':>10s} {'pred':>8s} {'err%':>7s}")
        for name in ("train", "val"):
            errs, peak_errs = [], []
            for i, fb in enumerate(split[name]):
                p = os.path.join(samples_dir, fb)
                vm_t, vm_p, rl, _, d = evaluate(p)
                errs.append(rl)
                ep = 100.0 * (vm_p.max() - vm_t.max()) / (vm_t.max() + 1e-30)
                peak_errs.append(ep)
                print(f"{name:6s} {i:4d} {FAMILIES[int(d[KEY_FAMILY])]:3s} "
                      f"{np.linalg.norm(d['loads']):9.1f} "
                      f"{np.linalg.norm(d['moments']):11.1f} "
                      f"{float(d['nu']):6.3f} {rl:7.3f} "
                      f"{vm_t.max():10.1f} {vm_p.max():8.1f} {ep:+7.1f}")
            if not errs:
                continue
            e, q = np.array(errs), np.array(peak_errs)
            print(f"--> {name}: mean relL2={e.mean():.3f} "
                  f"median={np.median(e):.3f} p90={np.percentile(e, 90):.3f} "
                  f"| median peak error={np.median(q):+.1f}% "
                  f"|abs| med={np.median(np.abs(q)):.1f}% "
                  f"p90={np.percentile(np.abs(q), 90):.1f}% "
                  f"({len(e)} cases)\n")
        return

    files = split[args.split]
    if not (0 <= args.case < len(files)):
        raise SystemExit(f"--case must be in [0, {len(files)-1}]")
    path = os.path.join(samples_dir, files[args.case])
    vm_true, vm_pred, relL2, nodes, d = evaluate(path)
    fam = FAMILIES[int(d[KEY_FAMILY])]
    nF = float(np.linalg.norm(d["loads"]))
    nM = float(np.linalg.norm(d["moments"]))
    err = np.abs(vm_pred - vm_true)
    print(f"[{args.split} #{args.case}] {fam}-bracket | |F|={nF:.0f} N "
          f"|M|={nM:.0f} N·mm | vm_max true={vm_true.max():.1f} "
          f"pred={vm_pred.max():.1f} MPa | relL2={relL2:.3f}")

    cells = mesh_cells(path, d, nodes, args.meshes)
    import pyvista as pv
    vmax = float(max(vm_true.max(), vm_pred.max()))
    clim = [0.0, vmax]
    SUP = dict(algorithm="dataset_surface")
    s_t = build_grid(nodes, cells, vm_true,
                     "von Mises (FEA)").extract_surface(**SUP)
    s_p = build_grid(nodes, cells, vm_pred,
                     "von Mises (DeepONet)").extract_surface(**SUP)
    s_e = build_grid(nodes, cells, err,
                     "|error| (MPa)").extract_surface(**SUP)
    s_d = build_grid(nodes, cells, vm_pred - vm_true,
                     "signed error (MPa)").extract_surface(**SUP)
    emax = float(np.abs(vm_pred - vm_true).max() + 1e-12)

    E = next((float(d[k]) for k in ("E", "young", "E_MPa")
              if k in d.files), float("nan"))
    nu = float(d["nu"])
    _lines = [f"|F| = {nF:.0f} N",
              f"|M| = {nM:.0f} N.mm",
              f"E = {E/1000.0:.0f} GPa   nu = {nu:.3f}",
              f"{nodes.shape[0]} nodes, {cells.shape[0]} tets"]
    _w = max(len(x) for x in _lines)
    header = "\n".join(x.ljust(_w) for x in _lines)

    def scalar_bar(title):
        return {"title": title, "vertical": False, "width": 0.72,
                "height": 0.060, "position_x": 0.14, "position_y": 0.030,
                "title_font_size": 30, "label_font_size": 26,
                "n_labels": 4, "color": "black"}

    def new_plotter(off):
        p = pv.Plotter(shape=(1, 3), window_size=(2400, 1250),
                       off_screen=off)
        return p

    def close_plotter(p, fpath, html):
        p.link_views()
        p.reset_camera()
        p.camera.zoom(0.84)
        if html:
            p.export_html(html)
        elif fpath:
            p.show(screenshot=fpath)
        else:
            p.show()
        p.close()

    off = args.save is not None or args.html is not None
    root = (args.save or args.html or "scene")
    root = root[:-4] if root.lower().endswith((".png", "html")) else root
    root = root[:-1] if root.endswith(".") else root

    # ---------------- figure 1: boundary | mesh | signed error -----------
    pl = new_plotter(off)
    pl.subplot(0, 0)
    pl.add_text("Boundary conditions", font_size=24)
    pl.add_mesh(s_t, color="lightgray", opacity=0.55, smooth_shading=False)
    fixm = np.zeros(nodes.shape[0], bool)
    for h in np.asarray(d[KEY_FIXATION_HOLES], float).reshape(-1, 8):
        fixm |= mark_cylinder_np(nodes, h)
    loadm = np.zeros(nodes.shape[0], bool)
    for h in np.asarray(d[KEY_LOAD_HOLES], float).reshape(-1, 8):
        loadm |= mark_cylinder_np(nodes, h)
    if fixm.any():
        pl.add_points(nodes[fixm], color="royalblue", point_size=7,
                      render_points_as_spheres=True)
    if loadm.any():
        pl.add_points(nodes[loadm], color="orange", point_size=7,
                      render_points_as_spheres=True)
    diag = float(np.linalg.norm(nodes.max(0) - nodes.min(0)))
    center = np.asarray(d["load_centers"], float).reshape(3)
    _arrow(pl, center, np.asarray(d["loads"]).reshape(3), 0.22 * diag, "red")
    _arrow(pl, center, np.asarray(d["moments"]).reshape(3), 0.19 * diag,
           "purple", double=True)
    _legend = ["blue = clamped   orange = loaded",
               "red arrow = F    double purple = M"]
    _wl = max(len(x) for x in _legend)
    mono_text(pl, "\n".join(x.ljust(_wl) for x in _legend),
              position="lower_left", font_size=18)

    pl.subplot(0, 1)
    pl.add_text("Mesh", font_size=24)
    pl.add_mesh(s_t, color="white", show_edges=True, edge_color="gray",
                line_width=0.5)
    mono_text(pl, header, position="lower_left", font_size=18)

    pl.subplot(0, 2)
    pl.add_text("Signed error (MPa)", font_size=24)
    pl.add_mesh(s_d, scalars="signed error (MPa)", cmap="coolwarm",
                clim=[-emax, emax], smooth_shading=False,
                scalar_bar_args=scalar_bar("signed error (MPa)"))
    close_plotter(pl, f"{root}_setup.png" if args.save else None,
                  f"{root}_setup.html" if args.html else None)

    # ---------------- figure 2: DeepONet | FEA | |error| -----------------
    pl = new_plotter(off)
    pl.subplot(0, 0)
    pl.add_text(f"DeepONet prediction (relL2 = {relL2:.2f})", font_size=24)
    pl.add_mesh(s_p, scalars="von Mises (DeepONet)", cmap="turbo", clim=clim,
                smooth_shading=False,
                scalar_bar_args=scalar_bar("von Mises (DeepONet)"))

    pl.subplot(0, 1)
    pl.add_text("Finite element reference", font_size=24)
    pl.add_mesh(s_t, scalars="von Mises (FEA)", cmap="turbo", clim=clim,
                smooth_shading=False,
                scalar_bar_args=scalar_bar("von Mises (FEA)"))

    pl.subplot(0, 2)
    pl.add_text("Absolute error (MPa)", font_size=24)
    pl.add_mesh(s_e, scalars="|error| (MPa)", cmap="hot", smooth_shading=False,
                scalar_bar_args=scalar_bar("|error| (MPa)"))
    close_plotter(pl, f"{root}_result.png" if args.save else None,
                  f"{root}_result.html" if args.html else None)

    if args.save:
        print(f"figures saved to {root}_setup.png and {root}_result.png")
    elif args.html:
        print(f"scenes saved to {root}_setup.html and {root}_result.html")
    return


if __name__ == "__main__":
    main()
