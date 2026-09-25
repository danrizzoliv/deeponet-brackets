#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VISUAL INSPECTION — simple brackets (L/Z/U).

  python inspecionar_simples.py --ckpt deeponet_simples.pt \\
      --dataset ~/projetos/simples/dataset_v1 --lista
  python inspecionar_simples.py --ckpt ... --dataset ... \\
      --split val --caso 3 [--salvar val3.png]

2x2 layout: BCs (clamp blue / loaded face orange / arrows F and M) |
FEA | DeepONet | error. Each case has its own mesh and family.
"""

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, os.environ.get("TORCH_THREADS", "2"))

import argparse
import numpy as np
import torch

torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "2")))

from dataset_simples import (case_branch, trunk_feats, denormalize_vm)
from train_simples import SimpleDeepONet, predict_full

FAMILIES = ["L", "Z", "U", "T", "O", "G", "E",
            "F", "X", "J", "W", "S"]


def load_ckpt(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    a = ckpt["args"]
    model = SimpleDeepONet(branch_dim=ckpt["stats"]["branch_dim"],
                            p=a.get("p", 128), hidden=a.get("hidden", 128),
                            layers=a.get("layers", 4),
                            fourier=a.get("fourier", 0),
                            d_extra=2 if a.get("dist_furos") else 0,
                            decoder=a.get("decoder", "dot"),
                            trunk=a.get("trunk", "tanh"))
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt["stats"], ckpt["split"]


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

    The argument name changed between versions ('font' / 'font_family') and in
    some it does not exist; without monospace the left alignment is lost,
    but the figure is still correct."""
    for extra in ({"font": "courier"}, {"font_family": "courier"}, {}):
        try:
            return pl.add_text(txt, **kw, **extra)
        except TypeError:
            continue
    return pl.add_text(txt, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="deeponet_simples.pt")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", choices=["treino", "val"], default="val")
    ap.add_argument("--caso", type=int, default=0)
    ap.add_argument("--lista", action="store_true")
    ap.add_argument("--salvar", default=None, metavar="ARQ.png")
    ap.add_argument("--html", default=None, metavar="ARQ.html",
                    help="exports an INTERACTIVE scene for the Windows browser")
    args = ap.parse_args()

    model, stats, split = load_ckpt(args.ckpt)
    samples_dir = os.path.join(args.dataset, "samples")

    use_dist = bool(model.d_extra)

    def load_case(path):
        b, s, vm, nodes, d = case_branch(path, stats)
        ft = trunk_feats(nodes, d, stats)
        if not use_dist:
            ft = ft[:, :3]
        y = predict_full(model, b, ft, "cpu")
        vm_p = np.clip(denormalize_vm(y, s, stats), 0, None)
        rl = float(np.linalg.norm(vm_p - vm) / (np.linalg.norm(vm) + 1e-9))
        return vm, vm_p, rl, nodes

    if args.lista:
        print(f"{'split':6s} {'case':>4s} {'fam':3s} {'|F| (N)':>9s} "
              f"{'|M| (N·mm)':>11s} {'nu':>6s} {'relL2':>7s} "
              f"{'true peak':>10s} {'pred':>8s} {'err%':>7s}")
        for name in ("treino", "val"):
            errs, peak_errs = [], []
            for i, fb in enumerate(split[name]):
                p = os.path.join(samples_dir, fb)
                d = np.load(p, allow_pickle=True)
                vm_t, vm_p, rl, _ = load_case(p)
                errs.append(rl)
                ep = 100.0 * (vm_p.max() - vm_t.max()) / (vm_t.max() + 1e-30)
                peak_errs.append(ep)
                print(f"{name:6s} {i:4d} {FAMILIES[int(d['familia'])]:3s} "
                      f"{np.linalg.norm(d['loads']):9.1f} "
                      f"{np.linalg.norm(d['moments']):11.1f} "
                      f"{float(d['nu']):6.3f} {rl:7.3f} "
                      f"{vm_t.max():10.1f} {vm_p.max():8.1f} {ep:+7.1f}")
            e, q = np.array(errs), np.array(peak_errs)
            print(f"--> {name}: mean relL2={e.mean():.3f} "
                  f"median={np.median(e):.3f} p90={np.percentile(e, 90):.3f} "
                  f"| median peak error={np.median(q):+.1f}% "
                  f"|abs| med={np.median(np.abs(q)):.1f}% "
                  f"p90={np.percentile(np.abs(q), 90):.1f}% "
                  f"({len(e)} cases)\n")
        return

    files = split[args.split]
    if not (0 <= args.caso < len(files)):
        raise SystemExit(f"--caso must be in [0, {len(files)-1}]")
    path = os.path.join(samples_dir, files[args.caso])
    vm_true, vm_pred, relL2, nodes = load_case(path)
    d = np.load(path, allow_pickle=True)
    fam = FAMILIES[int(d["familia"])]
    nF = float(np.linalg.norm(d["loads"]))
    nM = float(np.linalg.norm(d["moments"]))
    err = np.abs(vm_pred - vm_true)
    print(f"[{args.split} #{args.caso}] {fam}-bracket | |F|={nF:.0f} N "
          f"|M|={nM:.0f} N·mm | vm_max true={vm_true.max():.1f} "
          f"pred={vm_pred.max():.1f} MPa | relL2={relL2:.3f}")

    import pyvista as pv
    cells = d["cells"]
    vmax = float(max(vm_true.max(), vm_pred.max()))
    clim = [0.0, vmax]
    SUP = dict(algorithm="dataset_surface")
    s_t = build_grid(nodes, cells, vm_true,
                     "von Mises (FEA)").extract_surface(**SUP)
    s_p = build_grid(nodes, cells, vm_pred,
                     "von Mises (DeepONet)").extract_surface(**SUP)
    s_e = build_grid(nodes, cells, err,
                     "|erro| (MPa)").extract_surface(**SUP)
    s_d = build_grid(nodes, cells, vm_pred - vm_true,
                     "erro assinado (MPa)").extract_surface(**SUP)
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

    off = args.salvar is not None or args.html is not None
    root = (args.salvar or args.html or "cena")
    root = root[:-4] if root.lower().endswith((".png", "html")) else root
    root = root[:-1] if root.endswith(".") else root

    # ---------------- row 1: boundary | mesh | signed error --------------
    pl = new_plotter(off)
    pl.subplot(0, 0)
    pl.add_text("Boundary conditions", font_size=24)
    pl.add_mesh(s_t, color="lightgray", opacity=0.55, smooth_shading=False)
    fixm = np.zeros(nodes.shape[0], bool)
    for h in np.asarray(d["furos_fix"], float).reshape(-1, 8):
        fixm |= mark_cylinder_np(nodes, h)
    loadm = np.zeros(nodes.shape[0], bool)
    for h in np.asarray(d["furos_carga"], float).reshape(-1, 8):
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
    pl.add_mesh(s_d, scalars="erro assinado (MPa)", cmap="coolwarm",
                clim=[-emax, emax], smooth_shading=False,
                scalar_bar_args=scalar_bar("signed error (MPa)"))
    close_plotter(pl, f"{root}_setup.png" if args.salvar else None,
           f"{root}_setup.html" if args.html else None)

    # ---------------- row 2: DeepONet | FEA | |error| --------------------
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
    pl.add_mesh(s_e, scalars="|erro| (MPa)", cmap="hot", smooth_shading=False,
                scalar_bar_args=scalar_bar("|error| (MPa)"))
    close_plotter(pl, f"{root}_result.png" if args.salvar else None,
           f"{root}_result.html" if args.html else None)

    if args.salvar:
        print(f"figures saved to {root}_setup.png and {root}_result.png")
    elif args.html:
        print(f"scenes saved to {root}_setup.html and {root}_result.html")
    return



if __name__ == "__main__":
    main()
