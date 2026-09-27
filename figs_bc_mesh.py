#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BOUNDARY-CONDITION AND MESH PANELS — standalone figures, in English.

Generates separate images of the same case, for the methodology section:
  figs/bc_{FAM}_{i}.png     boundary conditions (clamp, load, F and M)
  figs/mesh_{FAM}_{i}.png   tetrahedral mesh
  figs/vm_{FAM}_{i}.png     FEA von Mises field

    python figs_bc_mesh.py --dataset ~/projetos/tcc_brackets/dataset_v3_p2 --caso 0
    python figs_bc_mesh.py --dataset ... --caso 2000 --angulo 40
"""

import os
import glob
import argparse
import numpy as np

FAMILIES = ["L", "Z", "U", "T", "O", "G", "E", "F", "X", "J", "W", "S"]
PART_COLOR = "AEBECE"


def mark_cylinder(nodes, h, tol=0.12):
    """h = [cx,cy,cz, ax,ay,az, r, half] -> mask of the nodes ON THE SHELL of the hole.

    The tolerance is a fraction of the radius and grows by trial: it starts
    tight, so that the markers form a clean ring instead of a
    smudge, and loosens only if the mesh is too coarse to hit
    any node."""
    c = np.array(h[:3], float)
    axis = np.array(h[3:6], float)
    axis = axis / (np.linalg.norm(axis) + 1e-30)
    radius, half = float(h[6]), float(h[7])
    d = nodes - c
    proj = d @ axis
    radial = np.linalg.norm(d - np.outer(proj, axis), axis=1)
    inside = np.abs(proj) < half + 0.5
    for f in (tol, 2 * tol, 4 * tol):
        m = (np.abs(radial - radius) < f * radius + 0.05) & inside
        if m.sum() >= 8:
            return m
    return m


def arrow(pl, origin, direction, length, color, double=False):
    import pyvista as pv
    direction = np.asarray(direction, float)
    n = np.linalg.norm(direction)
    if n < 1e-12:
        return
    direction = direction / n
    pl.add_mesh(pv.Arrow(start=origin, direction=direction, scale=length,
                         tip_length=0.22, tip_radius=0.055,
                         shaft_radius=0.018), color=color)
    if double:
        pl.add_mesh(pv.Arrow(start=origin, direction=direction,
                             scale=0.84 * length, tip_length=0.26,
                             tip_radius=0.066, shaft_radius=0.0), color=color)


def make_grid(nodes, cells):
    import pyvista as pv
    nC = cells.shape[0]
    vtk = np.empty((nC, 5), np.int64)
    vtk[:, 0] = 4
    vtk[:, 1:] = cells
    return pv.UnstructuredGrid(
        vtk.ravel(), np.full(nC, int(pv.CellType.TETRA), np.uint8),
        np.ascontiguousarray(nodes))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--caso", type=int, default=0)
    ap.add_argument("--angulo", type=float, default=0.0,
                    help="extra camera rotation about z (degrees)")
    ap.add_argument("--zoom", type=float, default=0.95)
    ap.add_argument("--cor-malha", default="3D4A5C",
                    help="color of the mesh edges in hex; use 000000 for "
                         "black or 1C6EA4 for the palette blue")
    ap.add_argument("--esp-malha", type=float, default=0.9,
                    help="thickness of the mesh edges")
    ap.add_argument("--res", type=int, default=2400,
                    help="image width in pixels (height = 0.9*width); "
                         "2400 is good for print, 3600 for heavy zooming")
    ap.add_argument("--sem-aa", action="store_true",
                    help="turns off anti-aliasing (faster, worse)")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    import pyvista as pv

    out = args.out_dir or os.path.join(os.path.dirname(args.dataset), "figs")
    os.makedirs(out, exist_ok=True)

    p = os.path.join(args.dataset, "samples", f"sample_{args.caso:06d}.npz")
    if not os.path.exists(p):
        fs = sorted(glob.glob(os.path.join(args.dataset, "samples", "*.npz")))
        raise SystemExit(f"{p} does not exist ({len(fs)} samples in the directory)")

    d = np.load(p, allow_pickle=True)
    nodes = np.asarray(d["nodes"], np.float64)
    cells = np.asarray(d["cells"], np.int64)
    fam = FAMILIES[int(d["familia"])]
    surf = make_grid(nodes, cells).extract_surface()

    def camera(pl):
        pl.enable_parallel_projection()
        pl.view_isometric()
        if args.angulo:
            pl.camera.azimuth += args.angulo
        pl.reset_camera()
        pl.camera.zoom(args.zoom)

    # ---------- panel 1: boundary conditions ----------
    pl = pv.Plotter(off_screen=True, window_size=(args.res, int(0.9 * args.res)),
                    lighting="three lights", border=False)
    pl.set_background("white")
    pl.add_mesh(surf, color=PART_COLOR, opacity=0.58,
                smooth_shading=False)

    fixm = np.zeros(nodes.shape[0], bool)
    for h in np.asarray(d["furos_fix"], float).reshape(-1, 8):
        fixm |= mark_cylinder(nodes, h)
    loadm = np.zeros(nodes.shape[0], bool)
    for h in np.asarray(d["furos_carga"], float).reshape(-1, 8):
        loadm |= mark_cylinder(nodes, h)

    if fixm.any():
        pl.add_points(nodes[fixm], color="1C6EA4", point_size=11,
                      render_points_as_spheres=True)
    if loadm.any():
        pl.add_points(nodes[loadm], color="E08A1E", point_size=11,
                      render_points_as_spheres=True)

    diag = float(np.linalg.norm(nodes.max(0) - nodes.min(0)))
    center = np.asarray(d["load_centers"], float).reshape(3)
    # sphere at the center of the virtual pin: without it the arrows look loose
    pl.add_mesh(pv.Sphere(radius=0.022 * diag, center=center),
                color="3D4A5C")
    arrow(pl, center, np.asarray(d["loads"]).reshape(3), 0.34 * diag, "C1440E")
    arrow(pl, center, np.asarray(d["moments"]).reshape(3), 0.26 * diag,
         "7A3B8F", double=True)
    # the legend goes in the LaTeX caption, not burned into the image
    camera(pl)
    a1 = os.path.join(out, f"bc_{fam}_{args.caso}.png")
    pl.show(screenshot=a1)
    pl.close()

    # ---------- panel 2: mesh ----------
    pl = pv.Plotter(off_screen=True, window_size=(args.res, int(0.9 * args.res)),
                    lighting="three lights", border=False)
    pl.set_background("white")
    pl.add_mesh(surf, color="FAFBFC", show_edges=True,
                edge_color=args.cor_malha, line_width=args.esp_malha,
                smooth_shading=False)
    # the node count goes in the LaTeX caption
    if not args.sem_aa:
        try:
            pl.enable_anti_aliasing("ssaa")
        except Exception:
            pass
    camera(pl)
    a2 = os.path.join(out, f"mesh_{fam}_{args.caso}.png")
    pl.show(screenshot=a2)
    pl.close()

    # ---------- panel 3: von Mises field ----------
    vm = np.asarray(d["von_mises"], np.float64)
    surf_vm = make_grid(nodes, cells).extract_surface()
    surf_vm["von Mises (MPa)"] = vm[:surf_vm.n_points] \
        if surf_vm.n_points <= vm.size else vm
    pl = pv.Plotter(off_screen=True,
                    window_size=(args.res, int(0.9 * args.res)),
                    lighting="three lights", border=False)
    pl.set_background("white")
    pl.add_mesh(make_grid(nodes, cells).extract_surface(),
                scalars=vm, cmap="turbo", smooth_shading=False,
                scalar_bar_args={"title": "von Mises (MPa)",
                                 "color": "3D4A5C", "n_labels": 5,
                                 "title_font_size": 26, "label_font_size": 22,
                                 "width": 0.62, "height": 0.07,
                                 "position_x": 0.19, "position_y": 0.03})
    if not args.sem_aa:
        try:
            pl.enable_anti_aliasing("ssaa")
        except Exception:
            pass
    camera(pl)
    a3 = os.path.join(out, f"vm_{fam}_{args.caso}.png")
    pl.show(screenshot=a3)
    pl.close()
    print(f"  von Mises peak: {vm.max():.1f} MPa")

    print(f"{fam} case {args.caso}: {nodes.shape[0]} nodes, "
          f"{cells.shape[0]} tets\n  {a1}\n  {a2}\n  {a3}")


if __name__ == "__main__":
    main()
