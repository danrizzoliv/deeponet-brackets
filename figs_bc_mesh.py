#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BOUNDARY-CONDITION, MESH AND FEA PANELS: standalone figures of one case.

Reads one case of a dataset (no model involved) and writes three separate
images with the same camera, for the methodology section; the legends and
the node count go in the LaTeX captions, not in the images:
  figs/bc_{FAM}_{i}.png     boundary conditions (clamped and loaded hole
                            nodes, virtual pin, force F and moment M)
  figs/mesh_{FAM}_{i}.png   tetrahedral mesh
  figs/vm_{FAM}_{i}.png     FEA von Mises field
The output folder defaults to figs/ next to the dataset folder.

    python figs_bc_mesh.py --dataset ~/projetos/tcc_brackets/dataset_v3_p2 --case 0
    python figs_bc_mesh.py --dataset ... --case 2000 --angle 40
"""

import os
import glob
import argparse
import numpy as np

from bracket_dataset import (FAMILIES, KEY_FAMILY, KEY_FIXATION_HOLES,
                             KEY_LOAD_HOLES)

PART_COLOR = "AEBECE"


def mark_cylinder(nodes, h, tol=0.12):
    """h = [cx,cy,cz, ax,ay,az, r, half] -> mask of the nodes ON THE SHELL
    of the hole.

    The tolerance is a fraction of the radius and grows by trial: it starts
    tight, so that the markers form a clean ring instead of a smudge, and
    loosens only if the mesh is too coarse to hit at least 8 nodes."""
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
    """Force arrow; `double` adds the second head of a moment arrow."""
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
    n_cells = cells.shape[0]
    vtk = np.empty((n_cells, 5), np.int64)
    vtk[:, 0] = 4
    vtk[:, 1:] = cells
    return pv.UnstructuredGrid(
        vtk.ravel(), np.full(n_cells, int(pv.CellType.TETRA), np.uint8),
        np.ascontiguousarray(nodes))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--case", type=int, default=0,
                    help="case index: reads samples/sample_<case>.npz")
    ap.add_argument("--angle", type=float, default=0.0,
                    help="extra camera rotation about z (degrees)")
    ap.add_argument("--zoom", type=float, default=0.95)
    ap.add_argument("--mesh-color", default="3D4A5C",
                    help="colour of the mesh edges in hex; use 000000 for "
                         "black or 1C6EA4 for the palette blue")
    ap.add_argument("--mesh-width", type=float, default=0.9,
                    help="thickness of the mesh edges")
    ap.add_argument("--res", type=int, default=2400,
                    help="image width in pixels (height = 0.9*width); "
                         "2400 is good for print, 3600 for heavy zooming")
    ap.add_argument("--no-aa", action="store_true",
                    help="turns off anti-aliasing (faster, worse)")
    ap.add_argument("--out-dir", default=None)
    # option names before the translation
    ap.add_argument("--caso", dest="case", type=int, help=argparse.SUPPRESS)
    ap.add_argument("--angulo", dest="angle", type=float,
                    help=argparse.SUPPRESS)
    ap.add_argument("--cor-malha", dest="mesh_color", help=argparse.SUPPRESS)
    ap.add_argument("--esp-malha", dest="mesh_width", type=float,
                    help=argparse.SUPPRESS)
    ap.add_argument("--sem-aa", dest="no_aa", action="store_true",
                    help=argparse.SUPPRESS)
    args = ap.parse_args()

    import pyvista as pv

    out = args.out_dir or os.path.join(os.path.dirname(args.dataset), "figs")
    os.makedirs(out, exist_ok=True)

    p = os.path.join(args.dataset, "samples", f"sample_{args.case:06d}.npz")
    if not os.path.exists(p):
        fs = sorted(glob.glob(os.path.join(args.dataset, "samples", "*.npz")))
        raise SystemExit(f"{p} does not exist ({len(fs)} samples in the directory)")

    d = np.load(p, allow_pickle=True)
    nodes = np.asarray(d["nodes"], np.float64)
    cells = np.asarray(d["cells"], np.int64)
    fam = FAMILIES[int(d[KEY_FAMILY])]
    surf = make_grid(nodes, cells).extract_surface()

    def camera(pl):
        pl.enable_parallel_projection()
        pl.view_isometric()
        if args.angle:
            pl.camera.azimuth += args.angle
        pl.reset_camera()
        pl.camera.zoom(args.zoom)

    # ---------- panel 1: boundary conditions ----------
    pl = pv.Plotter(off_screen=True, window_size=(args.res, int(0.9 * args.res)),
                    lighting="three lights", border=False)
    pl.set_background("white")
    pl.add_mesh(surf, color=PART_COLOR, opacity=0.58,
                smooth_shading=False)

    fix_mask = np.zeros(nodes.shape[0], bool)
    for h in np.asarray(d[KEY_FIXATION_HOLES], float).reshape(-1, 8):
        fix_mask |= mark_cylinder(nodes, h)
    load_mask = np.zeros(nodes.shape[0], bool)
    for h in np.asarray(d[KEY_LOAD_HOLES], float).reshape(-1, 8):
        load_mask |= mark_cylinder(nodes, h)

    if fix_mask.any():
        pl.add_points(nodes[fix_mask], color="1C6EA4", point_size=11,
                      render_points_as_spheres=True)
    if load_mask.any():
        pl.add_points(nodes[load_mask], color="E08A1E", point_size=11,
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
    a1 = os.path.join(out, f"bc_{fam}_{args.case}.png")
    pl.show(screenshot=a1)
    pl.close()

    # ---------- panel 2: mesh ----------
    pl = pv.Plotter(off_screen=True, window_size=(args.res, int(0.9 * args.res)),
                    lighting="three lights", border=False)
    pl.set_background("white")
    pl.add_mesh(surf, color="FAFBFC", show_edges=True,
                edge_color=args.mesh_color, line_width=args.mesh_width,
                smooth_shading=False)
    # the node count goes in the LaTeX caption
    if not args.no_aa:
        try:
            pl.enable_anti_aliasing("ssaa")
        except Exception:
            pass
    camera(pl)
    a2 = os.path.join(out, f"mesh_{fam}_{args.case}.png")
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
    if not args.no_aa:
        try:
            pl.enable_anti_aliasing("ssaa")
        except Exception:
            pass
    camera(pl)
    a3 = os.path.join(out, f"vm_{fam}_{args.case}.png")
    pl.show(screenshot=a3)
    pl.close()
    print(f"  von Mises peak: {vm.max():.1f} MPa")

    print(f"{fam} case {args.case}: {nodes.shape[0]} nodes, "
          f"{cells.shape[0]} tets\n  {a1}\n  {a2}\n  {a3}")


if __name__ == "__main__":
    main()
