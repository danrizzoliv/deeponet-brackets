#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RENDER OF THE SIX GEOMETRIES — one clean image per family, for illustration.

No stress field, no boundary conditions: only the part, in isometric view,
white background and identical framing. Meant for the catalogue slide.

    python render_families.py                 # picks the finest mesh
    python render_families.py --casos L=2007,O=8042
    python render_families.py --angulo 30     # rotates the camera about z

Generates figs/geom_{L,Z,U,T,O,G}.png (1400x1400).
"""

import os
import glob
import argparse
import numpy as np

# start of the block of each family in dataset_v3_p2
# (L 2000-3999, Z 4000-5999, T 6000-7999, O 8000-9999, G 10000-11999;
#  the U comes in by symlink at indices 0-1999)
BLOCKS = {"L": 2000, "Z": 4000, "U": 0, "T": 6000, "O": 8000, "G": 10000}
N_SCAN = 80          # how many cases to look at to pick the finest mesh

BASE = os.path.expanduser("~/projetos/tcc_brackets")
SRC = os.path.join(BASE, "dataset_v3_p2", "samples")
OUT = os.path.join(BASE, "figs")

PART_COLOR = "AEBECE"    # light steel; too dark swallows the details


def choose_cases(fixed):
    """Finest mesh among the first N_SCAN of each block: holes and
    fillets come out round instead of polygonal."""
    chosen = {}
    for fam, i0 in BLOCKS.items():
        if fam in fixed:
            chosen[fam] = fixed[fam]
            continue
        best, best_n = None, -1
        for i in range(i0, i0 + N_SCAN):
            p = os.path.join(SRC, f"sample_{i:06d}.npz")
            if not os.path.exists(p):
                continue
            n = int(np.load(p, allow_pickle=True)["n_nodes"])
            if n > best_n:
                best, best_n = i, n
        if best is None:
            print(f"{fam}: no case found in block {i0}")
        else:
            chosen[fam] = best
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--casos", default="",
                    help="fixes indices, e.g.: L=2007,O=8042")
    ap.add_argument("--angulo", type=float, default=0.0,
                    help="extra camera rotation about z (degrees)")
    ap.add_argument("--zoom", type=float, default=0.82,
                    help="<1 zooms out (more margin); >1 zooms in")
    args = ap.parse_args()

    fixed = {}
    for pair in filter(None, args.casos.split(",")):
        k, v = pair.split("=")
        fixed[k.strip().upper()] = int(v)

    import pyvista as pv
    os.makedirs(OUT, exist_ok=True)
    cases = choose_cases(fixed)

    for fam, i in cases.items():
        d = np.load(os.path.join(SRC, f"sample_{i:06d}.npz"), allow_pickle=True)
        nodes = np.ascontiguousarray(d["nodes"], np.float64)
        cells = np.asarray(d["cells"], np.int64)

        nC = cells.shape[0]
        vtk = np.empty((nC, 5), np.int64)
        vtk[:, 0] = 4
        vtk[:, 1:] = cells
        grid = pv.UnstructuredGrid(
            vtk.ravel(), np.full(nC, int(pv.CellType.TETRA), np.uint8), nodes)
        surf = grid.extract_surface().clean()

        # 'three lights' gives volume; without it the part looks flat
        pl = pv.Plotter(off_screen=True, window_size=(1400, 1400),
                        lighting="three lights", border=False)
        pl.set_background("white")
        try:
            # split_sharp_edges: smooths the CURVED surfaces while keeping the
            # sharp corners. It is what removes the blotches on the fillets without
            # rounding the corners.
            pl.add_mesh(surf, color=PART_COLOR, smooth_shading=True,
                        split_sharp_edges=True, feature_angle=32,
                        specular=0.45, specular_power=22,
                        ambient=0.22, diffuse=0.72)
        except TypeError:
            surf = surf.compute_normals(feature_angle=32, split_vertices=True,
                                        consistent_normals=True)
            pl.add_mesh(surf, color=PART_COLOR, smooth_shading=True,
                        specular=0.45, specular_power=22,
                        ambient=0.22, diffuse=0.72)
        try:
            pl.enable_anti_aliasing("ssaa")     # edges without aliasing
        except Exception:
            pass

        pl.enable_parallel_projection()         # faithful proportions between parts
        pl.view_isometric()
        if args.angulo:
            pl.camera.azimuth += args.angulo
        pl.reset_camera()                       # frames the whole part...
        pl.camera.zoom(args.zoom)               # ...and leaves a margin around it

        out_path = os.path.join(OUT, f"geom_{fam}.png")
        pl.show(screenshot=out_path)
        pl.close()

        gp = np.asarray(d["geo_params"], float)
        print(f"{fam}: case {i:6d} | A={gp[0]:5.1f} B={gp[1]:5.1f} "
              f"T={gp[4]:4.1f} | {nodes.shape[0]:5d} nodes -> {out_path}")


if __name__ == "__main__":
    main()
