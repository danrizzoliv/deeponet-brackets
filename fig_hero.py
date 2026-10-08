#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OPENING FIGURE — von Mises field with the look of a commercial
post-processor.

Three things produce that look: DISCRETE color bands instead of a
continuous gradient (the classic FEA contour), smooth shading without
facets, and black feature edges.

Reads one case of a dataset (no model involved; FEA field only) and draws
an arrow and label at the node of maximum stress. --search ranks
candidate cases instead of rendering.

    python fig_hero.py --dataset dataset_v4_p2 --case 4445
    python fig_hero.py --dataset dataset_v4_p2 --search U     # lists candidates
    python fig_hero.py --dataset dataset_v4_p2 --case 4445 --bands 12 --html

The --html mode exports the scene for the browser, which renders with
accelerated WebGL; PyVista's off-screen mode under WSL uses software
rasterization and looks worse.
"""

import os
import glob
import argparse
import numpy as np

from bracket_dataset import (FAMILIES, KEY_FAMILY, KEY_FIXATION_HOLES,
                             KEY_LOAD_HOLES)

PROJECT_DIR = os.path.expanduser("~/projetos/tcc_brackets")  # author's project folder; adjust
SEARCH_ALL = ("all", "todas")      # --search values that sweep every family
GRADIENT = ("gradient", "gradiente")   # --background values for the gradient


def candidates(src, fam, n=14, rmin=0.0):
    """Looks for AESTHETICALLY good cases, which is different from cases with a
    high peak. A peak 60x the median leaves the whole part in dark blue with a
    red dot: the field does not span the color scale. What makes a nice image
    is a WELL-DISTRIBUTED field --- peak between 2.5 and 6.5 times the
    median, and a peak of 80 to 700 MPa --- together with large holes and a
    fine mesh. Only every fifth file is examined; prints the best `n` and
    returns them all, best first."""
    target_fam = (FAMILIES.index(fam.upper())
                  if fam and fam.upper() in FAMILIES else None)
    out = []
    for p in sorted(glob.glob(os.path.join(src, "*.npz")))[::5]:
        try:
            d = np.load(p, allow_pickle=True)
            if target_fam is not None and int(d[KEY_FAMILY]) != target_fam:
                continue
            vm = np.asarray(d["von_mises"], float)
            peak = float(vm.max())
            if not (80.0 <= peak <= 700.0):      # plausible range for steel
                continue
            med = float(np.median(vm))
            ratio = peak / max(med, 1e-9)
            if not (2.5 <= ratio <= 6.5):        # field spans the scale
                continue
            # radius of the largest hole, relative to the part diagonal
            radii = []
            for k in (KEY_FIXATION_HOLES, KEY_LOAD_HOLES):
                h = np.asarray(d[k], float).reshape(-1, 8)
                if len(h):
                    radii.append(float(h[:, 6].max()))
            r = max(radii) if radii else 0.0
            nodes = np.asarray(d["nodes"], float)
            diag = float(np.linalg.norm(nodes.max(0) - nodes.min(0)))
            rel = r / max(diag, 1e-9)            # large hole relative to the part
            if rel < rmin:
                continue
            nn = int(d["n_nodes"])
            # fraction of nodes in the upper half of the scale: measures how much
            # warm color the image has
            hot = float((vm > 0.5 * peak).mean())
            score = (min(rel / 0.05, 1.5) * 2.0
                     + min(nn / 7000.0, 1.3) * 1.5
                     + min(hot / 0.08, 1.5))
            out.append((score, ratio, peak, rel, nn, hot,
                        os.path.basename(p)))
        except Exception:
            continue
    out.sort(reverse=True)
    print(f"{'score':>5s} {'ratio':>6s} {'peak':>7s} {'hole':>6s} "
          f"{'nodes':>6s} {'hot':>7s}  file")
    for score, ratio, peak, rel, nn, hot, f in out[:n]:
        print(f"{score:5.2f} {ratio:6.2f} {peak:7.1f} {rel:6.3f} {nn:6d} "
              f"{100*hot:6.1f}%  {f}")
    if not out:
        print("no candidates; loosen --min-radius or try another family")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dataset", default="dataset_v4_p2",
                    help="dataset, relative to PROJECT_DIR")
    ap.add_argument("--case", type=int, default=None)
    ap.add_argument("--file", default=None, help="name of the .npz")
    ap.add_argument("--search", default=None,
                    help="lists candidates of a family (" + ",".join(FAMILIES)
                         + "); use 'all' to sweep the whole catalogue")
    ap.add_argument("--min-radius", type=float, default=0.030,
                    help="minimum radius of the largest hole, relative to the "
                         "diagonal of the part; raise to 0.045 for really "
                         "large holes")
    ap.add_argument("--bands", type=int, default=12,
                    help="discrete colour levels; this is what gives the "
                         "contour look of commercial post-processors")
    ap.add_argument("--cmap", default="jet",
                    help="jet recalls Ansys/Abaqus; turbo is the modern "
                         "equivalent, with better transitions")
    ap.add_argument("--res", type=int, default=2600)
    ap.add_argument("--angle", type=float, default=0.0,
                    help="extra camera rotation about z (degrees)")
    ap.add_argument("--elevation", type=float, default=0.0,
                    help="extra camera elevation (degrees)")
    ap.add_argument("--zoom", type=float, default=0.98)
    ap.add_argument("--aspect", type=float, default=0.80,
                    help="window height relative to the width")
    ap.add_argument("--outline-width", type=float, default=1.6)
    ap.add_argument("--no-outline", action="store_true",
                    help="omits the black feature edges")
    ap.add_argument("--mesh", action="store_true",
                    help="draws the finite element edges on top of the "
                         "field, as commercial post-processors do")
    ap.add_argument("--mesh-width", type=float, default=0.65,
                    help="thickness of the mesh edges; thin on purpose, "
                         "so as not to compete with the field")
    ap.add_argument("--mesh-color", default="000000")
    ap.add_argument("--no-peak", action="store_true",
                    help="omits the marker of the maximum stress point")
    ap.add_argument("--arrow-dist", type=float, default=0.26,
                    help="offset of the arrow and the label, as a fraction "
                         "of the part diagonal")
    ap.add_argument("--arrow-color", default="black")
    ap.add_argument("--peak-font", type=float, default=58.0,
                    help="resolution divisor for the font of the MAX label; "
                         "smaller = larger font")
    ap.add_argument("--bar-y", type=float, default=0.025,
                    help="height of the colour bar in the window; raise it "
                         "to bring it closer to the part")
    ap.add_argument("--bar-width", type=float, default=0.46)
    ap.add_argument("--background", default="white",
                    help="white, or 'gradient' for the bluish gradient "
                         "that some solvers use")
    ap.add_argument("--html", action="store_true")
    ap.add_argument("--out", default=None,
                    help="default: PROJECT_DIR/figs/vm_hero.png")
    # option names before the translation
    for old, dest, kw in (
            ("--caso", "case", {"type": int}),
            ("--arquivo", "file", {}),
            ("--procurar", "search", {}),
            ("--raio-min", "min_radius", {"type": float}),
            ("--bandas", "bands", {"type": int}),
            ("--angulo", "angle", {"type": float}),
            ("--elevacao", "elevation", {"type": float}),
            ("--proporcao", "aspect", {"type": float}),
            ("--esp-contorno", "outline_width", {"type": float}),
            ("--sem-contorno", "no_outline", {"action": "store_true"}),
            ("--malha", "mesh", {"action": "store_true"}),
            ("--esp-malha", "mesh_width", {"type": float}),
            ("--cor-malha", "mesh_color", {}),
            ("--sem-pico", "no_peak", {"action": "store_true"}),
            ("--seta-dist", "arrow_dist", {"type": float}),
            ("--cor-seta", "arrow_color", {}),
            ("--fonte-pico", "peak_font", {"type": float}),
            ("--barra-y", "bar_y", {"type": float}),
            ("--barra-larg", "bar_width", {"type": float}),
            ("--fundo", "background", {})):
        ap.add_argument(old, dest=dest, help=argparse.SUPPRESS, **kw)
    args = ap.parse_args()

    src = os.path.join(PROJECT_DIR, os.path.expanduser(args.dataset),
                       "samples")
    if args.search:
        fam = None if args.search.lower() in SEARCH_ALL else args.search
        candidates(src, fam, rmin=args.min_radius)
        return

    if args.file:
        p = os.path.join(src, args.file)
    elif args.case is not None:
        p = os.path.join(src, f"sample_{args.case:06d}.npz")
    else:
        raise SystemExit("give --case, --file or --search")

    d = np.load(p, allow_pickle=True)
    nodes = np.ascontiguousarray(d["nodes"], np.float64)
    cells = np.asarray(d["cells"], np.int64)
    vm = np.asarray(d["von_mises"], np.float64)

    import pyvista as pv
    n_cells = cells.shape[0]
    vtk = np.empty((n_cells, 5), np.int64)
    vtk[:, 0] = 4
    vtk[:, 1:] = cells
    g = pv.UnstructuredGrid(vtk.ravel(),
                            np.full(n_cells, int(pv.CellType.TETRA), np.uint8),
                            nodes)
    g["von Mises (MPa)"] = vm
    # .clean() merges duplicate vertices; without it smooth_shading does not
    # compute consistent normals and the part comes out faceted
    surf = g.extract_surface(algorithm="dataset_surface").clean()

    pl = pv.Plotter(off_screen=not args.html, border=False,
                    window_size=(args.res, int(args.res * args.aspect)),
                    lighting="three lights")
    if args.background in GRADIENT:
        pl.set_background("white", top="C9D6E5")
    else:
        pl.set_background(args.background)

    bar_args = {"title": "von Mises (MPa)", "vertical": False,
                "width": args.bar_width, "height": 0.050,
                "position_x": (1.0 - args.bar_width) / 2.0,
                "position_y": args.bar_y, "n_labels": 6, "color": "black",
                "title_font_size": 30, "label_font_size": 26,
                "fmt": "%.0f"}
    try:
        pl.add_mesh(surf, scalars="von Mises (MPa)", cmap=args.cmap,
                    n_colors=args.bands, smooth_shading=True,
                    split_sharp_edges=True, feature_angle=32,
                    specular=0.30, specular_power=18,
                    ambient=0.26, diffuse=0.74,
                    scalar_bar_args=bar_args)
    except TypeError:
        surf = surf.compute_normals(feature_angle=32, split_vertices=True,
                                    consistent_normals=True)
        pl.add_mesh(surf, scalars="von Mises (MPa)", cmap=args.cmap,
                    n_colors=args.bands, smooth_shading=True,
                    scalar_bar_args=bar_args)

    if args.mesh:
        # on top of the field, not in its place: the mesh enters as thin edges
        pl.add_mesh(surf, style="wireframe", color=args.mesh_color,
                    line_width=args.mesh_width, lighting=False, opacity=1.0)

    if not args.no_outline:
        edges = surf.extract_feature_edges(feature_angle=28,
                                           boundary_edges=True,
                                           feature_edges=True,
                                           manifold_edges=False,
                                           non_manifold_edges=False)
        pl.add_mesh(edges, color="black", line_width=args.outline_width,
                    lighting=False)

    if not args.no_peak:
        k = int(np.argmax(vm))
        pmax = np.asarray(nodes[k], float)
        diag = float(np.linalg.norm(nodes.max(0) - nodes.min(0)))
        text = f" MAX {vm.max():.0f} MPa "
        font_px = max(14, int(args.res / args.peak_font))

        # arrow direction: out of the part, in the sense that moves the
        # label away from the center --- so it does not fall on the geometry
        center = 0.5 * (nodes.max(0) + nodes.min(0))
        u = pmax - center
        n = np.linalg.norm(u)
        u = (u / n) if n > 1e-9 else np.array([0.0, 0.0, 1.0])
        u = u + np.array([0.0, 0.0, 0.55])          # tilts upward
        u = u / np.linalg.norm(u)
        base = pmax + u * args.arrow_dist * diag

        # the arrow points from the base TO the point of maximum
        pl.add_mesh(pv.Arrow(start=base, direction=-u,
                             scale=args.arrow_dist * diag * 0.92,
                             tip_length=0.22, tip_radius=0.055,
                             shaft_radius=0.016),
                    color=args.arrow_color, lighting=False)

        for extra in (dict(shape="rounded_rect", shape_color="white",
                           shape_opacity=0.88, margin=int(args.res / 320),
                           always_visible=True, bold=True,
                           font_family="arial"),
                      dict(shape="rect", shape_color="white",
                           always_visible=True, bold=True),
                      dict(shape=None, always_visible=True, bold=True),
                      dict()):
            try:
                pl.add_point_labels(np.array([base]), [text],
                                    font_size=font_px, text_color="black",
                                    show_points=False, **extra)
                break
            except TypeError:
                continue

    try:
        pl.enable_anti_aliasing("ssaa")
    except Exception:
        pass
    pl.enable_parallel_projection()
    pl.view_isometric()
    if args.angle:
        pl.camera.azimuth += args.angle
    if args.elevation:
        pl.camera.elevation += args.elevation
    pl.reset_camera()
    pl.camera.zoom(args.zoom)

    out = args.out or os.path.join(PROJECT_DIR, "figs", "vm_hero.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if args.html:
        out = out[:-4] + ".html"
        pl.export_html(out)
    else:
        pl.show(screenshot=out)
    pl.close()

    print(f"peak {vm.max():.1f} MPa | median {np.median(vm):.1f} | "
          f"{nodes.shape[0]} nodes")
    print(out)


if __name__ == "__main__":
    main()
