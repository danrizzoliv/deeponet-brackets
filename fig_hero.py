#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OPENING FIGURE — von Mises field with the look of a commercial
post-processor.

Three things produce that look: DISCRETE color bands instead of a
continuous gradient (the classic FEA contour), smooth shading without
facets, and black feature edges.

    python fig_hero.py --dataset dataset_v4_p2 --caso 4445
    python fig_hero.py --dataset dataset_v4_p2 --procurar U   # lists candidates
    python fig_hero.py --dataset dataset_v4_p2 --caso 4445 --bandas 12 --html

The --html mode opens the scene in the browser, which renders with accelerated
WebGL; PyVista's off-screen mode under WSL uses software rasterization and looks worse.
"""

import os
import glob
import argparse
import numpy as np

BASE = os.path.expanduser("~/projetos/tcc_brackets")


def candidates(src, fam, n=14, rmin=0.0):
    """Looks for AESTHETICALLY good cases, which is different from cases with a
    high peak. A peak 60x the median leaves the whole part in dark blue with a
    red dot: the field does not span the color scale. What makes a nice image
    is a WELL-DISTRIBUTED field --- median between a fifth and a third of the
    maximum --- together with large holes and a fine mesh."""
    FAMS = {"L": 0, "Z": 1, "U": 2, "T": 3, "O": 4, "G": 5}
    target_fam = FAMS.get(fam.upper()) if fam else None
    out = []
    for p in sorted(glob.glob(os.path.join(src, "*.npz")))[::5]:
        try:
            d = np.load(p, allow_pickle=True)
            if target_fam is not None and int(d["familia"]) != target_fam:
                continue
            vm = np.asarray(d["von_mises"], float)
            peak = float(vm.max())
            if not (80.0 <= peak <= 700.0):      # plausible range for steel
                continue
            med = float(np.median(vm))
            ratio = peak / max(med, 1e-9)
            if not (2.5 <= ratio <= 6.5):        # field spans the scale
                continue
            # radius of the largest hole, and the diameter relative to the part
            radii = []
            for k in ("furos_fix", "furos_carga"):
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
    for sc, rz, pk, rl, nn, qt, f in out[:n]:
        print(f"{sc:5.2f} {rz:6.2f} {pk:7.1f} {rl:6.3f} {nn:6d} "
              f"{100*qt:6.1f}%  {f}")
    if not out:
        print("no candidates; loosen --raio-min or try another family")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dataset_v4_p2")
    ap.add_argument("--caso", type=int, default=None)
    ap.add_argument("--arquivo", default=None, help="name of the .npz")
    ap.add_argument("--procurar", default=None,
                    help="lists candidates of a family (L,Z,U,T,O,G); "
                         "use 'todas' to sweep the whole catalogue")
    ap.add_argument("--raio-min", type=float, default=0.030,
                    help="minimum radius of the largest hole, relative to the diagonal of the "
                         "part; raise to 0.045 for really large holes")
    ap.add_argument("--bandas", type=int, default=12,
                    help="discrete color levels; this is what gives the "
                         "contour look of commercial post-processors")
    ap.add_argument("--cmap", default="jet",
                    help="jet recalls Ansys/Abaqus; turbo is the modern "
                         "equivalent, with better transitions")
    ap.add_argument("--res", type=int, default=2600)
    ap.add_argument("--angulo", type=float, default=0.0)
    ap.add_argument("--elevacao", type=float, default=0.0)
    ap.add_argument("--zoom", type=float, default=0.98)
    ap.add_argument("--proporcao", type=float, default=0.80,
                    help="window height relative to the width")
    ap.add_argument("--esp-contorno", type=float, default=1.6)
    ap.add_argument("--sem-contorno", action="store_true")
    ap.add_argument("--malha", action="store_true",
                    help="draws the finite element edges on top of the "
                         "field, as commercial post-processors do")
    ap.add_argument("--esp-malha", type=float, default=0.65,
                    help="thickness of the mesh edges; thin on purpose, "
                         "so as not to compete with the field")
    ap.add_argument("--cor-malha", default="000000")
    ap.add_argument("--sem-pico", action="store_true",
                    help="omits the marker of the maximum stress point")
    ap.add_argument("--seta-dist", type=float, default=0.26,
                    help="offset of the arrow and the label, as a fraction of the "
                         "part diagonal")
    ap.add_argument("--cor-seta", default="black")
    ap.add_argument("--fonte-pico", type=float, default=58.0,
                    help="resolution divisor for the font of the MAX label; "
                         "smaller = larger font")
    ap.add_argument("--barra-y", type=float, default=0.025,
                    help="height of the color bar in the window; raise it to "
                         "bring it closer to the part")
    ap.add_argument("--barra-larg", type=float, default=0.46)
    ap.add_argument("--fundo", default="white",
                    help="white, or 'gradiente' for the bluish gradient "
                         "that some solvers use")
    ap.add_argument("--html", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    src = os.path.join(BASE, args.dataset, "samples")
    if args.procurar:
        fam = None if args.procurar.lower() == 'todas' else args.procurar
        candidates(src, fam, rmin=args.raio_min)
        return

    if args.arquivo:
        p = os.path.join(src, args.arquivo)
    elif args.caso is not None:
        p = os.path.join(src, f"sample_{args.caso:06d}.npz")
    else:
        raise SystemExit("give --caso, --arquivo or --procurar")

    d = np.load(p, allow_pickle=True)
    nodes = np.ascontiguousarray(d["nodes"], np.float64)
    cells = np.asarray(d["cells"], np.int64)
    vm = np.asarray(d["von_mises"], np.float64)

    import pyvista as pv
    nC = cells.shape[0]
    vtk = np.empty((nC, 5), np.int64)
    vtk[:, 0] = 4
    vtk[:, 1:] = cells
    g = pv.UnstructuredGrid(vtk.ravel(),
                            np.full(nC, int(pv.CellType.TETRA), np.uint8),
                            nodes)
    g["von Mises (MPa)"] = vm
    # .clean() merges duplicate vertices; without it smooth_shading does not
    # compute consistent normals and the part comes out faceted
    surf = g.extract_surface(algorithm="dataset_surface").clean()

    pl = pv.Plotter(off_screen=not args.html, border=False,
                    window_size=(args.res, int(args.res * args.proporcao)),
                    lighting="three lights")
    if args.fundo == "gradiente":
        pl.set_background("white", top="C9D6E5")
    else:
        pl.set_background(args.fundo)

    bar_args = {"title": "von Mises (MPa)", "vertical": False,
             "width": args.barra_larg, "height": 0.050,
             "position_x": (1.0 - args.barra_larg) / 2.0,
             "position_y": args.barra_y, "n_labels": 6, "color": "black",
             "title_font_size": 30, "label_font_size": 26,
             "fmt": "%.0f"}
    try:
        pl.add_mesh(surf, scalars="von Mises (MPa)", cmap=args.cmap,
                    n_colors=args.bandas, smooth_shading=True,
                    split_sharp_edges=True, feature_angle=32,
                    specular=0.30, specular_power=18,
                    ambient=0.26, diffuse=0.74,
                    scalar_bar_args=bar_args)
    except TypeError:
        surf = surf.compute_normals(feature_angle=32, split_vertices=True,
                                    consistent_normals=True)
        pl.add_mesh(surf, scalars="von Mises (MPa)", cmap=args.cmap,
                    n_colors=args.bandas, smooth_shading=True,
                    scalar_bar_args=bar_args)

    if args.malha:
        # on top of the field, not in its place: the mesh enters as thin edges
        pl.add_mesh(surf, style="wireframe", color=args.cor_malha,
                    line_width=args.esp_malha, lighting=False, opacity=1.0)

    if not args.sem_contorno:
        ar = surf.extract_feature_edges(feature_angle=28, boundary_edges=True,
                                        feature_edges=True,
                                        manifold_edges=False,
                                        non_manifold_edges=False)
        pl.add_mesh(ar, color="black", line_width=args.esp_contorno,
                    lighting=False)

    if not args.sem_pico:
        k = int(np.argmax(vm))
        pmax = np.asarray(nodes[k], float)
        diag = float(np.linalg.norm(nodes.max(0) - nodes.min(0)))
        text = f" MAX {vm.max():.0f} MPa "
        font_px = max(14, int(args.res / args.fonte_pico))

        # arrow direction: out of the part, in the sense that moves the
        # label away from the center --- so it does not fall on the geometry
        center = 0.5 * (nodes.max(0) + nodes.min(0))
        u = pmax - center
        n = np.linalg.norm(u)
        u = (u / n) if n > 1e-9 else np.array([0.0, 0.0, 1.0])
        u = u + np.array([0.0, 0.0, 0.55])          # tilts upward
        u = u / np.linalg.norm(u)
        base = pmax + u * args.seta_dist * diag

        # the arrow points from the base TO the point of maximum
        pl.add_mesh(pv.Arrow(start=base, direction=-u,
                             scale=args.seta_dist * diag * 0.92,
                             tip_length=0.22, tip_radius=0.055,
                             shaft_radius=0.016),
                    color=args.cor_seta, lighting=False)

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
    if args.angulo:
        pl.camera.azimuth += args.angulo
    if args.elevacao:
        pl.camera.elevation += args.elevacao
    pl.reset_camera()
    pl.camera.zoom(args.zoom)

    out = args.out or os.path.join(BASE, "figs", "vm_hero.png")
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
