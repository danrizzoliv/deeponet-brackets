#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CONTROLLED PAIR: two models on the same cases.

Evaluates a fixed-geometry model (by default fixoU400) and a varying-geometry
model (by default varU400) on the validation cases of the fixed-geometry
model, taken from the frozen-geometry dataset. Geometry, load and material
are identical, so the only difference between the two predictions is what
each model learned. Neither model saw these cases in training: they are the
validation partition of the fixed-geometry model, and the frozen geometry
is not part of the sampling of the varying-geometry set. Prints the relL2
table of both models, then renders two three-panel figures of one case
(boundary conditions | mesh | FEA, and fixed | varying | difference).
Works with checkpoints of the old and the current training code.

    python figs_pair.py                  # table + figure of the typical case
    python figs_pair.py --case 37        # forces a validation case (index)
    python figs_pair.py --mode extreme   # case where the difference is largest
    python figs_pair.py --ckpt-fixed a.pt --ckpt-varying b.pt --dataset fixo_U400
"""

import os
import argparse
import numpy as np
import torch

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
torch.set_num_threads(2)

from bracket_dataset import (KEY_FIXATION_HOLES, KEY_LOAD_HOLES,
                             load_edge_map)
from train_deeponet import (load_model, checkpoint_split, predict_case,
                            relative_l2)

PROJECT_DIR = os.path.expanduser("~/projetos/tcc_brackets")  # author's project folder; adjust
DEFAULT_DATASET = "fixo_U400"
DEFAULT_CKPT_FIXED = "deeponet_fixoU400_e600.pt"
DEFAULT_CKPT_VARYING = "deeponet_varU400_e600.pt"

# values of --mode before the translation -> current values
MODE_ALIASES = {"tipico": "typical", "extremo": "extreme"}


def project_path(p):
    """Relative paths are taken inside PROJECT_DIR; absolute ones as given."""
    return os.path.join(PROJECT_DIR, os.path.expanduser(p))


# ------------------------------------------------------- data preparation

def load_pair(ckpt_fixed, ckpt_varying, dataset):
    """Loads both models and, if either uses the edge distance, the edge
    map of `dataset`. Returns (models, val_files, edge_map), where
    models = [(model, stats, cfg), ...] and val_files are the full paths of
    the validation cases of the fixed-geometry model."""
    models, ckpts = [], []
    for path in (ckpt_fixed, ckpt_varying):
        model, ckpt, cfg = load_model(path, "cpu")
        models.append((model, ckpt["stats"], cfg))
        ckpts.append(ckpt)
    val = checkpoint_split(ckpts[0])["val"]
    val_files = [os.path.join(dataset, "samples", f) for f in val]
    edge_map = None
    if any(cfg.get("edge_distance") for _, _, cfg in models):
        edge_map, _ = load_edge_map(val_files)
    return models, val_files, edge_map


def predict(entry, path, edge_map):
    """Each model uses its own normalization statistics.
    Returns (vm_true, vm_pred, relL2, nodes, data)."""
    model, stats, cfg = entry
    vm, vm_pred, nodes, d = predict_case(model, path, stats, cfg, "cpu",
                                         edge_map)
    return vm, vm_pred, relative_l2(vm_pred, vm), nodes, d


def evaluate_pair(models, val_files, edge_map):
    """Both models on every validation case. One row per case:
    (index, relL2 fixed, relL2 varying, true peak, fixed peak, varying peak)."""
    rows = []
    for i, p in enumerate(val_files):
        vm, pred_f, rel_f, _, _ = predict(models[0], p, edge_map)
        _, pred_v, rel_v, _, _ = predict(models[1], p, edge_map)
        rows.append((i, rel_f, rel_v, float(vm.max()), float(pred_f.max()),
                     float(pred_v.max())))
    return rows


def print_table(rows):
    errs = np.array([[r[1], r[2]] for r in rows])
    print(f"\n{len(rows)} validation cases, frozen geometry\n")
    print(f"{'':22s} {'mean':>8s} {'median':>8s} {'p90':>8s}")
    for name, col in (("fixed geometry", 0), ("varying geometry", 1)):
        v = errs[:, col]
        print(f"{name:22s} {v.mean():8.3f} {np.median(v):8.3f} "
              f"{np.percentile(v, 90):8.3f}")
    print(f"\nratio of the means: {errs[:,1].mean()/errs[:,0].mean():.2f}x")


def choose_case(rows, case=None, mode="typical"):
    """The row to illustrate: a forced case, the one with the largest
    difference between the models ('extreme'), or the one whose difference
    is closest to the median difference ('typical')."""
    errs = np.array([[r[1], r[2]] for r in rows])
    if case is not None:
        return next(r for r in rows if r[0] == case)
    if mode == "extreme":
        return max(rows, key=lambda r: r[2] - r[1])
    target = np.median(errs[:, 1] - errs[:, 0])
    return min(rows, key=lambda r: abs((r[2] - r[1]) - target))


# --------------------------------------------------------------- rendering

def mark_cylinder(nodes, h):
    """h = [cx,cy,cz, ax,ay,az, r, half] -> mask of the nodes on the hole
    shell. Same tolerance as the marker of inspect_cases.py."""
    c = np.array(h[:3], float)
    axis = np.array(h[3:6], float)
    axis = axis / (np.linalg.norm(axis) + 1e-30)
    radius, half = float(h[6]), float(h[7])
    d = nodes - c
    proj = d @ axis
    radial = np.linalg.norm(d - np.outer(proj, axis), axis=1)
    return ((np.abs(radial - radius) < 0.35 * radius + 0.5)
            & (np.abs(proj) < half + 1.0))


def arrow(pl, origin, direction, length, color, double=False):
    """Force arrow; `double` adds the second head of a moment arrow.
    Same geometry as the arrows of inspect_cases.py."""
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


def grid(pv, nodes, cells, name, values):
    n_cells = cells.shape[0]
    vtk = np.empty((n_cells, 5), np.int64)
    vtk[:, 0] = 4
    vtk[:, 1:] = cells
    g = pv.UnstructuredGrid(vtk.ravel(),
                            np.full(n_cells, int(pv.CellType.TETRA), np.uint8),
                            np.ascontiguousarray(nodes, np.float64))
    g[name] = np.asarray(values, np.float64)
    return g.extract_surface(algorithm="dataset_surface")


def render(vm, pred_f, pred_v, nodes, d, out_dir, res):
    """Saves pair_setup.png and pair_result.png in `out_dir`."""
    import pyvista as pv
    cells = np.asarray(d["cells"], np.int64)
    disagree = np.abs(pred_f - pred_v)        # disagreement between the two models
    clim = [0.0, float(max(vm.max(), pred_f.max(), pred_v.max()))]
    os.makedirs(out_dir, exist_ok=True)

    def scalar_bar(t):
        return {"title": t, "vertical": False, "width": 0.70, "height": 0.055,
                "position_x": 0.15, "position_y": 0.025, "n_labels": 4,
                "title_font_size": 26, "label_font_size": 22, "color": "black"}

    def panel(tasks, out_path):
        pl = pv.Plotter(shape=(1, 3), off_screen=True,
                        window_size=(res, int(res * 0.52)))
        for k, t in enumerate(tasks):
            pl.subplot(0, k)
            t(pl)
        pl.link_views()
        pl.reset_camera()
        pl.camera.zoom(0.80)
        pl.show(screenshot=out_path)
        pl.close()
        return out_path

    def field(data, title, colormap, cl):
        def _f(pl):
            pl.add_text(title, font_size=20)
            pl.add_mesh(grid(pv, nodes, cells, title, data), scalars=title,
                        cmap=colormap, clim=cl, smooth_shading=False,
                        scalar_bar_args=scalar_bar(title))
        return _f

    def boundary(pl):
        pl.add_text("Boundary conditions", font_size=20)
        surf = grid(pv, nodes, cells, "x", np.zeros(nodes.shape[0]))
        pl.add_mesh(surf, color="lightgray", opacity=0.55,
                    smooth_shading=False)
        fix_mask = np.zeros(nodes.shape[0], bool)
        for h in np.asarray(d[KEY_FIXATION_HOLES], float).reshape(-1, 8):
            fix_mask |= mark_cylinder(nodes, h)
        load_mask = np.zeros(nodes.shape[0], bool)
        for h in np.asarray(d[KEY_LOAD_HOLES], float).reshape(-1, 8):
            load_mask |= mark_cylinder(nodes, h)
        if fix_mask.any():
            pl.add_points(nodes[fix_mask], color="royalblue", point_size=7,
                          render_points_as_spheres=True)
        if load_mask.any():
            pl.add_points(nodes[load_mask], color="orange", point_size=7,
                          render_points_as_spheres=True)
        diag = float(np.linalg.norm(nodes.max(0) - nodes.min(0)))
        ctr = np.asarray(d["load_centers"], float).reshape(3)
        arrow(pl, ctr, np.asarray(d["loads"]).reshape(3), 0.22 * diag, "red")
        arrow(pl, ctr, np.asarray(d["moments"]).reshape(3), 0.19 * diag,
              "purple", double=True)
        pl.add_text("blue = clamped   orange = loaded\n"
                    "red arrow = F    double purple = M",
                    position="lower_left", font_size=16)

    def mesh(pl):
        pl.add_text("Mesh", font_size=20)
        pl.add_mesh(grid(pv, nodes, cells, "x", np.zeros(nodes.shape[0])),
                    color="white", show_edges=True, edge_color="gray",
                    line_width=0.5)
        F = np.asarray(d["loads"], float).reshape(3)
        M = np.asarray(d["moments"], float).reshape(3)
        E = next((float(d[k]) for k in ("E", "young", "E_MPa")
                  if k in d.files), float("nan"))
        lines = [f"|F| = {np.linalg.norm(F):.0f} N",
                 f"|M| = {np.linalg.norm(M):.0f} N.mm",
                 f"E = {E/1000.0:.0f} GPa   nu = {float(d['nu']):.3f}",
                 f"{nodes.shape[0]} nodes, {cells.shape[0]} tets"]
        w = max(len(x) for x in lines)
        pl.add_text("\n".join(x.ljust(w) for x in lines),
                    position="lower_left", font_size=16)

    out1 = panel([boundary, mesh,
                  field(vm, "FEA reference (MPa)", "turbo", clim)],
                 os.path.join(out_dir, "pair_setup.png"))
    out2 = panel([field(pred_f, "fixed-geometry model (MPa)", "turbo", clim),
                  field(pred_v, "varying-geometry model (MPa)", "turbo", clim),
                  field(disagree, "difference between models (MPa)", "hot",
                        None)],
                 os.path.join(out_dir, "pair_result.png"))
    return out1, out2


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--case", type=int, default=None,
                    help="index of the validation case to show")
    ap.add_argument("--caso", dest="case", type=int, help=argparse.SUPPRESS)
    ap.add_argument("--mode", type=lambda s: MODE_ALIASES.get(s, s),
                    choices=["typical", "extreme"], default="typical")
    ap.add_argument("--modo", dest="mode",
                    type=lambda s: MODE_ALIASES.get(s, s),
                    choices=["typical", "extreme"], help=argparse.SUPPRESS)
    ap.add_argument("--res", type=int, default=2400)
    ap.add_argument("--ckpt-fixed", default=DEFAULT_CKPT_FIXED,
                    help="fixed-geometry model (relative to PROJECT_DIR)")
    ap.add_argument("--ckpt-varying", default=DEFAULT_CKPT_VARYING,
                    help="varying-geometry model (relative to PROJECT_DIR)")
    ap.add_argument("--dataset", default=DEFAULT_DATASET,
                    help="frozen-geometry dataset (relative to PROJECT_DIR)")
    ap.add_argument("--out-dir", default=None,
                    help="default: PROJECT_DIR/figs")
    args = ap.parse_args()

    dataset = project_path(args.dataset)
    models, val_files, edge_map = load_pair(project_path(args.ckpt_fixed),
                                            project_path(args.ckpt_varying),
                                            dataset)

    # ---- both models on all the validation cases ----
    rows = evaluate_pair(models, val_files, edge_map)
    print_table(rows)

    # ---- choice of the case to illustrate ----
    i, rel_f, rel_v, peak, peak_f, peak_v = choose_case(rows, args.case,
                                                        args.mode)
    print(f"\ncase shown: #{i}   relL2 fixed {rel_f:.3f}   "
          f"varying {rel_v:.3f}   ({rel_v/rel_f:.2f}x)")
    print(f"true peak {peak:.1f} MPa -> fixed {peak_f:.1f}   "
          f"varying {peak_v:.1f} MPa")

    # ---- figures ----
    p = val_files[i]
    vm, pred_f, _, nodes, d = predict(models[0], p, edge_map)
    _, pred_v, _, _, _ = predict(models[1], p, edge_map)
    out_dir = args.out_dir or os.path.join(PROJECT_DIR, "figs")
    out1, out2 = render(vm, pred_f, pred_v, nodes, d, out_dir, args.res)
    print(f"\n{out1}\n{out2}")


if __name__ == "__main__":
    main()
