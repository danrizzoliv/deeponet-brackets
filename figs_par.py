#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CONTROLLED PAIR — the two models on THE SAME case.

Evaluates the FIXED-GEOMETRY model (fixoU400) and the VARYING-GEOMETRY one
(varU400) on the same validation cases of the frozen-geometry set.
Identical geometry, identical load, identical material: the only
difference between the two predictions is what each model learned.

Neither saw these cases in training — those of fixo_U400 are the validation
partition of that model, and the frozen geometry does not belong to the
sampling of var_U400.

    python figs_par.py                 # table + figure of the typical case
    python figs_par.py --caso 37       # forces a case
    python figs_par.py --modo extremo  # case where the difference is largest
"""

import os
import argparse
import numpy as np
import torch

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
torch.set_num_threads(2)

from dataset_simples import case_branch, trunk_feats, denormalize_vm
from train_simples import SimpleDeepONet, predict_full

BASE = os.path.expanduser("~/projetos/tcc_brackets")
DS = os.path.join(BASE, "fixo_U400")


def load_model(ckpt):
    c = torch.load(ckpt, map_location="cpu", weights_only=False)
    a = c["args"]
    m = SimpleDeepONet(branch_dim=c["stats"]["branch_dim"], p=a.get("p", 128),
                        hidden=a.get("hidden", 128), layers=a.get("layers", 4),
                        fourier=a.get("fourier", 0),
                        d_extra=2 if a.get("dist_furos") else 0,
                        decoder=a.get("decoder", "dot"),
                        trunk=a.get("trunk", "tanh"))
    m.load_state_dict(c["model"])
    m.eval()
    return m, c["stats"], c["split"]


def predict(model, stats, path):
    """Each model uses ITS OWN normalization statistics."""
    b, s, vm, nodes, d = case_branch(path, stats)
    ft = trunk_feats(nodes, d, stats)
    if not model.d_extra:
        ft = ft[:, :3]
    vmp = np.clip(denormalize_vm(predict_full(model, b, ft, "cpu"), s, stats),
                  0, None)
    rel = float(np.linalg.norm(vmp - vm) / (np.linalg.norm(vm) + 1e-9))
    return vm, vmp, rel, nodes, d


def grid(pv, nodes, cells, name, values):
    nC = cells.shape[0]
    vtk = np.empty((nC, 5), np.int64)
    vtk[:, 0] = 4
    vtk[:, 1:] = cells
    g = pv.UnstructuredGrid(vtk.ravel(),
                            np.full(nC, int(pv.CellType.TETRA), np.uint8),
                            np.ascontiguousarray(nodes, np.float64))
    g[name] = np.asarray(values, np.float64)
    return g.extract_surface(algorithm="dataset_surface")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--caso", type=int, default=None)
    ap.add_argument("--modo", choices=["tipico", "extremo"], default="tipico")
    ap.add_argument("--res", type=int, default=2400)
    args = ap.parse_args()

    mF, stF, spF = load_model(os.path.join(BASE, "deeponet_fixoU400_e600.pt"))
    mV, stV, _ = load_model(os.path.join(BASE, "deeponet_varU400_e600.pt"))
    val = spF["val"]

    # ---- both models on all the validation cases ----
    rows = []
    for i, fb in enumerate(val):
        p = os.path.join(DS, "samples", fb)
        vm, pf, rf, _, _ = predict(mF, stF, p)
        _, pv_, rv, _, _ = predict(mV, stV, p)
        rows.append((i, rf, rv, float(vm.max()), float(pf.max()),
                       float(pv_.max())))
    A = np.array([[r[1], r[2]] for r in rows])
    print(f"\n{len(rows)} validation cases, frozen geometry\n")
    print(f"{'':22s} {'mean':>8s} {'median':>8s} {'p90':>8s}")
    for name, col in (("fixed geometry", 0), ("varying geometry", 1)):
        v = A[:, col]
        print(f"{name:22s} {v.mean():8.3f} {np.median(v):8.3f} "
              f"{np.percentile(v, 90):8.3f}")
    print(f"\nratio of the means: {A[:,1].mean()/A[:,0].mean():.2f}x")

    # ---- choice of the case to illustrate ----
    if args.caso is not None:
        values = next(r for r in rows if r[0] == args.caso)
    elif args.modo == "extremo":
        values = max(rows, key=lambda r: r[2] - r[1])
    else:
        target = np.median(A[:, 1] - A[:, 0])
        values = min(rows, key=lambda r: abs((r[2] - r[1]) - target))
    i, rf, rv, peak, pf_max, pv_max = values
    print(f"\ncase shown: #{i}   relL2 fixed {rf:.3f}   "
          f"varying {rv:.3f}   ({rv/rf:.2f}x)")
    print(f"true peak {peak:.1f} MPa -> fixed {pf_max:.1f}   "
          f"varying {pv_max:.1f} MPa")

    # ---- figures ----
    import pyvista as pv
    p = os.path.join(DS, "samples", val[i])
    vm, predF, _, nodes, d = predict(mF, stF, p)
    _, predV, _, _, _ = predict(mV, stV, p)
    cells = np.asarray(d["cells"], np.int64)
    disagree = np.abs(predF - predV)          # disagreement between the two models
    clim = [0.0, float(max(vm.max(), predF.max(), predV.max()))]
    os.makedirs(os.path.join(BASE, "figs"), exist_ok=True)

    def scalar_bar(t):
        return {"title": t, "vertical": False, "width": 0.70, "height": 0.055,
                "position_x": 0.15, "position_y": 0.025, "n_labels": 4,
                "title_font_size": 26, "label_font_size": 22, "color": "black"}

    def panel(tasks, out_path):
        pl = pv.Plotter(shape=(1, 3), off_screen=True,
                        window_size=(args.res, int(args.res * 0.52)))
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
        from inspecionar_simples import mark_cylinder_np, _arrow
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
        ctr = np.asarray(d["load_centers"], float).reshape(3)
        _arrow(pl, ctr, np.asarray(d["loads"]).reshape(3), 0.22 * diag, "red")
        _arrow(pl, ctr, np.asarray(d["moments"]).reshape(3), 0.19 * diag,
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
        ln = [f"|F| = {np.linalg.norm(F):.0f} N",
              f"|M| = {np.linalg.norm(M):.0f} N.mm",
              f"E = {E/1000.0:.0f} GPa   nu = {float(d['nu']):.3f}",
              f"{nodes.shape[0]} nodes, {cells.shape[0]} tets"]
        w = max(len(x) for x in ln)
        pl.add_text("\n".join(x.ljust(w) for x in ln),
                    position="lower_left", font_size=16)

    a1 = panel([boundary, mesh,
                 field(vm, "FEA reference (MPa)", "turbo", clim)],
                os.path.join(BASE, "figs", "par_setup.png"))
    a2 = panel([field(predF, "fixed-geometry model (MPa)", "turbo", clim),
                 field(predV, "varying-geometry model (MPa)", "turbo", clim),
                 field(disagree, "difference between models (MPa)", "hot", None)],
                os.path.join(BASE, "figs", "par_result.png"))
    print(f"\n{a1}\n{a2}")


if __name__ == "__main__":
    main()
