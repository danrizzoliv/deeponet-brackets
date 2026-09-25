#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SIMPLE PARAMETRIC BRACKET GENERATOR — L, Z and U — v2: HOLES + FILLETS.

Like real engineering brackets:
  - FIXATION HOLES in the base plate (1, 2 or 4, as they fit): the clamp is
    applied on the CYLINDRICAL SHELL of the holes (RBE2 style, as in SimJEB);
  - LOAD HOLE in the free member: F and M applied on the hole shell with the
    exact traction t = F/S + a×r (RBE3 style, as in SimJEB);
  - FILLETS at the concave corners between members (sampled radius), removing
    the sharp-corner singularity; if OCC rejects the fillet for a combination
    of dimensions, it tries a smaller radius and, as a last resort, proceeds
    without (RF=0 recorded in geo_params).

Replaces SimJEB's CAD by gmsh primitives (fused boxes), keeping all the rest
of the validated pipeline: CG+GAMG solver with nullspace, moment applied as
traction t(x) = F/S + a × r with a = K⁻¹M (zero force, exact moment),
variable material and compatible .npz format.

Each sample draws: family (L/Z/U), dimensions (A, B, C, W, T), load
(F, M), material (E, nu). EACH SAMPLE HAS ITS OWN MESH (the geometry
changes), so nodes/cells go inside each .npz.

Canonical boundary conditions (the same in all families):
  - CLAMP: lower face (z=0) of the base plate;
  - LOAD: free face of the last member (top of the vertical flange in the L,
    tip of the upper flange in the Z, top of the second wall in the U), with F and M.

Geometry saved in the .npz for the DeepONet branch:
  familia (0=L, 1=Z, 2=U), geo_params = [A, B, C, W, T] in mm,
  and the boxes of the BC faces (for visual inspection): fixed_box, load_box.

Usage:
    python gen_simples.py --out-dir ~/projetos/simples/dataset_v1 \
        --n 1000 --seed 0 [--check-lin]
"""

import os
import csv
import json
import gc
import argparse
import numpy as np

from mpi4py import MPI
import ufl
from dolfinx import fem, plot, default_scalar_type, mesh as dmesh
from dolfinx.io import gmsh as gmshio
from dolfinx.fem.petsc import LinearProblem
from petsc4py import PETSc

COMM = MPI.COMM_SELF
FIXED_MARKER, LOAD_MARKER, VOLUME_MARKER = 1, 10, 99

ITER_OPTS = {
    "ksp_type": "cg",
    "pc_type": "gamg",
    "ksp_rtol": 1e-8,
    "ksp_max_it": 500,
    "mg_levels_ksp_type": "chebyshev",
    "mg_levels_pc_type": "jacobi",
}

FAMILIES = ["L", "Z", "U", "T", "O", "G", "E",
            "F", "X", "J", "W", "S"]


def rigid_body_nullspace(V):
    bs = V.dofmap.index_map_bs
    basis = [fem.Function(V) for _ in range(6)]
    x = V.tabulate_dof_coordinates()
    for d in range(3):
        arr = basis[d].x.array
        arr[:] = 0.0
        arr[d::bs] = 1.0
    rots = [(1, 2), (0, 2), (0, 1)]
    signs = [(-1.0, 1.0), (1.0, -1.0), (-1.0, 1.0)]
    for r, ((a, b), (sa, sb)) in enumerate(zip(rots, signs)):
        arr = basis[3 + r].x.array
        arr[:] = 0.0
        arr[a::bs] = sa * x[:, b]
        arr[b::bs] = sb * x[:, a]
    vecs = [b.x.petsc_vec for b in basis]
    for i in range(6):
        for j in range(i):
            vecs[i].axpy(-vecs[i].dot(vecs[j]), vecs[j])
        vecs[i].normalize()
    return PETSc.NullSpace().create(vectors=vecs, comm=V.mesh.comm), basis


# --------------------------------------------------------- bracket families
def sample_geometry(rng, forced_fam=None):
    """Samples family, dimensions, holes and fillet.
    Families: 0=L, 1=Z, 2=U(clevis), 3=T(inner wall), 4=O(omega/hat),
    5=G(ribbed L/triangular gusset), 6=E(bent-sheet channel with notches
    and a large hole in the web).
    Returns (fam, params(11), boxes, fillet_edges, fixed_holes, load_holes):
      params = [A, B, C, W, T, RF, RB, DF, DL, NF, NL]; C is reused per
      family (Z: flange depth; T: wall position; O: foot
      length; G: rib depth; C: flange length; A: angle in
      degrees; B: tube length; D: cradle radius; K: tie-rod
      length; X: length of the 2nd wall; L/U: 0).
      *_holes = dicts {center, axis, radius, half} (cutting/BC cylinders)
      fillet_edges = (x0, z0) for concave lines along y (backward
      compatible) or ("x"|"y"|"z", c1, c2, len_min) for an arbitrary direction.
    """
    fam = int(rng.integers(0, 12))          # always consumes the draw, to
    if forced_fam is not None:            # keep the rest of the random
        fam = int(forced_fam)             # stream identical by index
    S = float(rng.uniform(60.0, 140.0))
    T = float(rng.uniform(0.08, 0.13) * S)             # thickness
    W = float(rng.uniform(0.45, 0.80) * S)             # width
    A = float(rng.uniform(0.75, 1.10) * S)             # base member
    B = float(rng.uniform(0.75, 1.10) * S)             # vertical member
    RF = float(rng.uniform(0.40, 0.80) * T)            # concave fillet
    RB = float(min(rng.uniform(0.20, 0.35) * T, 0.30 * T))  # edge break
    DF = float(min(rng.uniform(0.50, 1.40) * T, 0.26 * W))   # fixation diam.
    DL = float(min(rng.uniform(1.00, 1.80) * DF, 0.30 * W))  # load diam.

    def cil(cx, cy, cz, ax, ay, az, r, half):
        return {"center": [cx, cy, cz], "axis": [ax, ay, az],
                "radius": r, "half": half}

    ys = [0.30 * W, 0.70 * W] if W >= 0.55 * 100.0 else [0.5 * W]
    C = 0.0
    mirror = None                          # (x_mirror) for clevis/saddle
    span = np.array([0, 1, 0], float)       # direction of the load group
    extent = W                              # usable extent for the group
    half_l = T                              # half-height of the load cylinder
    cuts = []                             # cutting cylinders (no BC)
    cut_boxes = []                         # cutting boxes (notches/window)
    add_cyl = []                           # additive cylinders (boss/lug)
    fix_custom = None                       # own fixation holes
    nl_fixed = None                          # NL imposed by the family
    # fixation regions: list of (x_lo, x_hi); one column per region
    if fam == 0:                            # L: base + wall at x=0
        boxes = [(0, 0, 0, A, W, T), (0, 0, 0, T, W, B)]
        edges = [(T, T)]
        load_axis = [1, 0, 0]
        base_c = np.array([T / 2, W / 2, B - max(1.4 * DL, 0.15 * B)])
        mx = max(0.12 * A, 1.3 * DF, RF + DF)
        regions = [(T + mx, A - mx)]
    elif fam == 1:                          # Z: lower flange + web + upper flange
        C = float(rng.uniform(0.45, 0.90) * S)
        DL = float(min(DL, C / 2.2))
        boxes = [(0, 0, 0, A, W, T), (0, 0, 0, T, W, B),
                  (T - C, 0, B - T, C, W, T)]
        edges = [(T, T), (0.0, B - T)]
        load_axis = [0, 0, 1]
        base_c = np.array([(T - C) + max(1.4 * DL, 0.15 * C), W / 2,
                           B - T / 2])
        mx = max(0.12 * A, 1.3 * DF, RF + DF)
        regions = [(T + mx, A - mx)]
    elif fam == 2:                          # U: clevis (2 walls)
        boxes = [(0, 0, 0, A, W, T), (0, 0, 0, T, W, B),
                  (A - T, 0, 0, T, W, B)]
        edges = [(T, T), (A - T, T)]
        load_axis = [1, 0, 0]
        base_c = np.array([T / 2, W / 2, B - max(1.4 * DL, 0.15 * B)])
        mirror = A - T / 2
        mx = max(0.12 * A, 1.3 * DF, RF + DF)
        regions = [(T + mx, A - T - mx)]
    elif fam == 3:                          # T: INNER wall on the base
        C = float(rng.uniform(0.35, 0.60) * A)      # wall position
        # DF must fit in both spans of the base:
        DF = float(min(DF, 0.20 * min(C, A - C - T)))
        DL = float(min(DL, 0.30 * W))
        boxes = [(0, 0, 0, A, W, T), (C, 0, 0, T, W, B)]
        edges = [(C, T), (C + T, T)]
        load_axis = [1, 0, 0]
        base_c = np.array([C + T / 2, W / 2, B - max(1.4 * DL, 0.15 * B)])
        mx = max(0.10 * A, 1.3 * DF, RF + DF)
        regions = [(mx, C - mx), (C + T + mx, A - mx)]
    elif fam == 4:                          # O: omega/hat (feet + bridge)
        C = float(min(rng.uniform(0.20, 0.30) * A,
                      0.5 * (A - 2 * T - 0.30 * A)))  # foot length
        DF = float(min(DF, 0.22 * C))
        boxes = [(0, 0, 0, C + T, W, T),
                  (A - C - T, 0, 0, C + T, W, T),
                  (C, 0, 0, T, W, B),
                  (A - C - T, 0, 0, T, W, B),
                  (C, 0, B - T, A - 2 * C, W, T)]
        edges = [(C, T), (A - C, T),
                   (C + T, B - T), (A - C - T, B - T)]
        load_axis = [0, 0, 1]
        base_c = np.array([A / 2, W / 2, B - T / 2])
        mx = max(0.20 * C, 1.3 * DF, RF + DF)
        regions = [(mx, C - mx), (A - C + mx, A - mx)]
    elif fam == 5:                          # G: ribbed L (gusset)
        factor = float(rng.uniform(0.35, 0.55))
        CR = factor * (A - T)                # rib depth (x)
        HR = factor * (B - T)                # rib height (z)
        C = CR
        tr = 0.85 * T                       # rib thickness (y)
        # fixation holes kept away from the rib (which occupies y=W/2 +- tr/2):
        ys = [0.20 * W, 0.80 * W]
        DF = float(min(DF, 0.24 * W,
                       1.8 * (0.30 * W - 0.5 * tr - 0.3 * T)))
        # load hole on the wall above the top of the rib:
        DL = float(min(DL, (B - T - HR - 0.5 * RF - 0.05 * B) / 1.9))
        # TRIANGULAR rib (knee brace): wedge with the right angle at the
        # base-wall corner, tapering until it meets the wall at the top
        boxes = [(0, 0, 0, A, W, T), (0, 0, 0, T, W, B),
                  (T, W / 2 - tr / 2, T, CR, tr, HR, 0.0)]
        edges = [("y", T, T, 0.12 * W),
                   ("x", W / 2 - tr / 2, T, 1.5 * T),
                   ("x", W / 2 + tr / 2, T, 1.5 * T),
                   ("z", T, W / 2 - tr / 2, 1.5 * T),
                   ("z", T, W / 2 + tr / 2, 1.5 * T)]
        load_axis = [1, 0, 0]
        base_c = np.array([T / 2, W / 2, B - max(1.4 * DL, 0.15 * B)])
        mx = max(0.12 * A, 1.3 * DF, RF + DF)
        regions = [(T + mx, A - mx)]

    elif fam == 6:                          # E: channel with NOTCHES (sheet)
        # SHEET METAL proportions: thin wall, large area. Thickness
        # calibrated to keep the mesh cost feasible (~2 elements through the
        # thickness => nodes ~ 1/T^3).
        T = float(rng.uniform(0.060, 0.090) * S)
        W = float(rng.uniform(0.80, 1.15) * S)       # web width
        B = float(rng.uniform(0.85, 1.25) * S)       # web height
        RF = float(rng.uniform(1.0, 2.0) * T)        # bend radius
        RB = float(rng.uniform(0.15, 0.30) * T)
        DF = float(min(rng.uniform(0.9, 1.8) * T, 0.16 * W))
        Fl = float(max(rng.uniform(0.30, 0.48) * W, 4.5 * T))
        A = Fl
        DL = float(rng.uniform(0.26, 0.40) * min(W - 2 * T, B))
        Wn = float(rng.uniform(0.30, 0.48) * W)      # notch width
        Hn = float(rng.uniform(0.10, 0.20) * B)
        C = Wn
        boxes = [(0, 0, 0, T, W, B),                # web
                  (0, 0, 0, Fl, T, B), (0, W - T, 0, Fl, T, B)]   # flanges
        cut_boxes = [(-0.2 * T, 0.5 * (W - Wn), B - Hn,
                       1.4 * T, Wn, 1.3 * Hn),
                      (-0.2 * T, 0.5 * (W - Wn), -0.3 * Hn,
                       1.4 * T, Wn, 1.3 * Hn)]
        edges = [("z", T, T, 0.4 * B), ("z", T, W - T, 0.4 * B)]
        load_axis = [1, 0, 0]
        base_c = np.array([T / 2, W / 2, B / 2])
        nl_fixed = 1
        DF = float(min(DF, 0.55 * (Fl - T), 0.22 * B))
        xf = T + 0.55 * (Fl - T)
        fix_custom = [cil(xf, yy, zz, 0, 1, 0, DF / 2, T)
                      for yy in (T / 2, W - T / 2)
                      for zz in (0.20 * B, 0.80 * B)]

    elif fam == 7:                          # F: FLAT PLATE (link/tie rod)
        T = float(rng.uniform(0.050, 0.085) * S)
        RF = 0.0                                     # no bend: no corner
        RB = float(rng.uniform(0.15, 0.30) * T)
        DF = float(min(rng.uniform(0.9, 1.8) * T, 0.22 * W))
        DL = float(min(rng.uniform(1.2, 2.2) * DF, 0.30 * W))
        boxes = [(0, 0, 0, A, W, T)]
        edges = []
        load_axis = [0, 0, 1]
        base_c = np.array([0.82 * A, W / 2, T / 2])
        B = T
        C = 0.0
        mx = max(0.08 * A, 1.2 * DF)
        regions = [(mx, 0.32 * A)]
    elif fam == 8:                          # X: angle with CHAMFERED corners
        T = float(rng.uniform(0.050, 0.085) * S)
        Bw = float(rng.uniform(0.55, 0.95) * B)
        ch = float(rng.uniform(0.15, 0.28) * min(A, W))   # chamfer leg
        RF = float(rng.uniform(0.4, 0.8) * T)
        RB = float(rng.uniform(0.15, 0.30) * T)
        DF = float(min(rng.uniform(0.9, 1.8) * T, 0.20 * W))
        DL = float(min(rng.uniform(1.2, 2.2) * DF, 0.26 * W))
        boxes = [(0, 0, 0, A, W, T), (0, 0, 0, T, W, Bw),
                  {"wedge": (0, 0, 0, ch, ch, 2 * T, 0.0), "cut": True,
                   "rot": (0, 0, 1, np.pi / 2), "pivot": (0, 0, 0),
                   "trans": (A, 0, -0.5 * T)},
                  {"wedge": (0, 0, 0, ch, ch, 2 * T, 0.0), "cut": True,
                   "rot": (0, 0, 1, np.pi), "pivot": (0, 0, 0),
                   "trans": (A, W, -0.5 * T)}]
        edges = [("y", T, T, 0.4 * W)]
        load_axis = [1, 0, 0]
        base_c = np.array([T / 2, W / 2, Bw - max(1.4 * DL, 0.16 * Bw)])
        B = Bw
        C = ch
        mx = max(0.12 * A, 1.3 * DF, RF + DF)
        regions = [(T + mx, A - mx)]
    elif fam == 9:                          # J: angle with a WINDOW in the web
        T = float(rng.uniform(0.050, 0.085) * S)
        Bw = float(rng.uniform(0.65, 1.05) * B)
        Rj = float(rng.uniform(0.16, 0.26) * min(W, Bw))   # opening radius
        zj = T + float(rng.uniform(0.35, 0.55)) * (Bw - T)
        RF = float(rng.uniform(0.4, 0.8) * T)
        RB = float(rng.uniform(0.15, 0.30) * T)
        DF = float(min(rng.uniform(0.9, 1.8) * T, 0.20 * W))
        DL = float(min(rng.uniform(1.2, 2.2) * DF, 0.24 * W,
                       0.80 * (Bw - zj - Rj)))
        boxes = [(0, 0, 0, A, W, T), (0, 0, 0, T, W, Bw)]
        cuts = [cil(T / 2, W / 2, zj, 1, 0, 0, Rj, 2.0 * T)]
        edges = [("y", T, T, 0.4 * W)]
        load_axis = [1, 0, 0]
        base_c = np.array([T / 2, W / 2, Bw - max(1.5 * DL, 0.10 * Bw)])
        B = Bw
        C = Rj
        mx = max(0.12 * A, 1.3 * DF, RF + DF)
        regions = [(T + mx, A - mx)]
    elif fam == 10:                         # W: flat plate with a central WINDOW
        T = float(rng.uniform(0.050, 0.085) * S)
        Lw = float(rng.uniform(0.28, 0.42) * A)      # window length
        Ww = float(rng.uniform(0.35, 0.55) * W)      # window width
        Rw = float(rng.uniform(0.15, 0.30) * min(Lw, Ww))   # corner radius
        xw, yw = 0.52 * A, 0.5 * W
        RF = 0.0
        RB = float(rng.uniform(0.15, 0.30) * T)
        DF = float(min(rng.uniform(0.9, 1.8) * T, 0.22 * W))
        DL = float(min(rng.uniform(1.2, 2.2) * DF, 0.30 * W,
                       0.55 * (A - xw - Lw / 2)))
        boxes = [(0, 0, 0, A, W, T)]
        cut_boxes = [(xw - Lw / 2, yw - Ww / 2 + Rw, -0.2 * T,
                       Lw, Ww - 2 * Rw, 1.4 * T),
                      (xw - Lw / 2 + Rw, yw - Ww / 2, -0.2 * T,
                       Lw - 2 * Rw, Ww, 1.4 * T)]
        cuts = [cil(xw + sx * (Lw / 2 - Rw), yw + sy * (Ww / 2 - Rw),
                      T / 2, 0, 0, 1, Rw, T)
                  for sx in (-1.0, 1.0) for sy in (-1.0, 1.0)]
        edges = []
        load_axis = [0, 0, 1]
        base_c = np.array([xw + Lw / 2 + 0.5 * (A - xw - Lw / 2), W / 2,
                           T / 2])
        B = T
        C = Lw
        mx = max(0.07 * A, 1.2 * DF)
        regions = [(mx, xw - Lw / 2 - mx)]
    else:                                   # S: angle with a STEPPED base
        T = float(rng.uniform(0.050, 0.085) * S)
        Bw = float(rng.uniform(0.60, 1.00) * B)
        Ld = float(rng.uniform(0.28, 0.45) * A)      # step length
        Wd = float(rng.uniform(0.30, 0.50) * W)      # recess width
        RF = float(rng.uniform(0.4, 0.8) * T)
        RB = float(rng.uniform(0.15, 0.30) * T)
        DF = float(min(rng.uniform(0.9, 1.8) * T, 0.20 * (W - Wd)))
        DL = float(min(rng.uniform(1.2, 2.2) * DF, 0.26 * W))
        boxes = [(0, 0, 0, A, W, T), (0, 0, 0, T, W, Bw)]
        # rectangular recess at the free end of the base (high-y corner)
        cut_boxes = [(A - Ld, W - Wd, -0.2 * T, Ld + 0.1 * A, Wd + 0.1 * W,
                       1.4 * T)]
        edges = [("y", T, T, 0.4 * W)]
        load_axis = [1, 0, 0]
        base_c = np.array([T / 2, W / 2, Bw - max(1.4 * DL, 0.16 * Bw)])
        B = Bw
        C = Ld
        mx = max(0.12 * A, 1.3 * DF, RF + DF)
        # fixation column in the strip left intact + one in the full stretch
        ys = [0.5 * (W - Wd) * 0.55, 0.5 * (W - Wd) * 1.35]
        regions = [(T + mx, A - mx)]

    # ---- fixation holes: 1 x-column per region (2 if the region is long
    # and unique), on the y-rows of `ys` ----
    if fix_custom is not None:
        fixed_holes = list(fix_custom)
        NF = float(len(fixed_holes))
        xs = None
    else:
        xs = []
    if xs is not None and len(regions) > 1:
        # families with 2 spans (T, O): ALWAYS 1 column at the center of each
        # span — fixation guaranteed on both sides of the wall/bridge
        xs = [0.5 * (lo + hi) for (lo, hi) in regions]
    elif xs is not None:
        lo, hi = regions[0]
        xs = [lo, hi] if hi - lo >= 2.2 * DF else [0.5 * (lo + hi)]
    if xs is not None:
        fixed_holes = [cil(xf, yf, T / 2, 0, 0, 1, DF / 2, T)
                     for xf in xs for yf in ys]
        NF = float(len(fixed_holes))

    # ---- load holes: uniform group along y, centered at base_c
    NL = int(rng.integers(1, 4)) if nl_fixed is None else int(nl_fixed)
    if NL > 1:
        DL = float(min(DL, 0.45 * extent / NL))
    pitch = min(2.2 * DL, 0.7 * extent / max(NL, 1))
    offs = (np.arange(NL) - (NL - 1) / 2.0) * pitch
    load_holes = []
    for o in offs:
        c = base_c + span * o
        load_holes.append({"center": [float(c[0]), float(c[1]),
                                       float(c[2])],
                            "axis": [float(a) for a in load_axis],
                            "radius": DL / 2, "half": half_l})
    NL = float(len(load_holes))
    if mirror is not None:
        # U is a clevis: the "pin" crosses BOTH walls -> mirrors the
        # group on the opposite wall; the RBE3 splits F and M over the whole set.
        load_holes += [{"center": [mirror, h["center"][1], h["center"][2]],
                         "axis": list(h["axis"]), "radius": h["radius"],
                         "half": h["half"]} for h in load_holes]
        NL = float(len(load_holes))

    boxes = (list(boxes) + [{"cyl": c} for c in add_cyl]
              + [{"cut_cyl": h} for h in cuts]
              + [{"cut_box": b} for b in cut_boxes])
    params = np.array([A, B, C, W, T, RF, RB, DF, DL, NF, NL], np.float64)
    return fam, params, boxes, edges, fixed_holes, load_holes


def build_mesh(boxes, edges, fixed_holes, load_holes, T, W, RF, RB,
                    h_factor=1.0):
    """Fuses boxes, applies fillets on the concave edges, cuts the holes and
    generates the mesh (refined by curvature at the holes/fillets)."""
    import gmsh
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("b")
        def _eh(c, k):
            return isinstance(c, dict) and k in c

        def _prim(spec):
            """creates box / wedge / cylinder, with optional rotation and translation."""
            if isinstance(spec, dict):
                if "wedge" in spec:
                    t = gmsh.model.occ.addWedge(*spec["wedge"][:6],
                                                ltx=spec["wedge"][6])
                elif "cyl" in spec:
                    t = gmsh.model.occ.addCylinder(*spec["cyl"])
                else:
                    t = gmsh.model.occ.addBox(*spec["box"])
                if "rot" in spec:
                    ax_, ay_, az_, ang_ = spec["rot"]
                    px_, py_, pz_ = spec.get("pivot", (0.0, 0.0, 0.0))
                    gmsh.model.occ.rotate([(3, t)], px_, py_, pz_,
                                          ax_, ay_, az_, float(ang_))
                if "trans" in spec:
                    gmsh.model.occ.translate([(3, t)], *spec["trans"])
                return t
            if len(spec) == 6:
                return gmsh.model.occ.addBox(*spec)
            # native wedge reoriented (triangular rib): triangle in x-z
            xw_, yw_, zw_, dxw, dyw, dzw, ltxw = spec
            t = gmsh.model.occ.addWedge(0, 0, 0, dxw, dzw, dyw, ltx=ltxw)
            gmsh.model.occ.rotate([(3, t)], 0, 0, 0, 1, 0, 0,
                                  float(np.pi / 2))
            gmsh.model.occ.translate([(3, t)], xw_, yw_ + dyw, zw_)
            return t

        def _corte(c):
            return (_eh(c, "cut_cyl") or _eh(c, "cut_box")
                    or (isinstance(c, dict) and c.get("cut")))

        adds = [c for c in boxes if not _corte(c)]
        cuts = [c["cut_cyl"] for c in boxes if _eh(c, "cut_cyl")]
        cut_boxes = [c["cut_box"] for c in boxes if _eh(c, "cut_box")]
        cut_specs = [c for c in boxes
                       if isinstance(c, dict) and c.get("cut")
                       and not (_eh(c, "cut_cyl") or _eh(c, "cut_box"))]
        tags = [_prim(c) for c in adds]
        base = [(3, tags[0])]
        for t in tags[1:]:
            base, _ = gmsh.model.occ.fuse(base, [(3, t)])
        gmsh.model.occ.synchronize()
        vol = [v for v in gmsh.model.getEntities(3)][0][1]

        # locate the concave edges: (x0,z0)=line along y
        # (backward compatible) or ("x"|"y"|"z", c1, c2, len_min)
        tol = 1e-4 * max(T, 1.0) + 1e-6
        targets = []
        for edge in edges:
            if len(edge) == 2:
                edge = ("y", edge[0], edge[1], 0.5 * W)
            edge_axis, c1, c2, lmin = edge
            for (dim, ct) in gmsh.model.getEntities(1):
                bb = gmsh.model.getBoundingBox(dim, ct)
                if edge_axis == "y":       # x=c1, z=c2 fixed; free in y
                    matches = (abs(bb[0] - c1) < tol and abs(bb[3] - c1) < tol
                            and abs(bb[2] - c2) < tol
                            and abs(bb[5] - c2) < tol)
                    free_len = bb[4] - bb[1]
                elif edge_axis == "x":     # y=c1, z=c2 fixed; free in x
                    matches = (abs(bb[1] - c1) < tol and abs(bb[4] - c1) < tol
                            and abs(bb[2] - c2) < tol
                            and abs(bb[5] - c2) < tol)
                    free_len = bb[3] - bb[0]
                else:                   # "z": x=c1, y=c2 fixed; free in z
                    matches = (abs(bb[0] - c1) < tol and abs(bb[3] - c1) < tol
                            and abs(bb[1] - c2) < tol
                            and abs(bb[4] - c2) < tol)
                    free_len = bb[5] - bb[2]
                if matches and free_len > lmin:
                    targets.append(ct)
        rf_used = 0.0
        import tempfile

        def _snapshot():
            ftmp = tempfile.mktemp(suffix=".brep")
            gmsh.write(ftmp)
            return ftmp

        def _restore(ftmp):
            gmsh.clear()
            gmsh.open(ftmp)
            gmsh.model.occ.synchronize()
            return gmsh.model.getEntities(3)[0][1]

        if targets and RF > 0:
            snap = _snapshot()
            for rf in (RF, 0.65 * RF, 0.40 * RF, 0.22 * RF):
                try:
                    out = gmsh.model.occ.fillet([vol], targets, [rf],
                                                removeVolume=True)
                    gmsh.model.occ.synchronize()
                    if not gmsh.model.getEntities(3):
                        raise RuntimeError("empty volume after fillet")
                    vol = out[0][1]
                    rf_used = rf
                    break
                except Exception:
                    vol = _restore(snap)   # OCC may corrupt on failure

        # EDGE BREAK: rounds the remaining external edges (sharp
        # corners) with a small radius RB. Done BEFORE cutting the holes, so as
        # not to round the cylindrical BC shells. Transactional: a fillet
        # failure in OCC can destroy the solid.
        rb_used = 0.0
        if RB > 0:
            snap = _snapshot()
            for rb in (RB, 0.5 * RB, 0.25 * RB, 0.12 * RB):
                all_edges = []
                for (d1, ct) in gmsh.model.getEntities(1):
                    bb = gmsh.model.getBoundingBox(d1, ct)
                    ln = float(np.linalg.norm([bb[3] - bb[0], bb[4] - bb[1],
                                               bb[5] - bb[2]]))
                    if ln > 5.0 * rb:      # too short ones break OCC
                        all_edges.append(ct)
                try:
                    out = gmsh.model.occ.fillet([vol], all_edges, [rb],
                                                removeVolume=True)
                    gmsh.model.occ.synchronize()
                    if not gmsh.model.getEntities(3):
                        raise RuntimeError("empty volume after fillet")
                    vol = out[0][1]
                    rb_used = rb
                    break
                except Exception:
                    vol = _restore(snap)

        # cut the BC holes and the purely geometric cuts (e.g.:
        # the semicircular cradle of the clamp, which receives no BC)
        cils = [(3, gmsh.model.occ.addBox(*b)) for b in cut_boxes]
        cils += [(3, _prim(c)) for c in cut_specs]
        for h in list(fixed_holes) + list(load_holes) + list(cuts):
            c, ax = np.array(h["center"], float), np.array(h["axis"], float)
            ax = ax / np.linalg.norm(ax)
            p0 = c - ax * h["half"] * 1.5
            d = ax * h["half"] * 3.0
            cils.append((3, gmsh.model.occ.addCylinder(*p0, *d, h["radius"])))
        obj, _ = gmsh.model.occ.cut([(3, vol)], cils)
        gmsh.model.occ.synchronize()
        vols = gmsh.model.getEntities(3)
        gmsh.model.addPhysicalGroup(3, [v[1] for v in vols], VOLUME_MARKER)

        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(-1, -1)
        diag = float(np.linalg.norm([xmax - xmin, ymax - ymin, zmax - zmin]))
        f = float(h_factor)                 # <1 refines; 1 = default mesh
        h = min(diag / 18.0, T / 2.0) * f
        gmsh.option.setNumber("Mesh.MeshSizeMax", h)
        # floor proportional to the SMALLER of T and the hole diameter, to
        # resolve the features without blowing up the count:
        gmsh.option.setNumber("Mesh.MeshSizeMin", max(h / 3.0, 0.35 * T * f))
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature",
                              int(round(8 / f)))
        gmsh.option.setNumber("Mesh.Optimize", 1)
        gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
        gmsh.model.mesh.generate(3)
        gmsh.model.mesh.setOrder(1)
        res = gmshio.model_to_mesh(gmsh.model, COMM, rank=0, gdim=3)
        domain = res.mesh if hasattr(res, "mesh") else res[0]
    finally:
        gmsh.finalize()
    return domain, rf_used, rb_used


def mark_cylinder(hole):
    """Nodes/facets on the cylindrical shell of the hole (same predicate as SimJEB)."""
    c = np.array(hole["center"], float)
    axis = np.array(hole["axis"], float)
    axis = axis / (np.linalg.norm(axis) + 1e-30)
    radius, half = float(hole["radius"]), float(hole["half"])
    tol_r = 0.35 * radius + 0.5
    tol_a = 0.5 * half + 1.0

    def marker(x):
        d = x.T - c
        proj = d @ axis
        radial = np.linalg.norm(d - np.outer(proj, axis), axis=1)
        return (np.abs(radial - radius) < tol_r) & (np.abs(proj) < half + tol_a)
    return marker


def mark_box(box):
    lo, hi = np.asarray(box[0], float), np.asarray(box[1], float)

    def marker(x):
        return ((x[0] >= lo[0]) & (x[0] <= hi[0]) &
                (x[1] >= lo[1]) & (x[1] <= hi[1]) &
                (x[2] >= lo[2]) & (x[2] <= hi[2]))
    return marker


def surface_props(domain, ds, marker):
    one = fem.Constant(domain, default_scalar_type(1.0))
    S = fem.assemble_scalar(fem.form(one * ds(marker)))
    if S <= 0:
        raise RuntimeError("load area = 0")
    x = ufl.SpatialCoordinate(domain)
    c = np.array([fem.assemble_scalar(fem.form(x[i] * ds(marker))) / S
                  for i in range(3)], float)
    r = x - ufl.as_vector([float(v) for v in c])
    K = np.zeros((3, 3))
    for i in range(3):
        for j in range(i, 3):
            expr = (ufl.dot(r, r) * (1.0 if i == j else 0.0) - r[i] * r[j])
            K[i, j] = K[j, i] = fem.assemble_scalar(fem.form(expr * ds(marker)))
    if np.linalg.cond(K) > 1e8:
        raise RuntimeError("ill-conditioned K tensor on the load face")
    return float(S), c, K


def solve_case(domain, fixed_holes, load_holes, F, M, E, nu, degree=1):
    """Clamp on the shells of the fixation holes; TOTAL F and M applied to the
    set of load holes (all shells receive the same marker
    LOAD_MARKER, and the traction t = F/S_total + a×r is assembled over the total
    area of the group, with r measured from the group CENTROID). This splits
    the load evenly among the holes and keeps F,M as total resultants."""
    fdim = domain.topology.dim - 1
    fix_sets = [dmesh.locate_entities_boundary(domain, fdim,
                mark_cylinder(h)) for h in fixed_holes]
    fix_sets = [a for a in fix_sets if len(a)]
    if not fix_sets:
        raise RuntimeError("0 facets on the fixation holes")
    fixed_f = np.unique(np.concatenate(fix_sets))
    load_sets = [dmesh.locate_entities_boundary(domain, fdim,
                 mark_cylinder(h)) for h in load_holes]
    load_sets = [a for a in load_sets if len(a)]
    if not load_sets:
        raise RuntimeError("0 facets on the load holes")
    load_f = np.unique(np.concatenate(load_sets))

    idx = np.concatenate([fixed_f, load_f]).astype(np.int32)
    mk = np.concatenate([np.full(len(fixed_f), FIXED_MARKER, np.int32),
                         np.full(len(load_f), LOAD_MARKER, np.int32)])
    order = np.argsort(idx)
    idx, mk = idx[order], mk[order]
    uniq, first = np.unique(idx, return_index=True)
    ftags = dmesh.meshtags(domain, fdim, uniq, mk[first])

    mu_v = E / (2 * (1 + nu))
    lam_v = E * nu / ((1 + nu) * (1 - 2 * nu))
    V = fem.functionspace(domain, ("Lagrange", int(degree), (3,)))
    mu = fem.Constant(domain, default_scalar_type(mu_v))
    lmbda = fem.Constant(domain, default_scalar_type(lam_v))

    def eps(u):
        return ufl.sym(ufl.grad(u))

    def sig(u):
        return lmbda * ufl.tr(eps(u)) * ufl.Identity(3) + 2 * mu * eps(u)

    dofs = fem.locate_dofs_topological(V, fdim, ftags.find(FIXED_MARKER))
    if len(dofs) == 0:
        raise RuntimeError("no clamped dof")
    bc_d = fem.dirichletbc(
        fem.Constant(domain, default_scalar_type((0., 0., 0.))), dofs, V)
    ds = ufl.Measure("ds", domain=domain, subdomain_data=ftags)
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = ufl.inner(sig(u), eps(v)) * ufl.dx

    # total area of the load hole group, centroid and tensor K over it
    S, c, K = surface_props(domain, ds, LOAD_MARKER)
    x = ufl.SpatialCoordinate(domain)
    avec = np.linalg.solve(K, np.asarray(M, float))
    r = x - ufl.as_vector([float(q) for q in c])
    traction = (ufl.as_vector([float(q) for q in np.asarray(F) / S])
                + ufl.cross(ufl.as_vector([float(q) for q in avec]), r))
    L = ufl.dot(traction, v) * ds(LOAD_MARKER)

    problem = LinearProblem(a, L, bcs=[bc_d], petsc_options_prefix="el_",
                            petsc_options=ITER_OPTS)
    ns, _b = rigid_body_nullspace(V)
    problem.A.setNearNullSpace(ns)
    uh = problem.solve()
    if problem.solver.getConvergedReason() <= 0:
        raise RuntimeError("solver did not converge "
                           f"({problem.solver.getConvergedReason()})")

    Vs = fem.functionspace(domain, ("Lagrange", 1))
    Vt = fem.functionspace(domain, ("Lagrange", 1, (3, 3)))

    def proj(expr, space, pref):
        p, q = ufl.TrialFunction(space), ufl.TestFunction(space)
        return LinearProblem(ufl.inner(p, q) * ufl.dx,
                             ufl.inner(expr, q) * ufl.dx,
                             petsc_options_prefix=pref,
                             petsc_options={"ksp_type": "cg",
                                            "pc_type": "jacobi"}).solve()

    s_dev = sig(uh) - (1. / 3.) * ufl.tr(sig(uh)) * ufl.Identity(3)
    vm = proj(ufl.sqrt(3. / 2. * ufl.inner(s_dev, s_dev)), Vs, "vm_")
    st = proj(sig(uh), Vt, "st_")
    cells_flat, _ct, nodes = plot.vtk_mesh(Vs)
    cells = cells_flat.reshape(-1, 5)[:, 1:5].astype(np.int32)
    six = [0, 4, 8, 1, 5, 2]
    Vu1 = fem.functionspace(domain, ("Lagrange", 1, (3,)))
    u1 = fem.Function(Vu1)
    u1.interpolate(uh)
    u_arr = u1.x.array.reshape(-1, 3).copy()
    vm_arr = vm.x.array.copy()
    st6 = st.x.array.reshape(-1, 9)[:, six].copy()
    if not (np.all(np.isfinite(vm_arr)) and np.all(np.isfinite(u_arr))):
        raise RuntimeError("non-finite solution")
    return nodes, cells, u_arr, vm_arr, st6, float(np.sqrt(S)), c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--i0", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--f-min", type=float, default=200.0)
    ap.add_argument("--f-max", type=float, default=5000.0)
    ap.add_argument("--e-min", type=float, default=60_000.0)
    ap.add_argument("--e-max", type=float, default=300_000.0)
    ap.add_argument("--nu-min", type=float, default=0.20)
    ap.add_argument("--nu-max", type=float, default=0.45)
    ap.add_argument("--check-lin", action="store_true")
    ap.add_argument("--h-fator", type=float, default=1.0,
                    help="element size factor (0.7 refines ~1.4x; "
                         "1.0 = default mesh) — for the convergence study")
    ap.add_argument("--grau", type=int, choices=[1, 2, 3], default=1,
                    help="degree of the displacement elements: 2 = much "
                         "cleaner stress target, solver ~4-8x slower")
    ap.add_argument("--geo-fixa", type=int, default=None,
                    metavar="SEED",
                    help="freezes the geometry (the same part in the whole batch), "
                         "varying only load and material")
    ap.add_argument("--fam", choices=FAMILIES, default=None,
                    help="forces the family (L/Z/U) in the whole batch; without the "
                         "flag, samples by index as before")
    ap.add_argument("--conv-grau", type=int, default=None, choices=[2, 3],
                    help="solves the SAME case (SAME mesh) also at this "
                         "degree and reports dpeak and relL2 — convergence "
                         "study by p-refinement without depending on "
                         "mesh reproducibility")
    args = ap.parse_args()
    if args.conv_grau is not None and args.conv_grau <= args.grau:
        raise SystemExit(f"--conv-grau ({args.conv_grau}) must be GREATER than "
                         f"--grau ({args.grau}): the higher degree is the reference")

    samples_dir = os.path.join(args.out_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "meta_simples.json"), "w") as f:
        json.dump({"familias": FAMILIES, "seed": args.seed,
                   "f_min": args.f_min, "f_max": args.f_max,
                   "e_min": args.e_min, "e_max": args.e_max,
                   "nu_min": args.nu_min, "nu_max": args.nu_max}, f, indent=2)
    manifest = os.path.join(args.out_dir, "manifest_simples.csv")
    fresh = not os.path.exists(manifest) or args.i0 == 0
    fout = open(manifest, "w" if fresh else "a", newline="")
    wr = csv.DictWriter(fout, fieldnames=["id", "status", "familia", "F", "M",
                                          "E", "nu", "n_nodes", "max_vm",
                                          "error"])
    if fresh:
        wr.writeheader()

    import time
    ok = fail = 0
    for i in range(args.i0, args.n):
        rng = np.random.default_rng(args.seed + i)
        # --geo-fixa: the GEOMETRY comes from its own seed (identical in
        # all cases of the batch); load and material are still drawn per
        # case from `rng`. Isolates the geometric variability.
        rng_geo = (rng if args.geo_fixa is None
                   else np.random.default_rng(args.geo_fixa))
        fam, params, boxes, edges, fixed_holes, load_holes = \
            sample_geometry(rng_geo, None if args.fam is None
                              else FAMILIES.index(args.fam))
        A, B, C, W, T, RF, RB, DF, DL, NF, NL = params
        dF = rng.normal(size=3); dF /= np.linalg.norm(dF)
        dM = rng.normal(size=3); dM /= np.linalg.norm(dM)
        F = float(rng.uniform(args.f_min, args.f_max)) * dF
        lever0 = np.sqrt(T * W)                      # ~side of the load face
        M = float(rng.uniform(args.f_min, args.f_max) * lever0) * dM
        E = float(np.exp(rng.uniform(np.log(args.e_min), np.log(args.e_max))))
        nu = float(rng.uniform(args.nu_min, args.nu_max))
        row = {"id": i, "familia": FAMILIES[fam],
               "F": f"{np.linalg.norm(F):.1f}", "M": f"{np.linalg.norm(M):.1f}",
               "E": f"{E:.0f}", "nu": f"{nu:.4f}"}
        t0 = time.time()
        try:
            domain, rf_used, rb_used = build_mesh(
                boxes, edges, fixed_holes, load_holes, T, W, RF, RB,
                h_factor=args.h_fator)
            params[5] = rf_used          # effective concave RF
            params[6] = rb_used          # effective edge RB
            nodes, cells, u_arr, vm_arr, st6, lever, load_c = solve_case(
                domain, fixed_holes, load_holes, F, M, E, nu,
                degree=args.grau)
            np.savez_compressed(
                os.path.join(samples_dir, f"sample_{i:06d}.npz"),
                nodes=nodes.astype(np.float32), cells=cells,
                u=u_arr.astype(np.float32), stress=st6.astype(np.float32),
                von_mises=vm_arr.astype(np.float32),
                E=np.float32(E), nu=np.float32(nu),
                loads=np.asarray(F, np.float32)[None],
                moments=np.asarray(M, np.float32)[None],
                l_ref=np.float32(lever),
                familia=np.int32(fam), geo_params=params.astype(np.float32),
                furos_fix=np.array([h["center"] + h["axis"] +
                                    [h["radius"], h["half"]]
                                    for h in fixed_holes], np.float32),
                furos_carga=np.array([h["center"] + h["axis"] +
                                      [h["radius"], h["half"]]
                                      for h in load_holes], np.float32),
                load_centers=np.asarray(load_c, np.float32)[None],
                fixed_centers=np.array([h["center"] for h in fixed_holes],
                                       np.float32),
                n_loads=np.int32(1), n_nodes=np.int32(nodes.shape[0]),
                n_cells=np.int32(cells.shape[0]),
                max_vm=np.float32(vm_arr.max()),
                max_disp=np.float32(np.linalg.norm(u_arr, axis=1).max()))
            row.update(status="ok", n_nodes=nodes.shape[0],
                       max_vm=f"{vm_arr.max():.2f}")
            ok += 1
            print(f"  [{i:06d}] OK {FAMILIES[fam]}  A={A:5.1f} B={B:5.1f} "
                  f"T={T:4.1f} | {len(fixed_holes)}holes+load rf={params[5]:.1f} "
                  f"| |F|={row['F']:>7s}N |M|={row['M']:>9s} | "
                  f"{nodes.shape[0]:6d} nodes | vm={vm_arr.max():7.1f} "
                  f"({time.time()-t0:.1f}s)")
            if args.check_lin and i == args.i0:
                _, _, _, vm2, _, _, _ = solve_case(
                    domain, fixed_holes, load_holes, 2 * F, 2 * M, E, nu,
                    degree=args.grau)
                rel = (np.linalg.norm(vm2 - 2 * vm_arr)
                       / (np.linalg.norm(2 * vm_arr) + 1e-30))
                print(f"  [check-lin] ||vm(2L)-2vm(L)||/||2vm(L)|| = {rel:.2e} "
                      f"{'OK' if rel < 1e-6 else 'SUSPECT'}")
            if args.conv_grau:
                t1 = time.time()
                _, _, _, vm_hi, _, _, _ = solve_case(
                    domain, fixed_holes, load_holes, F, M, E, nu,
                    degree=args.conv_grau)
                # same mesh in both solutions and both projected onto
                # degree-1 Lagrange at the vertices -> node-by-node relL2, without
                # interpolation and without depending on reproducibility
                rel_c = (np.linalg.norm(vm_arr - vm_hi)
                         / (np.linalg.norm(vm_hi) + 1e-30))
                dp_c = (100.0 * (vm_arr.max() - vm_hi.max())
                        / (vm_hi.max() + 1e-30))
                print(f"  [conv] {i:06d} {FAMILIES[fam]} degree {args.grau} vs "
                      f"{args.conv_grau} | peak {vm_arr.max():.1f} vs "
                      f"{vm_hi.max():.1f} | dpeak {dp_c:+.1f}% | "
                      f"relL2 {rel_c:.4f} | "
                      f"{'PASS' if abs(dp_c) <= 5.0 else 'FAIL'} "
                      f"({time.time()-t1:.1f}s)")
                # conv_ prefix (the training list_samples globs sample_*.npz,
                # so this file never enters a training dataset)
                np.savez_compressed(
                    os.path.join(samples_dir,
                                 f"conv_{i:06d}_g{args.conv_grau}.npz"),
                    von_mises=vm_hi.astype(np.float32),
                    von_mises_ref=vm_arr.astype(np.float32),
                    nodes=nodes.astype(np.float32), cells=cells,
                    grau=np.int32(args.grau),
                    grau_conv=np.int32(args.conv_grau),
                    familia=np.int32(fam),
                    geo_params=params.astype(np.float32),
                    loads=np.asarray(F, np.float32)[None],
                    moments=np.asarray(M, np.float32)[None],
                    E=np.float32(E), nu=np.float32(nu),
                    dpico=np.float32(dp_c), rel_l2=np.float32(rel_c))
            del domain
        except Exception as e:
            row.update(status="failed", error=repr(e)[:200])
            fail += 1
            print(f"  [{i:06d}] FAILED ({FAMILIES[fam]}): {e!r}")
        wr.writerow(row)
        fout.flush()
        gc.collect()
    fout.close()
    print(f"\nDone: {ok} ok, {fail} failures in {samples_dir}")


if __name__ == "__main__":
    main()
