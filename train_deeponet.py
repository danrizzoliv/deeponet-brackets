#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The DeepONet and its training loop.

Branch MLP and trunk MLP, 4 x 128 with tanh, joined either by the classical
inner product (--decoder dot) or by an element-wise product followed by a
non-linear head 128 -> 64 -> 1 (--decoder prod, the fusion decoder). The
geometry reaches the branch as explicit parameters; the trunk receives the
coordinates and, optionally, physical distance features
(see bracket_dataset.py).

Revised model of the monograph (relative L2 0.193 on 8,370 validation cases):

    python train_deeponet.py --dataset dataset_v4_train --device cuda \\
        --batch 32 --n-query 1024 --repeats 1 --decoder prod \\
        --hole-distances --edge-distance --hole-proximity \\
        --stress-weight 1 --weight-floor 0.3 \\
        --eval-every 10 --save-every 10 --final-eval 400 --resume \\
        --epochs 4000 --lr 3e-4 --out revised.pt

Delivered model (0.252): the same without the last three feature flags and
the two weighting options, as described in the README.

Checkpoints written before the code was translated (Portuguese option
names in `args`, split key `treino`) load through `load_model`.
"""

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, os.environ.get("TORCH_THREADS", "2"))

import argparse
import math
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "2")))

from bracket_dataset import (FAMILIES, KEY_FAMILY, NO_EDGES, list_samples,
                             split_files, compute_stats, BracketDataset,
                             case_branch, trunk_features, load_edge_map,
                             denormalize_vm)


# ------------------------------------------------------------- networks

class MLP(nn.Module):
    def __init__(self, d_in, d_out, hidden=128, layers=4, act=nn.Tanh):
        super().__init__()
        seq, d = [], d_in
        for _ in range(layers):
            seq += [nn.Linear(d, hidden), act()]
            d = hidden
        seq += [nn.Linear(d, d_out)]
        self.net = nn.Sequential(*seq)

    def forward(self, x):
        return self.net(x)


class Sine(nn.Module):
    def __init__(self, w0=30.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


class SirenMLP(nn.Module):
    """MLP with sine activations and the initialization of Sitzmann et al.
    (2020): first layer U(-1/d, 1/d), the others U(-sqrt(6/d)/w0,
    +sqrt(6/d)/w0). Same parameter count as the tanh MLP."""

    def __init__(self, d_in, d_out, hidden=128, layers=4, w0=30.0):
        super().__init__()
        seq, d = [], d_in
        for i in range(layers):
            lin = nn.Linear(d, hidden)
            with torch.no_grad():
                if i == 0:
                    lin.weight.uniform_(-1.0 / d, 1.0 / d)
                else:
                    b = math.sqrt(6.0 / d) / w0
                    lin.weight.uniform_(-b, b)
            seq += [lin, Sine(w0)]
            d = hidden
        out = nn.Linear(d, d_out)
        with torch.no_grad():
            b = math.sqrt(6.0 / d) / w0
            out.weight.uniform_(-b, b)
        seq.append(out)
        self.net = nn.Sequential(*seq)

    def forward(self, x):
        return self.net(x)


class DeepONet(nn.Module):
    """`d_extra` is the number of trunk inputs beyond the three coordinates.
    The attribute names (branch, trunk, head, bias) are those of the
    archived checkpoints and must not change."""

    def __init__(self, branch_dim=31, p=128, hidden=128, layers=4, fourier=0,
                 d_extra=0, decoder="dot", trunk="tanh"):
        super().__init__()
        self.fourier = int(fourier)
        self.d_extra = int(d_extra)
        self.decoder = str(decoder)
        d_trunk = 3 * (1 + 2 * self.fourier) + self.d_extra
        self.branch = MLP(branch_dim, p, hidden, layers)
        self.trunk = (SirenMLP(d_trunk, p, hidden, layers)
                      if trunk == "siren" else MLP(d_trunk, p, hidden, layers))
        self.bias = nn.Parameter(torch.zeros(1))
        if self.decoder == "prod":
            # element-wise fusion and a non-linear head (after Geom-DeepONet):
            # the field is no longer a linear combination of p global bases,
            # which is what lets a peak move with the geometry
            self.head = nn.Sequential(nn.Linear(p, 64), nn.Tanh(),
                                      nn.Linear(64, 1))

    def _encode(self, x):
        """Fixed Fourier features on the three coordinates only; the extra
        features pass through unchanged."""
        if self.fourier == 0:
            return x
        xyz, extra = x[..., :3], x[..., 3:]
        feats = [xyz]
        for j in range(self.fourier):
            w = (2.0 ** j) * torch.pi
            feats += [torch.sin(w * xyz), torch.cos(w * xyz)]
        feats.append(extra)
        return torch.cat(feats, dim=-1)

    def forward(self, b, x):
        B = self.branch(b)
        T = self.trunk(self._encode(x))
        if self.decoder == "prod":
            return self.head(B.unsqueeze(1) * T).squeeze(-1) + self.bias
        return torch.einsum("bp,bqp->bq", B, T) + self.bias


class StressWeightedLoss(nn.Module):
    """Mean squared error on the standardized log target, with each point
    weighted by its stress raised to alpha, normalized per case:

        w_i = t_i^alpha / mean_case(t^alpha),   t = sigma_vm / s
        w_i <- floor + (1 - floor) * w_i,       clipped at w_max
        loss = mean(w * (prediction - target)^2)

    Why: the MSE on the log gives a point at 1 MPa the same weight as a
    point at 300 MPa, while the relative L2 error, the metric, is dominated
    by the peaks. To first order a log error delta at stress v is an error
    v * delta in stress, so alpha = 2 reproduces the metric; alpha = 1 sits
    half way with less noisy gradients. The per-case normalization keeps
    every case with the same total weight, as in the per-case metric.
    alpha = 0 is the plain MSE. The floor guarantees every point a fraction
    of the mean weight (floor = 0.3 is 30% plain MSE + 70% weighted), which
    limits the loss of accuracy in the least stressed regions."""

    def __init__(self, alpha, stats, w_max=100.0, floor=0.0):
        super().__init__()
        self.alpha = float(alpha)
        self.floor = float(floor)
        self.w_max = float(w_max)
        self.log = stats.get("target_transform", "linear") == "log"
        self.t_mean = float(stats["t_mean"])
        self.t_std = float(stats["t_std"])
        self.t_eps = float(stats.get("t_eps", 0.0))

    def forward(self, pred, y):
        e2 = (pred - y) ** 2
        if self.alpha == 0:
            return e2.mean()
        with torch.no_grad():
            t = y * self.t_std + self.t_mean
            if self.log:
                t = torch.exp(t) - self.t_eps
            w = t.clamp_min(0) ** self.alpha                 # (B, Q)
            w = w / (w.mean(dim=-1, keepdim=True) + 1e-30)
            w = (self.floor + (1.0 - self.floor) * w).clamp_max(self.w_max)
        return (w * e2).mean()


# -------------------------------------------------- checkpoint interface

# option names before the translation -> current names
LEGACY_ARGS = {"dist_furos": "hole_distances",
               "dist_fillet": "edge_distance",
               "dist_rel": "hole_proximity",
               "peso_tensao": "stress_weight",
               "peso_piso": "weight_floor",
               "frac_furos": "hole_fraction"}


def model_config(args):
    """Training options of a checkpoint, with the current names."""
    cfg = dict(args)
    for old, new in LEGACY_ARGS.items():
        if old in cfg and new not in cfg:
            cfg[new] = cfg.pop(old)
    return cfg


def trunk_extra(cfg):
    """Number of trunk inputs beyond the coordinates."""
    return ((2 if cfg.get("hole_distances") else 0)
            + (1 if cfg.get("edge_distance") else 0)
            + (2 if cfg.get("hole_proximity") else 0))


def checkpoint_split(ckpt):
    """{'train': [...], 'val': [...]}, also for checkpoints whose training
    list is stored under the key 'treino'."""
    sp = ckpt["split"]
    return {"train": sp.get("train", sp.get("treino", [])),
            "val": sp["val"]}


def build_model(stats, cfg):
    return DeepONet(branch_dim=stats["branch_dim"], p=cfg.get("p", 128),
                    hidden=cfg.get("hidden", 128),
                    layers=cfg.get("layers", 4),
                    fourier=cfg.get("fourier", 0), d_extra=trunk_extra(cfg),
                    decoder=cfg.get("decoder", "dot"),
                    trunk=cfg.get("trunk", "tanh"))


def load_model(path, device="cpu"):
    """Returns (model, ckpt, cfg), with the model in eval mode."""
    ckpt = torch.load(os.path.expanduser(path), map_location="cpu",
                      weights_only=False)
    cfg = model_config(ckpt["args"])
    model = build_model(ckpt["stats"], cfg)
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), ckpt, cfg


def case_features(path, stats, cfg, edge_map=None):
    """Branch and trunk inputs of one case, as the model was trained.
    Returns (branch, trunk, s, vm, nodes, data)."""
    branch, s, vm, nodes, d = case_branch(path, stats)
    edges = None
    if cfg.get("edge_distance"):
        if edge_map is None:
            raise ValueError("this model uses the edge distance: pass the "
                             "map from bracket_dataset.load_edge_map")
        edges = edge_map.get(os.path.basename(path), NO_EDGES)
    ft = trunk_features(nodes, d, stats, edges=edges)
    if not cfg.get("hole_distances"):
        ft = ft[:, :3]
    return branch, ft, s, vm, nodes, d


@torch.no_grad()
def predict_full(model, branch, trunk_in, device, chunk=16384):
    """Standardized prediction at every node of one case."""
    model.eval()
    b = torch.from_numpy(branch)[None].to(device)
    y = np.empty(trunk_in.shape[0], np.float32)
    for i in range(0, trunk_in.shape[0], chunk):
        x = torch.from_numpy(trunk_in[i:i + chunk])[None].to(device)
        y[i:i + chunk] = model(b, x)[0].cpu().numpy()
    return y


def predict_case(model, path, stats, cfg, device="cpu", edge_map=None):
    """Returns (vm_true, vm_pred, nodes, data) of one case, in stress units."""
    branch, ft, s, vm, nodes, d = case_features(path, stats, cfg, edge_map)
    y = predict_full(model, branch, ft, device)
    return vm, denormalize_vm(y, s, stats), nodes, d


def relative_l2(vm_pred, vm_true):
    return float(np.linalg.norm(vm_pred - vm_true)
                 / (np.linalg.norm(vm_true) + 1e-9))


# --------------------------------------------------------------- training

def run_epoch(model, loader, opt, loss_fn, device, train):
    model.train(train)
    total, n = 0.0, 0
    for b, x, y in loader:
        b, x, y = b.to(device), x.to(device), y.to(device)
        if train:
            opt.zero_grad()
        loss = loss_fn(model(b, x), y)
        if train:
            loss.backward()
            opt.step()
        total += loss.item() * b.shape[0]
        n += b.shape[0]
    return total / max(n, 1)


@torch.no_grad()
def eval_field(model, files, stats, cfg, device, max_cases=None,
               edge_map=None, details=None):
    """Mean and median relative L2 error over the full fields. If `details`
    is a list, appends (file name, family, error) for every case."""
    errs = []
    for f in (files if max_cases is None else files[:max_cases]):
        vm, vm_p, _, d = predict_case(model, f, stats, cfg, device, edge_map)
        errs.append(relative_l2(vm_p, vm))
        if details is not None:
            details.append((os.path.basename(f), int(d[KEY_FAMILY]),
                            errs[-1]))
    return float(np.mean(errs)), float(np.median(errs))


def family_report(details, title):
    print(f"relL2 by family — {title}:")
    for fam in sorted({f for _, f, _ in details}):
        e = np.array([x for _, f, x in details if f == fam])
        print(f"  {FAMILIES[fam]}: mean={e.mean():.3f} "
              f"median={np.median(e):.3f} ({len(e)} cases)")


def parse_args():
    ap = argparse.ArgumentParser(
        description="Train the bracket DeepONet.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--dataset", required=True,
                    help="dataset folder (with samples/sample_*.npz)")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--n-query", type=int, default=1024,
                    help="query points drawn per case and visit")
    ap.add_argument("--repeats", type=int, default=4,
                    help="visits per case per epoch")
    ap.add_argument("--lr", type=float, default=None,
                    help="learning rate (default 1e-3); on --resume it "
                         "overrides the checkpoint's and restarts the "
                         "scheduler")
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--p", type=int, default=128, help="latent dimension")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--fourier", type=int, default=0,
                    help="octaves of Fourier features in the trunk (0 = off)")
    ap.add_argument("--decoder", choices=["dot", "prod"], default="dot",
                    help="dot = classical inner product; prod = element-wise "
                         "fusion and non-linear head")
    ap.add_argument("--trunk", choices=["tanh", "siren"], default="tanh")
    g = ap.add_argument_group("trunk features")
    g.add_argument("--hole-distances", action="store_true",
                   help="distances to the nearest fixation and load hole")
    g.add_argument("--edge-distance", action="store_true",
                   help="distance to the nearest concave edge; needs "
                        "concave_edges.pkl in the dataset; implies "
                        "--hole-distances")
    g.add_argument("--hole-proximity", action="store_true",
                   help="proximity r/(r+d) to the nearest fixation and load "
                        "hole, in the hole's own radius; implies "
                        "--hole-distances")
    g = ap.add_argument_group("loss and sampling")
    g.add_argument("--stress-weight", type=float, default=0.0,
                   metavar="ALPHA",
                   help="weight the loss by stress^ALPHA, normalized per "
                        "case (0 = plain MSE, 1 = revised model)")
    g.add_argument("--weight-floor", type=float, default=0.0,
                   help="with --stress-weight: minimum weight of a point, as "
                        "a fraction of the mean (0 to 1; revised model 0.3)")
    g.add_argument("--hole-fraction", type=float, default=0.0,
                   help="fraction of the query points drawn near the holes "
                        "(importance sampling)")
    ap.add_argument("--subset", type=int, default=0,
                    help="use only N cases drawn from the dataset with a "
                         "fixed seed (cheap screening; same N, same cases)")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--target", choices=["log", "linear"], default="log")
    ap.add_argument("--out", default="deeponet.pt")
    ap.add_argument("--resume", action="store_true",
                    help="resume from --out: weights, optimizer, scheduler "
                         "and epoch (copy --out.last over --out to resume "
                         "from the latest state rather than the best)")
    ap.add_argument("--stats-cache", default="",
                    help="file in which to keep the normalization statistics "
                         "between runs (computing them reads every case)")
    ap.add_argument("--final-eval", type=int, default=400,
                    help="cases per partition in the final evaluation "
                         "(0 = all)")
    ap.add_argument("--save-every", type=int, default=10,
                    help="also save a resume state every N epochs")
    # option names before the translation, accepted and hidden
    for old, new in LEGACY_ARGS.items():
        flag = "--" + old.replace("_", "-")
        if new in ("hole_distances", "edge_distance", "hole_proximity"):
            ap.add_argument(flag, dest=new, action="store_true",
                            help=argparse.SUPPRESS)
        else:
            ap.add_argument(flag, dest=new, type=float,
                            help=argparse.SUPPRESS)
    return ap.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[warning] CUDA not available; using CPU.")
        device = "cpu"
    if not 0.0 <= args.weight_floor < 1.0:
        raise SystemExit("--weight-floor must be in [0, 1)")
    for flag in ("edge_distance", "hole_proximity"):
        if getattr(args, flag) and not args.hole_distances:
            print(f"[{flag}] turning on --hole-distances, on which it builds")
            args.hole_distances = True

    files = list_samples(os.path.join(args.dataset, "samples"))
    if args.subset and args.subset < len(files):
        rs = np.random.default_rng(12345)
        files = sorted(files[i] for i in rs.choice(len(files), args.subset,
                                                   replace=False))
        print(f"[subset] {len(files)} cases drawn from the dataset")
    if len(files) < 10:
        raise SystemExit(f"too few cases ({len(files)})")
    tr, va = split_files(files, args.val_frac, args.seed)

    # the statistics depend only on (training files, target), both fixed by
    # the seed, so they can be reused across runs
    import pickle
    stats, key = None, {"n_tr": len(tr), "seed": args.seed,
                        "target": args.target, "val_frac": args.val_frac,
                        "subset": args.subset}
    if args.stats_cache and os.path.exists(args.stats_cache):
        with open(args.stats_cache, "rb") as fh:
            c = pickle.load(fh)
        if all(c.get(k, 0 if k == "subset" else None) == v
               for k, v in key.items()):
            stats = c["stats"]
            print(f"[stats] reused from {args.stats_cache}")
        else:
            print("[stats] cache does not match these options; recomputing")
    if stats is None:
        t0 = time.time()
        stats = compute_stats(tr, target=args.target)
        print(f"[stats] computed in {time.time() - t0:.0f}s")
        if args.stats_cache:
            with open(args.stats_cache, "wb") as fh:
                pickle.dump({"stats": stats, **key}, fh)
    # the proximity features travel in the stats, which go into the
    # checkpoint, so every script that rebuilds the features reproduces them
    stats = dict(stats)
    stats["hole_proximity"] = bool(args.hole_proximity)
    print(f"{len(files)} cases | train={len(tr)} val={len(va)} | "
          f"branch_dim={stats['branch_dim']} | device={device}")

    # the edge map is read after the split and does not change it: models
    # with and without the feature are trained and validated on the same cases
    edge_map = None
    if args.edge_distance:
        edge_map, missing = load_edge_map(files)
        print(f"[edge-distance] edges for {len(files) - missing} of "
              f"{len(files)} cases; {missing} without an edge read as 'far'")
        if missing > 0.01 * len(files):
            raise SystemExit("more than 1% of the cases have no edge; "
                             "check the seed given to concave_edges.py")

    cfg = model_config(vars(args))
    train_ds = BracketDataset(tr, stats, n_query=args.n_query,
                              repeats=args.repeats,
                              hole_distances=args.hole_distances,
                              hole_fraction=args.hole_fraction,
                              edge_map=edge_map)
    val_ds = BracketDataset(va, stats, n_query=args.n_query,
                            hole_distances=args.hole_distances,
                            edge_map=edge_map)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False)

    model = build_model(stats, cfg).to(device)
    n_params = sum(q.numel() for q in model.parameters())
    print(f"DeepONet: p={args.p} hidden={args.hidden} layers={args.layers} "
          f"decoder={args.decoder} trunk={args.trunk} "
          f"trunk_extra={trunk_extra(cfg)} stress_weight={args.stress_weight} "
          f"weight_floor={args.weight_floor} params={n_params:,}")
    opt = torch.optim.Adam(model.parameters(),
                           lr=args.lr if args.lr is not None else 1e-3,
                           weight_decay=args.weight_decay)

    def new_scheduler():
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, factor=0.5, patience=25, min_lr=1e-5)

    sched = new_scheduler()
    loss_fn = StressWeightedLoss(args.stress_weight, stats,
                                 floor=args.weight_floor)
    split = {"train": [os.path.basename(f) for f in tr],
             "val": [os.path.basename(f) for f in va]}
    best, ep0 = float("inf"), 1

    # ---- resume: weights, optimizer, scheduler and epoch; the split and
    # the statistics are those of the checkpoint
    if args.resume and os.path.exists(args.out):
        ck = torch.load(args.out, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        if "opt" in ck:
            opt.load_state_dict(ck["opt"])
        if "sched" in ck:
            sched.load_state_dict(ck["sched"])
        best = ck.get("best", float("inf"))
        ep0 = ck.get("epoch", 0) + 1
        if args.lr is not None:
            for grp in opt.param_groups:
                grp["lr"] = args.lr
            sched = new_scheduler()
            print(f"[resume] learning rate restarted at {args.lr:g}")
        if "split" in ck:
            n_ck = len(checkpoint_split(ck)["train"])
            if n_ck != len(tr):
                raise SystemExit(f"the checkpoint split ({n_ck} training "
                                 f"cases) differs from this one ({len(tr)}); "
                                 "use the same --val-frac and --seed")
            stats = ck["stats"]
        print(f"[resume] from epoch {ep0} | best val {best:.4e}")
        if ep0 > args.epochs:
            raise SystemExit(f"already trained to epoch {ep0 - 1}; raise "
                             "--epochs to continue")

    def save(path, ep):
        torch.save({"model": model.state_dict(), "args": vars(args),
                    "stats": stats, "split": split, "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "epoch": ep, "best": best},
                   path)

    for ep in range(ep0, args.epochs + 1):
        t_ep = time.time()
        tr_loss = run_epoch(model, train_dl, opt, loss_fn, device, True)
        va_loss = run_epoch(model, val_dl, opt, loss_fn, device, False)
        sched.step(tr_loss)
        msg = (f"epoch {ep:3d} | {time.time() - t_ep:5.1f}s | "
               f"train {tr_loss:.4e} | val {va_loss:.4e}")
        if ep % args.eval_every == 0 or ep == 1:
            tm, tmd = eval_field(model, tr, stats, cfg, device, 60, edge_map)
            vm_, vmd = eval_field(model, va, stats, cfg, device, 60, edge_map)
            msg += (f" | relL2 train mean={tm:.3f} median={tmd:.3f}"
                    f" | VAL mean={vm_:.3f} median={vmd:.3f}")
        print(msg, flush=True)
        if va_loss < best:
            best = va_loss
            save(args.out, ep)
        if args.save_every and ep % args.save_every == 0:
            save(args.out + ".last", ep)

    model.load_state_dict(torch.load(args.out, map_location="cpu",
                                     weights_only=False)["model"])
    model.to(device)
    n_eval = args.final_eval or None
    t0 = time.time()
    det_tr, det_va = [], []
    tm, tmd = eval_field(model, tr, stats, cfg, device, n_eval, edge_map,
                         det_tr)
    vm_, vmd = eval_field(model, va, stats, cfg, device, n_eval, edge_map,
                          det_va)
    print(f"\nfinal evaluation on {len(det_tr)} training and {len(det_va)} "
          f"validation cases ({time.time() - t0:.0f}s)")
    print(f"best validation loss: {best:.4e}")
    print(f"relL2 — TRAIN: mean={tm:.3f} median={tmd:.3f}")
    print(f"relL2 — VAL  : mean={vm_:.3f} median={vmd:.3f}")
    family_report(det_va, "VAL")
    # per-case validation error, for paired comparisons (compare_models.py)
    path = os.path.splitext(args.out)[0] + "_val_errors.csv"
    with open(path, "w") as fh:
        fh.write("file,family,relL2\n")
        for name, fam, e in det_va:
            fh.write(f"{name},{FAMILIES[fam]},{e:.6f}\n")
    print(f"per-case validation errors in {path}")
    print(f"model saved in {args.out}")


if __name__ == "__main__":
    main()
