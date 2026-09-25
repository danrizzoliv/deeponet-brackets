#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TRAINING — CLASSICAL DeepONet for simple parametric brackets.

Back to basics, on purpose: branch MLP + trunk MLP + dot product.
No Fourier, no PointNet, no local features. Geometry enters as
explicit parameters in the branch (one-hot family + dimensions).

    python train_simples.py --dataset ~/projetos/simples/dataset_v1 \\
        --epochs 300 --out deeponet_simples.pt
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

from dataset_simples import (list_samples, split_files, compute_stats,
                             SimpleDataset, case_branch, trunk_feats,
                             denormalize_vm)


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
    """MLP with sine activation and the initialization of Sitzmann et al. (2020):
    1st layer U(-1/d, 1/d); the others U(-sqrt(6/d)/w0, +sqrt(6/d)/w0).
    Same parameter count as the equivalent tanh MLP."""

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


class SimpleDeepONet(nn.Module):
    def __init__(self, branch_dim=16, p=128, hidden=128, layers=4, fourier=0,
                 d_extra=0, decoder="dot", trunk="tanh"):
        super().__init__()
        self.fourier = int(fourier)
        self.d_extra = int(d_extra)          # extra physical features (dists)
        self.decoder = str(decoder)
        d_xyz = 3 * (1 + 2 * self.fourier) + self.d_extra
        self.branch = MLP(branch_dim, p, hidden, layers)
        self.trunk = (SirenMLP(d_xyz, p, hidden, layers)
                      if trunk == "siren" else MLP(d_xyz, p, hidden, layers))
        self.bias = nn.Parameter(torch.zeros(1))
        if self.decoder == "prod":
            # element-wise fusion + non-linear head (in the spirit of
            # Geom-DeepONet): the field is no longer a LINEAR combination of
            # p global bases — which is what allows peaks that move around
            self.head = nn.Sequential(nn.Linear(p, 64), nn.Tanh(),
                                      nn.Linear(64, 1))

    def _enc(self, x):
        """FIXED Fourier on the 3 spatial coords only; extras pass through."""
        xyz, extra = x[..., :3], x[..., 3:]
        if self.fourier == 0:
            return x
        feats = [xyz]
        for j in range(self.fourier):
            w = (2.0 ** j) * torch.pi
            feats += [torch.sin(w * xyz), torch.cos(w * xyz)]
        feats.append(extra)
        return torch.cat(feats, dim=-1)

    def forward(self, b, xyz):
        B = self.branch(b)
        T = self.trunk(self._enc(xyz))
        if self.decoder == "prod":
            h = B.unsqueeze(1) * T               # (b, q, p) fusion by product
            return self.head(h).squeeze(-1) + self.bias
        return torch.einsum("bp,bqp->bq", B, T) + self.bias


def run_epoch(model, loader, opt, loss_fn, device, train):
    model.train(train)
    total, n = 0.0, 0
    for b, xyz, y in loader:
        b, xyz, y = b.to(device), xyz.to(device), y.to(device)
        if train:
            opt.zero_grad()
        loss = loss_fn(model(b, xyz), y)
        if train:
            loss.backward()
            opt.step()
        total += loss.item() * b.shape[0]
        n += b.shape[0]
    return total / max(n, 1)


@torch.no_grad()
def predict_full(model, branch, xyz_norm, device, chunk=16384):
    model.eval()
    b = torch.from_numpy(branch)[None].to(device)
    N = xyz_norm.shape[0]
    y = np.empty(N, np.float32)
    for i in range(0, N, chunk):
        xt = torch.from_numpy(xyz_norm[i:i + chunk])[None].to(device)
        y[i:i + chunk] = model(b, xt)[0].cpu().numpy()
    return y


@torch.no_grad()
def eval_field(model, files, stats, device, max_cases=None, dist=False):
    errs = []
    for f in (files if max_cases is None else files[:max_cases]):
        branch, s, vm, nodes, d = case_branch(f, stats)
        ft = trunk_feats(nodes, d, stats)
        if not dist:
            ft = ft[:, :3]
        y = predict_full(model, branch, ft, device)
        vm_p = denormalize_vm(y, s, stats)
        errs.append(np.linalg.norm(vm_p - vm) / (np.linalg.norm(vm) + 1e-9))
    return float(np.mean(errs)), float(np.median(errs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--n-query", type=int, default=1024)
    ap.add_argument("--repeats", type=int, default=4)
    ap.add_argument("--lr", type=float, default=None,
                    help="learning rate; default 1e-3. On "
                         "resume, overrides the checkpoint's")
    ap.add_argument("--weight-decay", type=float, default=0.0,
                    help="Adam L2 regularization (e.g. 1e-4)")
    ap.add_argument("--p", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--fourier", type=int, default=0,
                    help="number of Fourier octaves in the trunk (0=off)")
    ap.add_argument("--dist-furos", action="store_true",
                    help="adds to the trunk the distances to the nearest "
                         "fixation and load holes (2 features)")
    ap.add_argument("--frac-furos", type=float, default=0.0,
                    help="fraction of the training points drawn near "
                         "holes (importance sampling; e.g. 0.3)")
    ap.add_argument("--decoder", choices=["dot", "prod"], default="dot",
                    help="dot=classical dot product; prod=element-wise "
                         "fusion + non-linear head")
    ap.add_argument("--trunk", choices=["tanh", "siren"], default="tanh",
                    help="trunk activation; siren = sines with the init of "
                         "Sitzmann (same parameter count)")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--target", choices=["log", "linear"], default="log")
    ap.add_argument("--out", default="deeponet_simples.pt")
    ap.add_argument("--resume", action="store_true",
                    help="resumes from --out if it exists: weights, Adam state, "
                         "scheduler and the epoch where it stopped. Essential on "
                         "Colab, where the session drops before the end.")
    ap.add_argument("--stats-cache", default="",
                    help="file in which to keep the normalization "
                         "statistics. Computing them sweeps all the training "
                         "cases, which with tens of thousands costs minutes "
                         "at every start — including every --resume")
    ap.add_argument("--final-eval", type=int, default=400,
                    help="how many cases to use in the final evaluation of each "
                         "partition; 0 = all. With tens of thousands of "
                         "cases, evaluating all of them costs more than several "
                         "epochs and does not change the mean appreciably")
    ap.add_argument("--save-every", type=int, default=10,
                    help="saves the resume state every N epochs, besides "
                         "the best checkpoint")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[warning] CUDA unavailable; using CPU.")
        device = "cpu"

    files = list_samples(os.path.join(args.dataset, "samples"))
    if len(files) < 10:
        raise SystemExit(f"too few samples ({len(files)})")
    tr, va = split_files(files, args.val_frac, args.seed)

    # The stats depend only on (training files, target), both fixed
    # by the seed. Reusing them across runs is safe and saves
    # several minutes per start.
    stats = None
    if args.stats_cache and os.path.exists(args.stats_cache):
        import pickle
        with open(args.stats_cache, "rb") as fh:
            c = pickle.load(fh)
        if (c.get("n_tr") == len(tr) and c.get("seed") == args.seed
                and c.get("target") == args.target
                and c.get("val_frac") == args.val_frac):
            stats = c["stats"]
            print(f"[stats] reused from {args.stats_cache}")
        else:
            print("[stats] cache incompatible with these arguments; "
                  "recomputing")
    if stats is None:
        t0 = time.time()
        stats = compute_stats(tr, target=args.target)
        print(f"[stats] computed in {time.time()-t0:.0f}s")
        if args.stats_cache:
            import pickle
            with open(args.stats_cache, "wb") as fh:
                pickle.dump({"stats": stats, "n_tr": len(tr),
                             "seed": args.seed, "target": args.target,
                             "val_frac": args.val_frac}, fh)
            print(f"[stats] saved to {args.stats_cache}")
    print(f"{len(files)} cases | train={len(tr)} val={len(va)} | "
          f"branch_dim={stats['branch_dim']} | device={device}")

    train_ds = SimpleDataset(tr, stats, n_query=args.n_query,
                              repeats=args.repeats, dist=args.dist_furos,
                              hole_frac=args.frac_furos)
    val_ds = SimpleDataset(va, stats, n_query=args.n_query,
                            dist=args.dist_furos)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False)

    model = SimpleDeepONet(branch_dim=stats["branch_dim"], p=args.p,
                            hidden=args.hidden, layers=args.layers,
                            fourier=args.fourier,
                            d_extra=2 if args.dist_furos else 0,
                            decoder=args.decoder,
                            trunk=args.trunk).to(device)
    nparams = sum(q.numel() for q in model.parameters())
    print(f"SimpleDeepONet: p={args.p} hidden={args.hidden} "
          f"layers={args.layers} fourier={args.fourier} "
          f"dist_furos={args.dist_furos} "
          f"frac_furos={args.frac_furos} decoder={args.decoder} "
          f"trunk={args.trunk} "
          f"params={nparams:,}")
    opt = torch.optim.Adam(model.parameters(),
                           lr=(args.lr if args.lr is not None else 1e-3),
                           weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5,
                                                       patience=25,
                                                       min_lr=1e-5)
    loss_fn = nn.MSELoss()

    split = {"treino": [os.path.basename(f) for f in tr],
             "val": [os.path.basename(f) for f in va]}
    best = float("inf")
    ep0 = 1

    # ---- resume ----
    # The checkpoint also stores optimizer, scheduler and epoch, so that
    # an interrupted session continues where it stopped instead of restarting.
    # The split and the stats are those of the checkpoint: recomputing them
    # would give another partition and leak seen cases into validation.
    if args.resume and os.path.exists(args.out):
        ck = torch.load(args.out, map_location="cpu", weights_only=False)
        if "epoch" not in ck:
            print("[warning] old checkpoint, no resume state; "
                  "loading the weights only")
        model.load_state_dict(ck["model"])
        if "opt" in ck:
            opt.load_state_dict(ck["opt"])
        if "sched" in ck:
            sched.load_state_dict(ck["sched"])
        best = ck.get("best", float("inf"))
        ep0 = ck.get("epoch", 0) + 1
        # the Adam state carries the rate at which training stopped; if the user
        # passed --lr explicitly, it overrides the checkpoint's --- and
        # the scheduler restarts, so as not to return to the floor at once
        if args.lr is not None:
            for g in opt.param_groups:
                g["lr"] = args.lr
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt, factor=0.5, patience=25, min_lr=1e-5)
            print(f"[resume] learning rate restarted at {args.lr:g}")
        # The split derives from (files, val_frac, seed), all equal
        # to this run's, so it is already identical to the checkpoint's. Rebuilding
        # the datasets here would duplicate 47 thousand file handles and bring down the
        # process — checking that they match is enough.
        if "split" in ck:
            n_ck = len(ck["split"]["treino"])
            if n_ck != len(tr):
                raise SystemExit(
                    f"checkpoint split ({n_ck} training cases) differs "
                    f"from the current one ({len(tr)}); use the same --val-frac and --seed")
            stats = ck["stats"]
        print(f"[resume] resuming from epoch {ep0} | best val {best:.4e}")
        if ep0 > args.epochs:
            raise SystemExit(f"already trained up to epoch {ep0-1}; increase "
                             f"--epochs to continue")

    def save_state(fpath, ep):
        torch.save({"model": model.state_dict(), "args": vars(args),
                    "stats": stats, "split": split,
                    "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "epoch": ep, "best": best}, fpath)

    for ep in range(ep0, args.epochs + 1):
        t_ep = time.time()
        tr_loss = run_epoch(model, train_dl, opt, loss_fn, device, True)
        va_loss = run_epoch(model, val_dl, opt, loss_fn, device, False)
        sched.step(tr_loss)
        dt = time.time() - t_ep
        msg = (f"época {ep:3d} | {dt:5.1f}s | treino {tr_loss:.4e} "
               f"| val {va_loss:.4e}")
        if ep % args.eval_every == 0 or ep == 1:
            tm, tmd = eval_field(model, tr, stats, device, max_cases=60,
                                 dist=args.dist_furos)
            vm_, vmd = eval_field(model, va, stats, device, max_cases=60,
                                  dist=args.dist_furos)
            msg += (f" | relL2 treino méd={tm:.3f} med={tmd:.3f}"
                    f" | VAL méd={vm_:.3f} med={vmd:.3f}")
        print(msg)
        if va_loss < best:
            best = va_loss
            save_state(args.out, ep)
        # resume state, even when the epoch was not the best
        if args.save_every and ep % args.save_every == 0:
            save_state(args.out + ".last", ep)

    ckpt = torch.load(args.out, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    nfe = args.final_eval or None
    t0 = time.time()
    tm, tmd = eval_field(model, tr, stats, device, max_cases=nfe,
                         dist=args.dist_furos)
    vm_, vmd = eval_field(model, va, stats, device, max_cases=nfe,
                          dist=args.dist_furos)
    how_many = ("all cases" if nfe is None
               else f"{min(nfe, len(tr))} training and "
                    f"{min(nfe, len(va))} validation")
    print(f"\nfinal evaluation over {how_many} ({time.time()-t0:.0f}s)")
    print(f"Best val MSE: {best:.4e}")
    print(f"relL2 — TREINO: méd={tm:.3f} med={tmd:.3f}")
    print(f"relL2 — VAL   : méd={vm_:.3f} med={vmd:.3f}")
    print(f"model saved to {args.out}")


if __name__ == "__main__":
    main()
