import collections, numpy as np, sys
rows = collections.defaultdict(list)
for ln in open(sys.argv[1]):
    p = ln.split()
    if len(p) >= 10 and p[0] == "val":
        try: rows[p[2]].append((float(p[6]), int(p[1]), float(p[7]), float(p[9])))
        except ValueError: pass
print(f"{'fam':4s} {'mean relL2':>10s} {'med |peak|':>11s} {'bias':>8s} {'n':>6s}")
all_rows = []
for fam in ("L","Z","U","T","O","G"):
    v = rows[fam]; all_rows += v
    if not v: continue
    r = np.array([x[0] for x in v]); q = np.array([x[3] for x in v])
    print(f"{fam:4s} {r.mean():10.3f} {np.median(np.abs(q)):10.1f}% "
          f"{np.median(q):+7.1f}% {len(v):6d}")
if all_rows:
    r = np.array([x[0] for x in all_rows]); q = np.array([x[3] for x in all_rows])
    print(f"{'GLOB':4s} {r.mean():10.3f} {np.median(np.abs(q)):10.1f}% "
          f"{np.median(q):+7.1f}% {len(all_rows):6d}\n")
for fam in ("L","Z","U","T","O","G"):
    v = rows[fam]
    if not v: continue
    mr = np.median([x[0] for x in v]); mp = np.median([abs(x[3]) for x in v])
    c = [x for x in v if abs(x[0]-mr) < 0.03 and x[2] >= 50.0]
    c.sort(key=lambda x: abs(abs(x[3]) - mp))
    if c:
        print(f"# {fam}: --caso {c[0][1]}  relL2 {c[0][0]:.3f}  "
              f"peak {c[0][2]:.1f} MPa  error {c[0][3]:+.1f}%")
