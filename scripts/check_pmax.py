"""Appendix B: the maximum in Eq. (19) over the continuous interval 0 <= P <= 2/3 versus the
nine-point grid, and the smallest flat-space conversion factor.

For Pbar = P the rates are quadratic in P, so the nine values fix them exactly. For every
configuration, temperature and luminosity ratio and conversion profile, and for both flat-space
references, prints the largest excess of the continuous maximum and the smallest R_flat.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import flavour as F  # noqa: E402

Pf = np.linspace(0, 2 / 3, 2001)
worst = {f: (0.0, None) for f in F.FLAT}
rmin = {f: np.inf for f in F.FLAT}
for model, tag, spin in F.configurations():
    cases = {c: F.Case(F.fname(c, model, spin, tag)) for c in ("gr",) + F.FLAT
             if os.path.exists(F.fname(c, model, spin, tag))}
    flats = [c for c in F.FLAT if c in cases]
    profs = list(F.PROFILES) if cases["gr"].weighted else ["uniform"]
    for tx, teb, gx in F.SWEEP:
        kT = {"e": 4.0, "ebar": 4.0 * teb, "x": 4.0 * tx}
        grey = {"e": 1.0, "ebar": 1.0, "x": gx}
        E = {k: c.moments(kT, grey) for k, c in cases.items()}
        for p in profs:
            rows = F.factors(cases, kT, grey, p, E=E)
            P = np.array([w["P"] for w in rows])
            g = np.polyval(np.polyfit(P, [w["R_gr"] for w in rows], 2), Pf)
            for f in flats:
                fl = np.polyval(np.polyfit(P, [w["R_" + f] for w in rows], 2), Pf)
                excess = np.max(np.abs(g / fl - 1)) - F.maxdev(rows, f)
                if excess > worst[f][0]:
                    worst[f] = (excess, (model, tag, spin, tx, teb, gx, p))
                rmin[f] = min(rmin[f], fl.min())
    print(f"{model}{tag} a={spin}: done", flush=True)
for f in F.FLAT:
    print(f"{f}: largest excess {100 * worst[f][0]:.3f} percentage points at {worst[f][1]}; "
          f"smallest R_flat {rmin[f]:.3f}")
