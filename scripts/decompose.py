"""Decomposition of the sign reversal (layout iii, a = 0.99, T_x = T_e = 4 MeV, T_ebar = 6 MeV).

The deposition is bilinear in the neutrino and antineutrino moments; with nu_e, nubar_e from the
outer disc (o) and nu_x from the inner ring (i), every term is one of four geometric integrals
I_oo = E[K(e, ebar)], I_ii = E[K(x, xbar)], I_oi = E[K(e, xbar)], I_io = E[K(x, ebar)], where
E[.] is the energy at infinity of the pair rate with D = 1. Conversion only changes the
coefficients. Writes output/decompose.json.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "python"))
sys.path.insert(0, HERE)
import flavour as F  # noqa: E402
from annihilus import physics as ph  # noqa: E402

OUT = os.path.join(HERE, "..", "output")


def integrals(case, kT):
    E = case.moments(kT, {s: 1.0 for s in F.SPECIES})
    E = {k: v[..., 0, :, :] for k, v in E.items()}

    def I(a, b):
        q = ph.qdot(E[a], E[b], 1.0)
        p = ph.pdot(E[a], E[b], 1.0)
        return float(np.sum(case.W * (case.al * q + case.om_sg * p[..., 2])))
    return dict(oo=I("e", "ebar"), ii=I("x", "x"), oi=I("e", "x"), io=I("x", "ebar"))


def coefficients(P, Pb):
    De, Dx = ph.couplings(0.23)
    return dict(oo=De * (1 - P) * (1 - Pb) + 2 * Dx * (P / 2) * (Pb / 2),
                ii=De * P * Pb + 2 * Dx * (1 - P / 2) * (1 - Pb / 2),
                oi=De * (1 - P) * Pb + 2 * Dx * (P / 2) * (1 - Pb / 2),
                io=De * P * (1 - Pb) + 2 * Dx * (1 - P / 2) * (Pb / 2))


def main():
    kT = {"e": 4.0, "ebar": 6.0, "x": 4.0}
    out = {}
    for case in ("gr",) + F.FLAT:
        I = integrals(F.Case(F.fname(case, "disjoint", 0.99, "")), kT)
        c0, c1 = coefficients(0, 0), coefficients(2 / 3, 2 / 3)
        E0 = sum(c0[k] * I[k] for k in I)
        E1 = sum(c1[k] * I[k] for k in I)
        out[case] = dict(I=I, E0=E0, E1=E1, R=E1 / E0,
                         share0={k: c0[k] * I[k] / E0 for k in I},
                         share1={k: c1[k] * I[k] / E0 for k in I})
    g = out["gr"]
    for ref in F.FLAT:
        f = out[ref]
        print(f"{ref}: integral   gain(GR/flat)   share of E(0): GR  flat   "
              "share of E(2/3) [in units of E(0)]: GR  flat")
        for k in ("oo", "ii", "oi", "io"):
            print(f"  {k}      {g['I'][k] / f['I'][k]:6.2f}         {g['share0'][k]:6.3f} "
                  f"{f['share0'][k]:6.3f}            {g['share1'][k]:6.3f} {f['share1'][k]:6.3f}")
        print(f"R(2/3): GR {g['R']:.3f}  flat {f['R']:.3f}; cross (oi+io) gain "
              f"{(g['I']['oi'] + g['I']['io']) / (f['I']['oi'] + f['I']['io']):.2f}; "
              f"cross share at P = 2/3: GR {g['share1']['oi'] + g['share1']['io']:.3f} "
              f"flat {f['share1']['oi'] + f['share1']['io']:.3f}")
    with open(os.path.join(OUT, "decompose.json"), "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
