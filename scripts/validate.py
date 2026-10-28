"""Benchmarks: Salmonson & Wilson (1999) sphere (Fig. 1) and Birkl et al. (2007) models D, DN,
S, SN (Table 1).

Writes output/validation.json and prints a table; the published values are printed alongside if
data/published_values.json is present (data/README.md).
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
import annihilus as an  # noqa: E402
from annihilus import deposition as dp, physics as ph, published  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "output")
KB = 8.617333262e-11  # MeV/K


def sphere():
    rows = []
    for RM in (10.0, 5.0, 4.0, 3.5, 3.0):
        cfg = an.Config(M=1.0, emitter=an.SPHERE, R=RM, n_phi=4)

        def qgeo(r):
            mom, _ = an.moments(cfg, r, np.ones_like(r))
            return ph.qdot_geometric(mom)
        rows.append(dict(R_over_M=RM, F_analytic=ph.sphere_enhancement(RM),
                         F_engine=ph.sphere_enhancement(RM, qgeo=qgeo)))
    return rows


def birkl(nr=12, nth=16):
    """Isothermal non-rotating ring 6-7.7 M around M = 2 Msun, T = 5e10 K, nu_e only."""
    kT, Mcm = 5e10 * KB, 2 * ph.GM_SUN
    De, _ = ph.couplings(0.23)
    U = dp.rate_unit(kT, Mcm, De) / 1e49
    edges = [2.0 * 1.01, 3.0, 6.0, 7.7, 12.0, 30.0, 200.0]
    res = {}
    for name, M in (("D", 1.0), ("DN", 0.0)):
        cfg = an.Config(M=M, emitter=an.DISK, r_in=6.0, r_out=7.7, keplerian=0)
        d = dp.deposition(cfg, edges, nr=nr, nth=nth)
        R, _ = np.meshgrid(d["r"], d["th"], indexing="ij")
        r, wr = dp.radial_nodes(np.array(edges), nr, 2.0 if M > 0 else 0.0)
        _, wt = dp.polar_nodes(nth)
        W = 4 * np.pi * np.outer(wr, wt) * R**2
        up = d["pgeo"][..., 0] > 0
        res[name] = dict(E_tot=d["E_loc"] * U, E_up=d["E_loc_up"] * U,
                         E_tot_inf_birkl=np.sum(W * d["qgeo"]) * U,
                         E_up_inf_birkl=np.sum((W * d["qgeo"])[up]) * U,
                         E_tot_inf=d["E_inf"] * U, E_up_inf=d["E_inf_up"] * U)
    # spheres S (GR) and SN (Newtonian), R = 3.4 M: proper-volume rate
    Rcm = 3.4 * Mcm
    r, w = ph.sphere_radii(1.0, 64)
    I = np.sum(w * ph.theta_sphere(ph.x_sphere(r, 1.0)))
    SN = ph.qdot_thermal(kT, 1.0, De) * 4 * np.pi * Rcm**3 * I * ph.MEV_ERG / 1e49
    AR = 1 - 2 / 3.4
    S = SN * ph.sphere_enhancement(3.4, n=64) * AR**2.25   # fixed T: L_inf scales as A_R
    res["S"] = dict(E_tot=S)
    res["SN"] = dict(E_tot=SN)
    return res


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    s = sphere()
    sw = published.get("salmonson_wilson_1999") or {}
    print("Sphere, F(M/R) at fixed L_inf")
    for row in s:
        quoted = dict(zip(sw.get("R_over_M", []), sw.get("F", []))).get(row["R_over_M"])
        print("  R/M = {R_over_M:4.1f}   analytic {F_analytic:8.4f}   engine {F_engine:8.4f}"
              .format(**row) + ("" if quoted is None else f"   S&W {quoted}"))
    b = birkl()
    ref = published.get("birkl_2007_table1") or {}
    print("Birkl et al. 2007 models [1e49 erg/s]: E_tot E_up E_tot_inf E_up_inf")
    for k in ("D", "DN"):
        v = b[k]
        print(f"  {k:3s} ours (their convention): {v['E_tot']:.3f} {v['E_up']:.3f} "
              f"{v['E_tot_inf_birkl']:.3f} {v['E_up_inf_birkl']:.3f}"
              + (f"   Birkl: {ref[k]}" if k in ref else ""))
        print(f"      exact energy at infinity: {v['E_tot_inf']:.3f} {v['E_up_inf']:.3f}")
    for k in ("S", "SN"):
        print(f"  {k:3s} ours {b[k]['E_tot']:.4f}" + (f"   Birkl: {ref[k]}" if k in ref else ""))
    with open(os.path.join(OUT, "validation.json"), "w") as f:
        json.dump(dict(sphere=s, birkl=b), f, indent=1)
