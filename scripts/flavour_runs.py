"""Moments on a common (r, th) grid for discs with species-dependent neutrinospheres.

Toy post-merger disc around a black hole of spin a, Keplerian rotation, temperature profile
T_s(r) = T_s,in (r/r_isco)^(-s_T) on the emitting range of each species. Cases:
    gr       relativistic (Schwarzschild or Kerr), Keplerian rotation;
    newt     flat space (M = 0), same discs, static;
    flatrot  flat space, same discs, Newtonian Keplerian velocities of the same mass,
             v = sqrt(M/r) (special-relativistic Doppler shifts only);
    flatloc  flat space, same discs, rotating at the local speeds of the relativistic disc: the
             speed of the Kerr Keplerian orbit relative to the ZAMO at the same coordinate radius,
             which isolates the effect of gravity at fixed local kinematics.
Models (emitting ranges in units of r_isco, outer radii in M):
    default   e: [1, 40M], ebar: [1, 25M], x: [1, 12M]
    xinner    x only from [1, 2]
    disjoint  x from [1, 2]; e and ebar only beyond 2 r_isco
Temperature profiles: power law with exponent s_T (default 0.5), or the Page-Thorne profile of a
radiatively efficient disc, T proportional to F^(1/4), normalised to the peak (--profile pt).
The moments carry four weight blocks (1, rho, mu_em, rho mu_em) with rho = exp(-(r_em-r_isco)/r_isco),
for flavour conversion that depends on the emission radius and angle.
Writes output/flavour_<case>[_<model>][_sT<s>|_pt]_a<spin>.npz.
Usage: flavour_runs.py [--model M] [--cases gr,newt,flatrot,flatloc] [--profile pl|pt] [--sT s]
                      [--refine] [spin ...]
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
import annihilus as an  # noqa: E402
from annihilus import deposition as dp  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "output")
S_T = 0.5
SPECIES = ("e", "ebar", "x")


def r_isco(a):
    z1 = 1 + (1 - a * a)**(1 / 3) * ((1 + a)**(1 / 3) + (1 - a)**(1 / 3))
    z2 = np.sqrt(3 * a * a + z1 * z1)
    return 3 + z2 - np.sqrt((3 - z1) * (3 + z1 + 2 * z2))


def ranges(model, ri):
    """Emitting range (r_in, r_out) of each species."""
    rng = {"e": (ri, 40.0), "ebar": (ri, 25.0), "x": (ri, 12.0)}
    if model in ("xinner", "disjoint"):
        rng["x"] = (ri, 2 * ri)
    if model == "disjoint":
        rng["e"] = (2 * ri, 40.0)
        rng["ebar"] = (2 * ri, 25.0)
    return rng


def fname(case, model, spin, profile="pl", s_T=S_T, refine=False):
    tag = "" if model == "default" else "_" + model
    if profile == "pt":
        tag += "_pt"
    elif s_T != S_T:
        tag += f"_sT{s_T:g}"
    if refine:
        tag += "_ref"
    return os.path.join(OUT, f"flavour_{case}{tag}_a{spin:.2f}.npz")


def run(spin=0.0, model="default", cases=("gr", "newt"), profile="pl", s_T=S_T, nr=10, nth=16,
        refine=False):
    """refine: doubled azimuths and Gauss-Legendre nodes, finer volume grid (convergence of R)."""
    quad = dict(n_phi=128, n_gl=32) if refine else {}
    if refine:
        nr, nth = 14, 24
    os.makedirs(OUT, exist_ok=True)
    ri = r_isco(spin)
    rng = ranges(model, ri)
    rh = dp.horizon(1.0, spin)
    edges = {1.01 * rh, 0.5 * (1.01 * rh + ri), ri, 12.0, 25.0, 40.0, 80.0, 300.0}
    if model != "default":
        edges.add(2 * ri)
    edges = sorted(edges)
    setups = {"gr": (1.0, 1, spin, 0.0, 0), "newt": (0.0, 0, 0.0, 0.0, 0),
              "flatrot": (0.0, 1, 0.0, 1.0, 0), "flatloc": (0.0, 1, 0.0, 1.0, 1)}
    rr = np.linspace(ri, 60.0, 20000)
    t_norm = float(np.max(an.page_thorne(spin, rr))) if profile == "pt" else 1.0
    for case in cases:
        M, kep, a, mrot, law = setups[case]
        data = {}
        for sp in SPECIES:
            rin_s, rout_s = rng[sp]
            cfg = an.Config(M=M, emitter=an.DISK, r_in=rin_s, r_out=rout_s, keplerian=kep,
                            t_index=s_T, spin=a, metric=int(spin != 0.0 and M > 0), m_rot=mrot,
                            rot_law=law, rot_spin=spin, n_w=4, l_p=ri, r_p=ri,
                            profile=int(profile == "pt"), pt_spin=spin, t_norm=t_norm, **quad)
            t = time.time()
            d = dp.deposition(cfg, edges, nr=nr, nth=nth)
            print(f"a={spin} {model} {profile} sT={s_T} {case} {sp}: {time.time() - t:.0f} s, "
                  f"{d['nrays'] / 1e6:.0f} M rays", flush=True)
            data["mom_" + sp] = d["mom"]
        data.update(r=d["r"], th=d["th"], M=M, spin=a, rot_law=law, edges=np.array(edges),
                    nr=nr, nth=nth, r_in=ri, s_T=s_T, model=model, profile=profile, l_p=ri, r_p=ri,
                    r_in_sp=np.array([rng[k][0] for k in SPECIES]),
                    r_out=np.array([rng[k][1] for k in SPECIES]))
        np.savez(fname(case, model, spin, profile, s_T, refine), **data)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="default")
    ap.add_argument("--cases", default="gr,newt")
    ap.add_argument("--profile", default="pl")
    ap.add_argument("--sT", type=float, default=S_T)
    ap.add_argument("--refine", action="store_true")
    ap.add_argument("spins", nargs="*", default=["0"])
    args = ap.parse_args()
    for a in args.spins:
        run(float(a), args.model, args.cases.split(","), args.profile, args.sT, refine=args.refine)
