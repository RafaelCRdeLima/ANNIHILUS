"""Convergence of the global energy at infinity with the numerical parameters.

Keplerian disc from the ISCO to 40 M around a Kerr hole (a = 0.9), T proportional to r^(-1/2),
nu_e and nubar_e on the same disc. Each line changes one parameter from the baseline.
Writes output/convergence.json.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
import annihilus as an  # noqa: E402
from annihilus import deposition as dp  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "output")
A, RI = 0.9, 2.320883
BASE = dict(n_scan=64, n_gl=16, n_phi=64, h=0.01, nr=10, nth=16)
VARIANTS = [("baseline", {}), ("n_scan", {"n_scan": 32}), ("n_scan", {"n_scan": 128}),
            ("n_gl", {"n_gl": 8}), ("n_gl", {"n_gl": 32}), ("n_phi", {"n_phi": 32}),
            ("n_phi", {"n_phi": 128}), ("h", {"h": 0.02}), ("h", {"h": 0.005}),
            ("nr", {"nr": 7}), ("nr", {"nr": 14}), ("nth", {"nth": 12}), ("nth", {"nth": 24})]


def run(p):
    cfg = an.Config(M=1.0, emitter=an.DISK, r_in=RI, r_out=40.0, keplerian=1, t_index=0.5,
                    spin=A, n_scan=p["n_scan"], n_gl=p["n_gl"], n_phi=p["n_phi"], h=p["h"])
    rh = dp.horizon(1.0, A)
    edges = [1.01 * rh, 0.5 * (1.01 * rh + RI), RI, 12.0, 25.0, 40.0, 80.0, 300.0]
    d = dp.deposition(cfg, edges, nr=p["nr"], nth=p["nth"])
    return d["E_inf"], d["E_inf_up"], d["nrays"]


def main():
    rows, ref = [], None
    for name, change in VARIANTS:
        p = dict(BASE, **change)
        t = time.time()
        e, eu, nr = run(p)
        ref = ref or e
        rows.append(dict(name=name, change=change, E_inf=e, E_inf_up=eu, nrays=nr,
                         rel=e / ref - 1, time=time.time() - t))
        print(f"{name:9s} {str(change):18s} E_inf {e:.6e}  rel {e / ref - 1:+.2e}  "
              f"up {eu:.6e}  {nr / 1e6:.0f} M rays  {time.time() - t:.0f} s", flush=True)
    with open(os.path.join(OUT, "convergence.json"), "w") as f:
        json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
