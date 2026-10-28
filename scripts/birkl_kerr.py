"""Birkl et al. (2007) spin series: models D, DA1 (ring 6-7.7 M, M = 2 Msun) and REF, A.5, A1
(ring 6-10 M, M = 3 Msun); isothermal non-rotating (zero angular momentum) rings at 5e10 K.

Two sets of integrals:
    birkl_conv   their definitions: rates of static observers, outside the ergosphere
                 (deposition_static; for a = 0 the static observer is the ZAMO and the integral
                 starts at 1.01 r_h, as in Table 1);
    tot, up, ... rates of ZAMOs, down to 1.01 r_h (into the ergosphere), and the exact energy at
                 infinity tot_inf, up_inf.
Writes output/birkl_kerr.json; the published values are printed alongside if
data/published_values.json is present (data/README.md).
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
import annihilus as an  # noqa: E402
from annihilus import deposition as dp, physics as ph, published  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "output")
KB = 8.617333262e-11
MODELS = {  # name: (M/Msun, a, r_out)
    "D": (2.0, 0.0, 7.7),
    "DA1": (2.0, 0.9999, 7.7),
    "REF": (3.0, 0.0, 10.0),
    "A.5": (3.0, 0.5, 10.0),
    "A1": (3.0, 0.9999, 10.0),
}


def main(nr=10, nth=14):
    kT = 5e10 * KB
    De, _ = ph.couplings(0.23)
    res = {}
    published_values = published.get("birkl_2007_table1") or {}
    for name, (Msun, a, rout) in MODELS.items():
        ref = published_values.get(name)
        cfg = an.Config(M=1.0, emitter=an.DISK, r_in=6.0, r_out=rout, keplerian=0, spin=a)
        rh = dp.horizon(1.0, a)
        edges = [1.01 * rh, 0.5 * (1.01 * rh + 6.0), 6.0, rout, 2 * rout, 30.0, 200.0]
        t = time.time()
        d = dp.deposition(cfg, edges, nr=nr, nth=nth)
        U = dp.rate_unit(kT, Msun * ph.GM_SUN, De) / 1e49
        s = dp.deposition_static(cfg, edges, nr=nr, nth=nth) if a > 0 else d
        res[name] = dict(a=a, tot=d["E_loc"] * U, up=d["E_loc_up"] * U,
                         tot_inf_birkl=d["E_birkl"] * U, up_inf_birkl=d["E_birkl_up"] * U,
                         tot_inf=d["E_inf"] * U, up_inf=d["E_inf_up"] * U,
                         birkl_conv=[s[k] * U for k in ("E_loc", "E_loc_up", "E_birkl",
                                                         "E_birkl_up")],
                         inf_outside_ergo=[s["E_inf"] * U, s["E_inf_up"] * U])
        v = res[name]
        print(f"{name:4s} a={a:.4f} ({time.time() - t:.0f} s): static, outside ergosphere "
              + " ".join(f"{x:.3f}" for x in v["birkl_conv"])
              + ("" if ref is None else f" | Birkl {ref}") + "\n"
              f"     ZAMO to 1.01 r_h {v['tot']:.3f} {v['up']:.3f} {v['tot_inf_birkl']:.3f} "
              f"{v['up_inf_birkl']:.3f} | exact inf {v['tot_inf']:.3f} {v['up_inf']:.3f}",
              flush=True)
    with open(os.path.join(OUT, "birkl_kerr.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
