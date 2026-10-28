"""Dependence of the volume integrals on the inner boundary for a nearly extremal black hole
(models DA1 and A1 of Birkl et al. 2007: zero-angular-momentum rings 6-7.7 M and 6-10 M around holes
of 2 and 3 Msun, a = 0.9999). The rates are those of ZAMOs, integrated into the ergosphere: E_loc in
proper volume, int sqrt(-g) qdot_ZAMO (the integrand of Birkl et al., who use static observers
outside the ergosphere instead) and the exact energy at infinity E_inf (Section 8.1, Table 2 at
1.0001 r_+, Appendix A)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
import annihilus as an  # noqa: E402
from annihilus import deposition as dp, physics as ph  # noqa: E402

a = 0.9999
kT, De = 5e10 * 8.617333262e-11, ph.couplings(0.23)[0]
rh = dp.horizon(1.0, a)
for name, Msun, rout in (("DA1", 2.0, 7.7), ("A1", 3.0, 10.0)):
    U = dp.rate_unit(kT, Msun * ph.GM_SUN, De) / 1e49
    cfg = an.Config(M=1.0, emitter=an.DISK, r_in=6.0, r_out=rout, keplerian=0, spin=a)
    for f in (1.1, 1.01, 1.001, 1.0001):
        edges = [f * rh, 0.5 * (f * rh + 6.0), 6.0, rout, 2 * rout, 30.0, 200.0]
        d = dp.deposition(cfg, edges, nr=10, nth=14)
        print(f"{name} a={a} inner={f}: E_loc {d['E_loc'] * U:9.3f}  "
              f"sqrt(-g) qdot_ZAMO {d['E_birkl'] * U:8.3f}  E_inf {d['E_inf'] * U:.4f}  "
              f"E_inf_up {d['E_inf_up'] * U:.4f}", flush=True)
