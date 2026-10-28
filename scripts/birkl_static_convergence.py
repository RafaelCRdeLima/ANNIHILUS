"""Section 8.1: convergence of the integrals in the convention of Birkl et al. (static observers,
outside the ergosphere) when the volume grid is refined from 10 x 14 to 14 x 20 nodes, for the
models DA1 and A.5. Prints E_tot, E_up, E_tot_inf, E_up_inf (1e49 erg/s) at both resolutions.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
import annihilus as an  # noqa: E402
from annihilus import deposition as dp, physics as ph  # noqa: E402

kT, De = 5e10 * 8.617333262e-11, ph.couplings(0.23)[0]
for name, Msun, a, rout in (("DA1", 2.0, 0.9999, 7.7), ("A.5", 3.0, 0.5, 10.0)):
    cfg = an.Config(M=1.0, emitter=an.DISK, r_in=6.0, r_out=rout, keplerian=0, spin=a)
    rh = dp.horizon(1.0, a)
    edges = [1.01 * rh, 0.5 * (1.01 * rh + 6.0), 6.0, rout, 2 * rout, 30.0, 200.0]
    U = dp.rate_unit(kT, Msun * ph.GM_SUN, De) / 1e49
    for nr, nth in ((10, 14), (14, 20)):
        s = dp.deposition_static(cfg, edges, nr=nr, nth=nth)
        print(f"{name} {nr}x{nth}: " + " ".join(f"{s[k] * U:.4f}" for k in
                                               ("E_loc", "E_loc_up", "E_birkl", "E_birkl_up")),
              flush=True)
