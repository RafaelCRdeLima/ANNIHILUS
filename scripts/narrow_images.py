"""Are narrow (higher-order) disc images captured by the edge-adapted quadrature?

For points above a Keplerian disc (6 M to 40 M, T ~ r^-1/2) around Schwarzschild and Kerr
(a = 0.9) holes: (1) the moments as the coarse scan in mu is refined; (2) the contribution of
images of order >= 1 (rays that cross the equatorial plane outside the disc before hitting it)
from the quadrature, as the difference between max_orders = 8 and 0, against brute-force
sampling with 2.6e6 rays on a uniform (mu, phi) grid. Writes output/narrow_images.json.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
import annihilus as an  # noqa: E402
from annihilus import physics as ph  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "output")
POINTS = [(30.0, 0.15), (8.0, 1.3), (3.0, 0.8)]


def main():
    res = []
    for a, rin in ((0.0, 6.0), (0.9, 2.321)):
        for r, th in POINTS:
            base = dict(M=1.0, emitter=an.DISK, r_in=rin, r_out=40.0, keplerian=1, t_index=0.5,
                        spin=a)
            scan = {}
            for ns in (16, 32, 64, 256):
                m, _ = an.moments(an.Config(n_scan=ns, **base), [r], [th])
                scan[ns] = float(m[0, 0, 0])
            m8, _ = an.moments(an.Config(**base), [r], [th])
            m0, _ = an.moments(an.Config(max_orders=0, **base), [r], [th])
            hi_quad = float(m8[0, 0, 0] - m0[0, 0, 0])
            # brute force on a uniform grid in (mu, phi), cell centres
            nmu, nph = 5120, 512
            mu = -1 + (np.arange(nmu) + 0.5) * 2 / nmu
            phg = (np.arange(nph) + 0.5) * 2 * np.pi / nph
            MU, PH = np.meshgrid(mu, phg, indexing="ij")
            h = an.trace_many(an.Config(**base), r, th, MU.ravel(), PH.ravel())
            gt = h["g"] * h["tau"]
            hit = h["status"] == an.HIT_DISK
            w = (2 / nmu) * (2 * np.pi / nph)
            G3 = np.where(hit, gt**4, 0.0)
            tot_bf = float(G3.sum() * w)
            hi_bf = float(G3[h["order"] >= 1].sum() * w)
            row = dict(a=a, r=r, th=th, scan=scan, M0=float(m8[0, 0, 0]), M0_brute=tot_bf,
                       higher_quad=hi_quad, higher_brute=hi_bf,
                       higher_fraction=hi_quad / float(m8[0, 0, 0]))
            res.append(row)
            sc = " ".join(f"{k}:{v / scan[256] - 1:+.1e}" for k, v in scan.items())
            print(f"a={a} r={r} th={th}: scan(rel. to 256) {sc}; M0 quad {m8[0, 0, 0]:.6e} brute "
                  f"{tot_bf:.6e}; order>=1: quad {hi_quad:.4e} brute {hi_bf:.4e} "
                  f"(fraction {hi_quad / m8[0, 0, 0]:.2e})", flush=True)
    with open(os.path.join(OUT, "narrow_images.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
