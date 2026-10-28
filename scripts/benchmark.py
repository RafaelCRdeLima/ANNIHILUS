"""Cost of the angular integration: direct double sum versus moment contraction.

Rays are sampled on a Fibonacci sphere (equal weights) at one point above a Keplerian disc
around a Kerr hole; the same ray set feeds both evaluations, so they agree to rounding.
Times are separated into ray tracing, direct double sum and moment contraction. The adaptive
quadrature of the engine is timed on the same point for comparison.
Writes output/benchmark.json.
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
import annihilus as an  # noqa: E402
from annihilus import physics as ph  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "output")


def fibonacci(n):
    k = np.arange(n) + 0.5
    mu = 1 - 2 * k / n
    ph_ = np.mod(np.pi * (1 + 5**0.5) * k, 2 * np.pi)
    return mu, ph_


def direct(n, w, G3, G4, chunk=2000):
    tot = 0.0
    for i in range(0, len(w), chunk):
        c = n[i:i + chunk] @ n.T
        k = (1 - c)**2
        a3, a4 = w[i:i + chunk] * G3[i:i + chunk], w[i:i + chunk] * G4[i:i + chunk]
        tot += a4 @ k @ (w * G3) + a3 @ k @ (w * G4)
    return tot


def moments_sum(n, w, G3, G4):
    def block(G):
        a = w * G
        m = np.zeros(20)
        m[0] = a.sum()
        m[1:4] = n.T @ a
        k = 4
        for i in range(3):
            for j in range(i, 3):
                m[k] = a @ (n[:, i] * n[:, j]); k += 1
        return m
    return ph.kernel(block(G4), block(G3)) + ph.kernel(block(G3), block(G4))


def main():
    a, r, th = 0.9, 6.0, 0.8
    cfg = an.Config(M=1.0, emitter=an.DISK, r_in=2.321, r_out=40.0, keplerian=1, t_index=0.5,
                    spin=a)
    t = time.perf_counter()
    mom, nr = an.moments(cfg, [r], [th], nthreads=1)
    t_eng = time.perf_counter() - t
    q_ref = ph.qdot_geometric(mom)[0]
    rows = []
    for N in (100, 1000, 10000, 100000):
        mu, ph_ = fibonacci(N)
        t = time.perf_counter()
        h = an.trace_many(cfg, r, th, mu, ph_, nthreads=1)
        t_tr = time.perf_counter() - t
        s = np.sqrt(1 - mu**2)
        n = np.stack([mu, s * np.cos(ph_), s * np.sin(ph_)], axis=1)
        gt = h["g"] * h["tau"]
        G3, G4, w = gt**4, gt**5, np.full(N, 4 * np.pi / N)
        t = time.perf_counter()
        qm = moments_sum(n, w, G3, G4)
        t_mom = time.perf_counter() - t
        if N <= 10000:
            t = time.perf_counter()
            qd = direct(n, w, G3, G4)
            t_dir = time.perf_counter() - t
        else:
            qd, t_dir = np.nan, np.nan
        rows.append(dict(N=N, t_trace=t_tr, t_direct=t_dir, t_moments=t_mom, q_moments=qm,
                         q_direct=qd, rel_diff=abs(qm - qd) / abs(qd) if N <= 10000 else None,
                         err_vs_adaptive=abs(qm - q_ref) / q_ref))
        print(f"N={N:6d}: trace {t_tr * 1e3:8.2f} ms  direct {t_dir * 1e3:9.2f} ms  "
              f"moments {t_mom * 1e3:6.3f} ms  |direct-moments|/q = {rows[-1]['rel_diff']}  "
              f"error vs adaptive = {rows[-1]['err_vs_adaptive']:.2e}", flush=True)
    print(f"adaptive quadrature: {nr} rays, {t_eng * 1e3:.0f} ms, q = {q_ref:.10e}")
    with open(os.path.join(OUT, "benchmark.json"), "w") as f:
        json.dump(dict(point=dict(a=a, r=r, th=th), adaptive=dict(nrays=nr, time=t_eng, q=q_ref),
                       rows=rows), f, indent=1)


if __name__ == "__main__":
    main()
