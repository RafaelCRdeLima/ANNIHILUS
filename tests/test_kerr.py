import os
import sys
import unittest

import numpy as np

import annihilus as an
from annihilus import physics as ph

sys.path.insert(0, os.path.dirname(__file__))
from flat_disc import direct_moments  # noqa: E402
from kerr_hamilton import first_crossing, ginv  # noqa: E402


class TestKerr(unittest.TestCase):
    def test_zero_spin_matches_schwarzschild(self):
        r = np.array([3.0, 8.0, 15.0]); th = np.array([0.4, 1.2, 0.8])
        for kep in (0, 1):
            base = dict(M=1.0, emitter=an.DISK, r_in=6.0, r_out=30.0, keplerian=kep, t_index=0.5)
            ms, _ = an.moments(an.Config(**base), r, th)
            mk, _ = an.moments(an.Config(metric=1, **base), r, th)
            np.testing.assert_allclose(ph.qdot_geometric(mk), ph.qdot_geometric(ms), rtol=1e-6)

    def test_reflection_symmetry(self):
        cfg = an.Config(M=1.0, emitter=an.DISK, r_in=2.32, r_out=30.0, keplerian=1, spin=0.9)
        mom, _ = an.moments(cfg, [5.0, 5.0], [0.7, np.pi - 0.7])
        sign = np.array([(-1) ** c.count("th") for c in an.COMPONENTS])
        np.testing.assert_allclose(mom[0], mom[1] * sign, rtol=1e-7, atol=1e-12 * mom[0, 0, 0])

    def test_flat_oblate(self):
        # Kerr with M = 0 is flat space in oblate spheroidal coordinates: every spin term of the
        # geodesic equations is active while the rays are straight lines
        a, rin, rout = 0.8, 1.5, 5.0
        cfg = an.Config(M=0.0, emitter=an.DISK, r_in=rin, r_out=rout, keplerian=0, spin=a,
                        h=0.005)  # RK4 error 2e-6 at h = 0.01, scaling as h^4
        for r, th in [(3.0, 0.4), (8.0, 1.1)]:
            mom, _ = an.moments(cfg, [r], [th])
            d = direct_moments(r, th, rin, rout, a=a)
            self.assertLess(np.max(np.abs(mom[0, 0, :10] - d)) / d[0], 1e-6, (r, th))


class TestKerrIntegrals(unittest.TestCase):
    def test_first_integrals(self):
        # (dr/dsigma)^2 = R(r) and (dx/dsigma)^2 = Theta_x(x) at the start and along the ray;
        # this fixes the sign conventions of R, the Carter constant and the ZAMO initial data
        cfg = an.Config(M=1.0, emitter=an.DISK, r_in=2.32, r_out=30.0, spin=0.9)
        for r, th, mu, ph, s in [(5.0, 0.7, 0.3, 1.1, 20.0), (3.0, 1.2, -0.6, 4.0, 50.0),
                                 (12.0, 0.2, 0.9, 2.5, 10.0)]:
            res = an.kerr_residual(cfg, r, th, mu, ph, s)
            self.assertLess(np.max(np.abs(res[:2])), 1e-12)
            self.assertLess(np.max(np.abs(res[2:])), 1e-7)


class TestIndependentKerr(unittest.TestCase):
    def test_crossing_radii(self):
        # first equatorial crossing from the Mino-time tracer versus Hamilton's equations with the
        # full inverse metric: checks R(r), Theta(theta), the ZAMO initial data and the events
        a = 0.9
        rp = 1 + np.sqrt(1 - a * a)
        cfg = an.Config(M=1.0, emitter=an.DISK, r_in=1.0001 * rp, r_out=1e3, spin=a, h=0.005)
        rays = [(6.0, 0.6, 0.8, 0.3), (6.0, 0.6, 0.95, 2.0), (10.0, 1.0, 0.2, 4.0),
                (4.0, 0.4, 0.9, 1.2), (20.0, 0.3, -0.3, 0.7), (3.0, 1.3, 0.6, 5.5),
                (8.0, 0.9, 0.5, 3.0), (15.0, 1.2, 0.1, 1.6)]
        for r, th, mu, ph in rays:
            h = an.trace(cfg, r, th, mu, ph)
            ref = first_crossing(1.0, a, r, th, mu, ph)
            if ref is None:
                self.assertNotEqual(h["status"], an.HIT_DISK, (r, th, mu, ph))
            else:
                self.assertEqual(h["status"], an.HIT_DISK, (r, th, mu, ph))
                self.assertAlmostEqual(h["r_em"] / ref, 1.0, places=6, msg=(r, th, mu, ph))

    def test_killing_projection(self):
        # -Q_t from the ZAMO components by explicit tetrad and metric, against the formula used
        # for the energy at infinity; and sqrt(-g) = Sigma sin(theta)
        from annihilus import deposition as dp
        rng = np.random.default_rng(1)
        for _ in range(5):
            a, r, th = rng.uniform(0, 0.99), rng.uniform(1.5, 20), rng.uniform(0.2, 1.4)
            if r < 1 + np.sqrt(1 - a * a) + 0.1:
                continue
            q, pr, pt, pp = rng.normal(size=4)
            g = np.linalg.inv(ginv(1.0, a, r, th))
            al, om, sg, sqg, sq3 = dp.metric_factors(1.0, a, r, th)
            Sig = r * r + a * a * np.cos(th)**2
            Del = r * r - 2 * r + a * a
            Qu = (q * np.array([1 / al, 0, 0, om / al]) + pr * np.array([0, np.sqrt(Del / Sig), 0, 0])
                  + pt * np.array([0, 0, 1 / np.sqrt(Sig), 0]) + pp * np.array([0, 0, 0, 1 / sg]))
            Qt = (g @ Qu)[0]
            self.assertAlmostEqual(-Qt, al * q + om * sg * pp, places=10)
            self.assertAlmostEqual(np.sqrt(-np.linalg.det(g)) / sqg, 1.0, places=10)
            self.assertAlmostEqual(np.sqrt(np.linalg.det(g[1:, 1:])) / sq3, 1.0, places=10)


if __name__ == "__main__":
    unittest.main()
