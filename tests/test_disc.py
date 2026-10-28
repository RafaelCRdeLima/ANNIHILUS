import os
import sys
import unittest

import numpy as np

import annihilus as an

sys.path.insert(0, os.path.dirname(__file__))
from flat_disc import direct_moments  # noqa: E402


class TestDisc(unittest.TestCase):
    def test_flat_annulus(self):
        rin, rout = 1.0, 5.0
        cfg = an.Config(M=0.0, emitter=an.DISK, r_in=rin, r_out=rout, keplerian=0)
        for r, th in [(3.0, 0.3), (8.0, 1.2), (20.0, 0.7), (200.0, 1.0)]:
            mom, _ = an.moments(cfg, [r], [th])
            d = direct_moments(r, th, rin, rout)
            self.assertLess(np.max(np.abs(mom[0, 0, :10] - d)) / d[0], 1e-6, (r, th))
            # flat static disc: G4 = G3
            np.testing.assert_allclose(mom[0, 1], mom[0, 0], atol=1e-14)

    def test_reflection_symmetry(self):
        cfg = an.Config(M=1.0, emitter=an.DISK, r_in=6.0, r_out=50.0, keplerian=1, t_index=0.75)
        mom, _ = an.moments(cfg, [12.0, 12.0], [0.6, np.pi - 0.6])
        a, b = mom[0], mom[1]
        # (-1)^(number of theta indices)
        sign = np.array([(-1) ** c.count("th") for c in an.COMPONENTS])
        np.testing.assert_allclose(a, b * sign, rtol=1e-7, atol=1e-12 * a[0, 0])

    def test_keplerian_azimuthal_flux(self):
        # prograde rotation blueshifts neutrinos with L_z > 0: net flux along +e_phi
        cfg = an.Config(M=1.0, emitter=an.DISK, r_in=6.0, r_out=30.0, keplerian=1)
        mom, _ = an.moments(cfg, [10.0], [0.5])
        self.assertGreater(mom[0, 1, 3], 0.0)
        cfg.keplerian = 0
        mom0, _ = an.moments(cfg, [10.0], [0.5])
        self.assertLess(abs(mom0[0, 1, 3]), 1e-12 * mom0[0, 1, 0])

    def test_flat_rotating_disc(self):
        # flat space with Keplerian velocities: m_rot = 0 is the static disc; rotation gives a
        # net azimuthal flux and the transverse Doppler redshift lowers the rate on the axis
        base = dict(M=0.0, emitter=an.DISK, r_in=6.0, r_out=30.0, keplerian=1)
        m0, _ = an.moments(an.Config(m_rot=0.0, **base), [10.0, 20.0], [0.5, 0.01])
        ms, _ = an.moments(an.Config(keplerian=0, **{k: v for k, v in base.items()
                                                     if k != "keplerian"}), [10.0, 20.0], [0.5, 0.01])
        np.testing.assert_allclose(m0, ms, rtol=1e-12, atol=1e-15)
        mr, _ = an.moments(an.Config(m_rot=1.0, **base), [10.0, 20.0], [0.5, 0.01])
        self.assertGreater(mr[0, 1, 3], 0.0)
        self.assertLess(mr[1, 1, 0], ms[1, 1, 0])

    def test_flat_local_velocities(self):
        # rot_law = 1: speed of the Kerr Keplerian orbit relative to the ZAMO at the same r,
        # sqrt(M/(r - 2M)) for a = 0; g = sqrt(1 - v^2)/(1 - v lam/r_em) on straight rays
        def v_zamo(a, r):
            D, A = r * r - 2 * r + a * a, (r * r + a * a)**2 - a * a * (r * r - 2 * r + a * a)
            return (1 / (r**1.5 + a) - 2 * a * r / A) * A / (r * r * np.sqrt(D))
        self.assertAlmostEqual(v_zamo(0.0, 6.0), 0.5, places=14)
        r, th = 9.0, 0.9
        for a in (0.0, 0.9, 0.99):
            cfg = an.Config(M=0.0, emitter=an.DISK, r_in=2.0, r_out=30.0, keplerian=1, m_rot=1.0,
                            rot_law=1, rot_spin=a)
            for mu, ph in [(-0.3, 2.5), (0.3, 2.0), (0.0, 2.8), (-0.2, 3.6), (0.4, 4.0)]:
                h = an.trace(cfg, r, th, mu, ph)
                self.assertEqual(h["status"], an.HIT_DISK)
                lam = r * np.sqrt(1 - mu * mu) * np.sin(th) * np.sin(ph)
                v = v_zamo(a, h["r_em"])
                self.assertAlmostEqual(h["g"], np.sqrt(1 - v * v) / (1 - v * lam / h["r_em"]),
                                       places=12)

    def test_weight_blocks(self):
        # n_w = 4: block 0 equals the unweighted moments; in flat space mu_em is |n_z|
        rin, rout = 1.0, 5.0
        base = dict(M=0.0, emitter=an.DISK, r_in=rin, r_out=rout, keplerian=0)
        m1, _ = an.moments(an.Config(**base), [8.0], [1.2])
        m4, _ = an.moments(an.Config(n_w=4, l_p=3.0, r_p=rin, **base), [8.0], [1.2])
        np.testing.assert_allclose(m4[0, 0], m1[0], rtol=1e-14, atol=1e-16)
        d = direct_moments(8.0, 1.2, rin, rout, cos_weight=True)
        self.assertLess(np.max(np.abs(m4[0, 2, 0, :10] - d)) / d[0], 1e-6)
        self.assertTrue(np.all(m4[0, 1, 0, 0] <= m4[0, 0, 0, 0]))

    def test_page_thorne(self):
        # flux shape against the Novikov-Thorne integral, normalised at r = 20
        from scipy.integrate import quad
        a = 0.5

        def ELO(r):
            d = np.sqrt(1 - 3 / r + 2 * a / r**1.5)
            return ((1 - 2 / r + a / r**1.5) / d,
                    np.sqrt(r) * (1 - 2 * a / r**1.5 + a * a / r**2) / d, 1 / (r**1.5 + a))
        rms, hh = 4.233002530, 1e-6
        dL = lambda r: (ELO(r + hh)[1] - ELO(r - hh)[1]) / (2 * hh)
        dO = lambda r: (ELO(r + hh)[2] - ELO(r - hh)[2]) / (2 * hh)

        def nt(r):
            E, L, O = ELO(r)
            I = quad(lambda x: (ELO(x)[0] - ELO(x)[2] * ELO(x)[1]) * dL(x), rms, r, limit=200)[0]
            return -dO(r) / (r * (E - O * L)**2) * I
        rs = np.array([5.0, 8.0, 12.0, 20.0])
        pt = an.page_thorne(a, rs)
        ref = np.array([nt(r) for r in rs])
        np.testing.assert_allclose(pt / pt[-1], ref / ref[-1], rtol=1e-6)
        self.assertEqual(an.page_thorne(a, 4.0), 0.0)

    def test_ray_outcomes(self):
        cfg = an.Config(M=1.0, emitter=an.DISK, r_in=6.0, r_out=50.0)
        # straight down from above the disc onto it
        h = an.trace(cfg, 20.0, 0.05, 0.999, 0.0)
        self.assertEqual(h["status"], an.HORIZON)
        h = an.trace(cfg, 20.0, 0.3, -0.999, 0.0)
        self.assertEqual(h["status"], an.ESCAPE)


if __name__ == "__main__":
    unittest.main()
