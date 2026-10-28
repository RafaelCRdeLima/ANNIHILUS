import unittest

import numpy as np

import annihilus as an
from annihilus import physics as ph


def qgeo_engine(R, M=1.0, n_phi=4):
    cfg = an.Config(M=M, emitter=an.SPHERE, R=R, n_phi=n_phi)
    def q(r):
        mom, _ = an.moments(cfg, r, np.ones_like(r))
        return ph.qdot_geometric(mom)
    return q


class TestSphere(unittest.TestCase):
    def test_theta_newtonian(self):
        q = qgeo_engine(1.0, M=0.0)
        r = np.array([1.0001, 1.1, 2.0, 5.0, 30.0])
        ana = ph.theta_sphere(ph.x_sphere(r, 1.0))
        np.testing.assert_allclose(q(r) / 2, ana, rtol=1e-9)

    def test_theta_schwarzschild(self):
        R = 10.0
        q = qgeo_engine(R)
        r = np.array([10.0001, 10.5, 15.0, 30.0, 100.0])
        g = np.sqrt((1 - 2 / R) / (1 - 2 / r))
        ana = ph.theta_sphere(ph.x_sphere(r, R, 1.0))
        np.testing.assert_allclose(q(r) / (2 * g**9), ana, rtol=1e-9)

    def test_cap_moments(self):
        # full block, including the rank-3 moments, against the analytic cone
        R, M = 10.0, 1.0
        cfg = an.Config(M=M, emitter=an.SPHERE, R=R, n_phi=8)
        r = np.array([10.001, 12.0, 40.0])
        mom, _ = an.moments(cfg, r, np.full(3, 0.4))
        g = np.sqrt((1 - 2 * M / R) / (1 - 2 * M / r))
        cap = ph.cap_moments(ph.x_sphere(r, R, M))
        np.testing.assert_allclose(mom[:, 0], cap * g[:, None]**4, rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(mom[:, 1], cap * g[:, None]**5, rtol=1e-9, atol=1e-12)

    def test_momentum_sphere(self):
        # S&W Phi_p = (1/2) int int (1 - n.n')^2 r.(n + n') = P^r(cap, cap)
        x = np.array([0.0, 0.5, 0.999])
        cap = ph.cap_moments(x)
        Pr = ph.pkernel(cap, cap)[..., 0]
        Th = ph.kernel(cap, cap)
        # the kernel cancels terms of order (1 - x)^2 to leave (1 - x)^4
        np.testing.assert_allclose(Th, ph.theta_sphere(x), rtol=1e-7)
        self.assertAlmostEqual(Pr[0] / Th[0], 0.4, places=12)   # 2/5 at the surface
        self.assertLess(1.0 - Pr[2] / Th[2], 1e-3)               # radial streaming far away
        np.testing.assert_allclose(ph.pkernel(cap, cap)[..., 1:], 0.0, atol=1e-14)

    def test_theta_far(self):
        x = ph.x_sphere(1e3, 1.0)
        self.assertAlmostEqual(ph.theta_sphere(x) / (5 * np.pi**2 / 12 * 1e-24), 1.0, places=4)

    def test_enhancement_engine_vs_analytic(self):
        for RM in (10.0, 5.0, 3.0):
            Fa = ph.sphere_enhancement(RM)
            Fe = ph.sphere_enhancement(RM, qgeo=qgeo_engine(RM))
            self.assertAlmostEqual(Fe / Fa, 1.0, places=9)

    def test_enhancement_values(self):
        # values given in Section 8.1 of the paper for R/M = 10, 5 and 3
        F = [ph.sphere_enhancement(RM) for RM in (10.0, 5.0, 3.0)]
        np.testing.assert_allclose(F, [1.87, 4.27, 28.8], rtol=5e-3)

    def test_newtonian_normalisation(self):
        # kappa D L51^(9/4) R6^(-3/2) 1e51 erg/s, Eq. (4) of the paper: kappa = 1.10e-5 with
        # the conventions of Salmonson & Wilson (g_s = 2, L = 4 pi R^2 (7/4) sigma T^4) and
        # 6.20e-5 for g_s = 1
        Q = ph.Qdot_newtonian_sphere(1e51, 1e6, 1.0, g_s=2.0, g_lum=4.0)
        self.assertLess(abs(Q / 1.10e46 - 1.0), 5e-3)
        Q1 = ph.Qdot_newtonian_sphere(1e51, 1e6, 1.0, g_s=1.0)
        self.assertLess(abs(Q1 / 6.20e46 - 1.0), 5e-3)
        Q2 = ph.Qdot_newtonian_sphere(2e51, 3e6, 1.0, g_s=2.0, g_lum=4.0)
        self.assertAlmostEqual(Q2 / Q, 2**2.25 * 3**-1.5, places=10)
        # at fixed luminosity Qdot scales as g_s^(-1/4)
        Qb =ph.Qdot_newtonian_sphere(1e51, 1e6, 1.0, g_s=2.0)
        self.assertAlmostEqual(Qb / Q1, 2**-0.25, places=10)

    def test_thermal_formula(self):
        # uniform thermal rate with g_s = 2 equals 7 D G_F^2 pi^3 zeta(5) (kT)^9 Theta / (2 c^5 h^6)
        from scipy.special import zeta
        kT = 5.0
        q = ph.qdot_thermal(kT, 1.0, 1.0, g_s=2.0)
        ref = 7 * ph.GF2 * np.pi**3 * zeta(5) * kT**9 / (2 * ph.C_LIGHT**5 * ph.H_PLANCK**6)
        self.assertAlmostEqual(q / ref, 1.0, places=12)


if __name__ == "__main__":
    unittest.main()
