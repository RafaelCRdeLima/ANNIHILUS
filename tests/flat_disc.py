"""Moments of a flat annulus (M = 0) by direct integration over its area."""
import numpy as np
from scipy import integrate


def direct_moments(r, th, rin, rout, eps=1e-11, a=0.0, cos_weight=False):
    """Moments (ranks 0-2) of G = 1 over the annulus rin < rho < rout seen from (r, th).

    With a != 0, (r, th) are Boyer-Lindquist coordinates of flat space (Kerr with M = 0):
    rho = sqrt(r^2 + a^2) sin(th), z = r cos(th), and the frame is the normalised
    (d_r, d_th, d_ph); rin and rout are then Boyer-Lindquist radii in the plane.
    cos_weight: weight each direction by the cosine between the ray and the disc normal.
    """
    q = np.sqrt(r * r + a * a)
    x0 = np.array([q * np.sin(th), 0.0, r * np.cos(th)])
    er = np.array([r / q * np.sin(th), 0.0, np.cos(th)])
    er /= np.linalg.norm(er)
    et = np.array([q * np.cos(th), 0.0, -r * np.sin(th)])
    et /= np.linalg.norm(et)
    ep = np.array([0.0, 1.0, 0.0])
    rin, rout = np.sqrt(rin**2 + a * a), np.sqrt(rout**2 + a * a)

    def comp(k):
        def g(phi, rho):
            d = x0 - np.array([rho * np.cos(phi), rho * np.sin(phi), 0.0])
            D = np.linalg.norm(d)
            n = d / D
            a, b, c = n @ er, n @ et, n @ ep
            v = (1.0, a, b, c, a * a, a * b, a * c, b * b, b * c, c * c)[k]
            w = abs(n[2]) if cos_weight else 1.0
            return abs(x0[2]) / D**3 * v * rho * w
        return integrate.dblquad(g, rin, rout, 0, 2 * np.pi, epsabs=eps, epsrel=1e-10)[0]

    return np.array([comp(k) for k in range(10)])
