"""Constants, rates and analytic results.

Units: MeV, cm, s. Energy moments of a species are
    E_n = (g_s/h^3) Gamma(n+1) eta(n+1) (k T_ref)^(n+1) * (engine moment of G_n),
and the deposition rate per unit proper volume and time is
    qdot = G_F^2/(3 pi c^5) * D * [ K(E4_nu, E3_nubar) + K(E3_nu, E4_nubar) ],
    K(a, b) = a^0 b^0 - 2 a_i b_i + a_ij b_ij.
"""
from itertools import permutations

import numpy as np
from scipy.special import zeta

C_LIGHT = 2.99792458e10          # cm/s
H_PLANCK = 4.135667696e-21       # MeV s
GF2 = 5.297e-44                  # G_F^2 (hbar c)^2 / (hbar c)^4 ... in cm^2 MeV^-2
MEV_ERG = 1.602176634e-6
GM_SUN = 1.476625e5              # G M_sun / c^2 in cm
S2W = 0.2312

C3 = 7.0 * np.pi**4 / 120.0      # Gamma(4) eta(4)
C4 = 45.0 * zeta(5) / 2.0        # Gamma(5) eta(5)
KRATE = GF2 / (3.0 * np.pi * C_LIGHT**5)


def couplings(s2w=S2W):
    """D for nu_e nubar_e and for each heavy-lepton flavour."""
    return 1 + 4 * s2w + 8 * s2w**2, 1 - 4 * s2w + 8 * s2w**2


# ---------------------------------------------------------------- moments

_I2 = [(i, j) for i in range(3) for j in range(i, 3)]
_I3 = [(i, j, k) for i in range(3) for j in range(i, 3) for k in range(j, 3)]


def tensors(m):
    """Moment block (..., 20) -> M (...), M_i (..., 3), M_ij (..., 3, 3), M_ijk (..., 3, 3, 3)."""
    m = np.asarray(m)
    sh = m.shape[:-1]
    m2 = np.empty(sh + (3, 3))
    for c, (i, j) in enumerate(_I2):
        m2[..., i, j] = m2[..., j, i] = m[..., 4 + c]
    m3 = np.empty(sh + (3, 3, 3))
    for c, idx in enumerate(_I3):
        for p in set(permutations(idx)):
            m3[(Ellipsis,) + p] = m[..., 10 + c]
    return m[..., 0], m[..., 1:4], m2, m3


def kernel(a, b):
    """K(a, b) = int int (1 - n.n')^2 a(n) b(n') dOmega dOmega'."""
    a0, a1, a2, _ = tensors(a)
    b0, b1, b2, _ = tensors(b)
    return a0 * b0 - 2.0 * np.einsum("...i,...i", a1, b1) + np.einsum("...ij,...ij", a2, b2)


def pkernel(a, b):
    """P^i(a, b) = int int (1 - n.n')^2 n^i a(n) b(n') dOmega dOmega' (frame r, th, ph)."""
    a0, a1, a2, a3 = tensors(a)
    b0, b1, b2, _ = tensors(b)
    return (a1 * b0[..., None] - 2.0 * np.einsum("...ij,...j", a2, b1)
            + np.einsum("...ijk,...jk", a3, b2))


def qdot_geometric(mom):
    """K(G4, G3) + K(G3, G4) for one species pair sharing emitter and temperature."""
    return kernel(mom[..., 1, :], mom[..., 0, :]) + kernel(mom[..., 0, :], mom[..., 1, :])


def pdot_geometric(mom):
    """P(G4, G3) for nu plus the same for nubar, when both share emitter and temperature."""
    return 2.0 * pkernel(mom[..., 1, :], mom[..., 0, :])


def energy_moments(mom, kT, g_s=1.0):
    """Physical moments E_3, E_4 (shape (..., 2, 10)) for reference temperature kT [MeV]."""
    E = np.array(mom, dtype=float, copy=True)
    E[..., 0, :] *= g_s / H_PLANCK**3 * C3 * kT**4
    E[..., 1, :] *= g_s / H_PLANCK**3 * C4 * kT**5
    return E


def qdot(E_nu, E_nubar, D):
    """Deposition rate [MeV cm^-3 s^-1] from physical moments of nu and nubar."""
    return KRATE * D * (kernel(E_nu[..., 1, :], E_nubar[..., 0, :])
                        + kernel(E_nu[..., 0, :], E_nubar[..., 1, :]))


def pdot(E_nu, E_nubar, D):
    """Momentum deposition rate times c [MeV cm^-3 s^-1], components (r, th, ph)."""
    return KRATE * D * (pkernel(E_nu[..., 1, :], E_nubar[..., 0, :])
                        + pkernel(E_nubar[..., 1, :], E_nu[..., 0, :]))


def mix(E_e, E_x, P):
    """Flavour conversion of the distributions (acts linearly on the moments)."""
    return (1 - P) * E_e + P * E_x, 0.5 * P * E_e + (1 - 0.5 * P) * E_x


def qdot_flavours(E, P=0.0, Pbar=0.0, s2w=S2W):
    """Total rate for species E['e'], E['ebar'], E['x'], E['xbar'] (x: one heavy flavour)."""
    De, Dx = couplings(s2w)
    e, x = mix(E["e"], E["x"], P)
    eb, xb = mix(E["ebar"], E["xbar"], Pbar)
    return qdot(e, eb, De) + 2.0 * qdot(x, xb, Dx)


def pdot_flavours(E, P=0.0, Pbar=0.0, s2w=S2W):
    """Total momentum deposition (times c) with flavour conversion, components (r, th, ph)."""
    De, Dx = couplings(s2w)
    e, x = mix(E["e"], E["x"], P)
    eb, xb = mix(E["ebar"], E["xbar"], Pbar)
    return pdot(e, eb, De) + 2.0 * pdot(x, xb, Dx)


# ---------------------------------------------------------------- luminosity

def kT_from_luminosity(L_erg_s, R_cm, g_s=1.0):
    """Surface temperature [MeV] of a Fermi-Dirac blackbody of radius R and luminosity L."""
    flux = L_erg_s / MEV_ERG / (4.0 * np.pi * R_cm**2)
    return (flux * C_LIGHT**2 * H_PLANCK**3 / (np.pi * g_s * C3))**0.25


# ---------------------------------------------------------------- sphere (Salmonson & Wilson)

def x_sphere(r, R, M=0.0):
    """Cosine of the half-angle of the neutrino cone at radius r."""
    A = lambda q: 1.0 - 2.0 * M / q
    return np.sqrt(np.clip(1.0 - (R / r)**2 * A(r) / A(R), 0.0, 1.0))


def cap_moments(x):
    """Moment block (20) of G = 1 on the cone mu > x around e_r."""
    m = np.zeros(np.shape(x) + (20,))
    m[..., 0] = 2 * np.pi * (1 - x)
    m[..., 1] = np.pi * (1 - x**2)
    m[..., 4] = 2 * np.pi / 3 * (1 - x**3)                       # rr
    m[..., 7] = m[..., 9] = np.pi * ((1 - x) - (1 - x**3) / 3)   # thth, phph
    m[..., 10] = np.pi / 2 * (1 - x**4)                          # rrr
    m[..., 13] = m[..., 15] = np.pi * ((1 - x**2) / 2 - (1 - x**4) / 4)  # rthth, rphph
    return m


def theta_sphere(x):
    """Theta = int int (1 - n.n')^2 over the cone mu > x."""
    return 2.0 * np.pi**2 / 3.0 * (1 - x)**4 * (x**2 + 4 * x + 5)


def _t_nodes(n):
    t, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (t + 1.0), 0.5 * w


def sphere_radii(R, n=48):
    """Radii and weights for int_R^inf f(r) r^2 dr = sum w f(r), via r = R/(1 - t^2)."""
    t, w = _t_nodes(n)
    y = 1.0 - t**2
    return R / y, w * R**3 * 2.0 * t / y**4


def sphere_enhancement(R_over_M, n=48, qgeo=None):
    """F(M/R) = Qdot_GR / Qdot_Newt at fixed luminosity at infinity, proper volume (S&W 1999).

    qgeo: optional function r -> K(G4,G3)+K(G3,G4) from the engine (M = 1, T_ref = T(R));
    otherwise the analytic cone is used.
    """
    R, M = float(R_over_M), 1.0
    r, w = sphere_radii(R, n)
    AR, Ar = 1 - 2 * M / R, 1 - 2 * M / r
    if qgeo is None:
        q = AR**4.5 * Ar**-4.5 * theta_sphere(x_sphere(r, R, M))
    else:
        q = 0.5 * qgeo(r)
    gr = AR**-2.25 * np.sum(w * q / np.sqrt(Ar))
    rN, wN = sphere_radii(1.0, n)
    newt = np.sum(wN * theta_sphere(x_sphere(rN, 1.0))) * R**3
    return gr / newt


def qdot_thermal(kT, Theta, D, g_s=1.0):
    """Uniform thermal nu, nubar at kT over a common angular factor Theta [MeV cm^-3 s^-1]."""
    return KRATE * D * (g_s / H_PLANCK**3)**2 * 2.0 * C3 * C4 * kT**9 * Theta


def Qdot_newtonian_sphere(L_erg_s, R_cm, D, g_s=1.0, n=48, g_lum=None):
    """Newtonian Qdot [erg/s] for one flavour pair, each species with luminosity L.

    g_lum: statistical weight used to convert L into T (default g_s). Salmonson & Wilson
    use f = 2/h^3 in the rate (g_s = 2) and L = 4 pi R^2 (7/4)(ac/4) T^4, i.e. g_lum = 4.
    """
    kT = kT_from_luminosity(L_erg_s, R_cm, g_s if g_lum is None else g_lum)
    r, w = sphere_radii(1.0, n)
    I = np.sum(w * theta_sphere(x_sphere(r, 1.0)))
    return qdot_thermal(kT, 1.0, D, g_s) * 4.0 * np.pi * R_cm**3 * I * MEV_ERG
