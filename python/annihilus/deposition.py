"""Deposition maps and volume integrals (Schwarzschild static observers or Kerr ZAMOs).

Two global rates, in geometric units (volume in M^3, rate per unit of
KRATE * D * (g_s/h^3)^2 * C3 * C4 * (k T_ref)^9):
    E_loc = int qdot dV_proper
            (Salmonson & Wilson; diverges logarithmically at a horizon),
    E_inf = int (-Q_t) sqrt(-g) d^3x,   -Q_t = alpha Q^t + omega sqrt(g_phph) Q^ph
            (energy at infinity per unit time at infinity; exact).
In Schwarzschild alpha = sqrt(A), omega = 0 and sqrt(-g) = r^2 sin(th). The 'up' versions keep
only the region where the radial momentum deposition is positive. E_birkl is the integral of
sqrt(-g) Q^t with the rate of the local observer: the definition of Birkl et al. (2007) in
Schwarzschild; in Kerr they use static observers outside the ergosphere (deposition_static).
"""
import numpy as np

from . import engine, physics


def radial_nodes(edges, n, r_h=0.0):
    """Gauss-Legendre nodes in ln(r - r_h) on each panel [edges[i], edges[i+1]]."""
    x, w = np.polynomial.legendre.leggauss(n)
    rs, ws = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        la, lb = np.log(a - r_h), np.log(b - r_h)
        s = 0.5 * (la + lb) + 0.5 * (lb - la) * x
        rr = r_h + np.exp(s)
        rs.append(rr)
        ws.append(0.5 * (lb - la) * w * (rr - r_h))
    return np.concatenate(rs), np.concatenate(ws)


def polar_nodes(n):
    """Gauss-Legendre nodes in cos(th) on (0, 1): upper hemisphere."""
    x, w = np.polynomial.legendre.leggauss(n)
    c = 0.5 * (x + 1.0)
    return np.arccos(c), 0.5 * w


def metric_factors(M, a, r, th):
    """alpha, omega, sqrt(g_phph), sqrt(-g)/(dr dth dph), sqrt(det g_ij)."""
    sth, cth = np.sin(th), np.cos(th)
    Sig = r**2 + a**2 * cth**2
    Del = r**2 - 2 * M * r + a**2
    AK = (r**2 + a**2)**2 - a**2 * Del * sth**2
    return (np.sqrt(Sig * Del / AK), 2 * M * a * r / AK, np.sqrt(AK / Sig) * sth, Sig * sth,
            np.sqrt(Sig * AK / Del) * sth)


def horizon(M, a):
    return M + np.sqrt(max(M * M - a * a, 0.0))


def deposition(cfg, edges, nr=16, nth=24, nthreads=0):
    """Map of qdot and momentum deposition on a (r, th) grid and its volume integrals.

    edges: radial panel boundaries (put r_in, r_out, R there); both hemispheres are counted
    (the emitters are symmetric about the equator).
    """
    r_h = horizon(cfg.M, cfg.spin) if cfg.M > 0 else 0.0
    r, wr = radial_nodes(np.asarray(edges, float), nr, r_h)
    th, wt = polar_nodes(nth)
    R, TH = np.meshgrid(r, th, indexing="ij")
    mom, nrays = engine.moments(cfg, R, TH, nthreads)
    m0 = mom[..., 0, :, :] if cfg.n_w == 4 else mom      # unweighted block
    q = physics.qdot_geometric(m0)
    p = physics.pdot_geometric(m0)
    al, om, sg, sqg, sq3 = metric_factors(cfg.M, cfg.spin, R, TH)
    W = 2.0 * 2.0 * np.pi * np.outer(wr, wt) / np.sin(TH)   # both hemispheres, azimuth
    loc = W * sq3 * q
    inf = W * sqg * (al * q + om * sg * p[..., 2])
    bk = W * sqg * q
    up = p[..., 0] > 0.0
    return dict(r=r, th=th, mom=mom, qgeo=q, pgeo=p, nrays=nrays,
                E_loc=loc.sum(), E_inf=inf.sum(), E_birkl=bk.sum(),
                E_loc_up=loc[up].sum(), E_inf_up=inf[up].sum(), E_birkl_up=bk[up].sum())


def deposition_static(cfg, edges, nr=16, nth=24, nthreads=0):
    """Volume integrals in the Kerr convention of Birkl et al. (2007): rates measured by static
    observers, u = e_t/sqrt(-g_tt), outside the ergosphere, r > r_E = M + sqrt(M^2 - a^2 cos^2 th).

    The static energy rate is Q^t_static = -Q_t/sqrt(-g_tt), with -Q_t = alpha Q^t
    + omega sqrt(g_phph) Q^ph in the ZAMO frame; the radial components coincide (the frames differ
    by a boost along e_phi), so the 'up' region is the same. E_loc = int Q^t_static dV (their
    Eq. 5), E_birkl = int sqrt(-g) Q^t_static d^3x (Eq. 7), E_inf as in deposition() over the same
    region. edges[0] is replaced by r_E(th); on [r_E, r_E + (r_E - r_h)] the nodes are
    Gauss-Legendre in sqrt(r - r_E), which absorbs the integrable singularity of 1/sqrt(-g_tt),
    then in ln(r - r_h) up to edges[1], edges[2], ...
    """
    M, a = cfg.M, cfg.spin
    r_h = horizon(M, a)
    th, wt = polar_nodes(nth)
    x, w = np.polynomial.legendre.leggauss(nr)
    R, WR = [], []
    for t in th:
        rE = M + np.sqrt(M * M - (a * np.cos(t))**2)
        d = rE - r_h
        s = 0.5 * np.sqrt(d) * (x + 1.0)
        r1, w1 = rE + s**2, 0.5 * np.sqrt(d) * w * 2.0 * s
        r2, w2 = radial_nodes(np.array([rE + d] + list(edges[1:])), nr, r_h)
        R.append(np.concatenate([r1, r2]))
        WR.append(np.concatenate([w1, w2]))
    R, WR = np.array(R).T, np.array(WR).T            # shape (radial, polar)
    TH = np.broadcast_to(th, R.shape)
    mom, nrays = engine.moments(cfg, R, TH, nthreads)
    m0 = mom[..., 0, :, :] if cfg.n_w == 4 else mom
    q = physics.qdot_geometric(m0)
    p = physics.pdot_geometric(m0)
    al, om, sg, sqg, sq3 = metric_factors(M, a, R, TH)
    mQt = al * q + om * sg * p[..., 2]
    mgtt = 1.0 - 2.0 * M * R / (R**2 + (a * np.cos(TH))**2)       # -g_tt
    qs = mQt / np.sqrt(mgtt)
    W = 2.0 * 2.0 * np.pi * WR * wt / np.sin(TH)
    loc, bk, inf = W * sq3 * qs, W * sqg * qs, W * sqg * mQt
    up = p[..., 0] > 0.0
    return dict(r=R, th=th, nrays=nrays, E_loc=loc.sum(), E_birkl=bk.sum(), E_inf=inf.sum(),
                E_loc_up=loc[up].sum(), E_birkl_up=bk[up].sum(), E_inf_up=inf[up].sum())


def rate_unit(kT, M_cm, D, g_s=1.0):
    """erg/s per geometric unit of E_loc, E_inf (kT in MeV, M in cm)."""
    return (physics.KRATE * D * (g_s / physics.H_PLANCK**3)**2 * physics.C3 * physics.C4
            * kT**9 * M_cm**3 * physics.MEV_ERG)
