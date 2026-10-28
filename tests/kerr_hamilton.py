"""Independent Kerr null geodesics: Hamilton's equations in Boyer-Lindquist coordinates with the
inverse metric written out and differentiated numerically (no Carter constant, no Mino time)."""
import numpy as np
from scipy.integrate import solve_ivp


def ginv(M, a, r, th):
    s2, c2 = np.sin(th)**2, np.cos(th)**2
    Sig = r * r + a * a * c2
    Del = r * r - 2 * M * r + a * a
    AK = (r * r + a * a)**2 - a * a * Del * s2
    G = np.zeros((4, 4))
    G[0, 0] = -AK / (Sig * Del)
    G[0, 3] = G[3, 0] = -2 * M * a * r / (Sig * Del)
    G[3, 3] = (Del - a * a * s2) / (Sig * Del * s2)
    G[1, 1] = Del / Sig
    G[2, 2] = 1 / Sig
    return G


def gmet(M, a, r, th):
    return np.linalg.inv(ginv(M, a, r, th))


def rhs(M, a):
    def f(lam, y):
        x, p = y[:4], y[4:]
        G = ginv(M, a, x[1], x[2])
        dx = G @ p
        dp = np.zeros(4)
        for k, h in ((1, 1e-6 * x[1]), (2, 1e-6)):
            xp, xm = x.copy(), x.copy()
            xp[k] += h
            xm[k] -= h
            dG = (ginv(M, a, xp[1], xp[2]) - ginv(M, a, xm[1], xm[2])) / (2 * h)
            dp[k] = -0.5 * p @ dG @ p
        return np.concatenate([dx, dp])
    return f


def first_crossing(M, a, r, th, mu, ph):
    """Trace backward the neutrino arriving at (r, th) along n = (mu, s cos ph, s sin ph) in the
    ZAMO frame; return the radius of the first crossing of the equatorial plane (or None)."""
    s = np.sqrt(1 - mu * mu)
    n = np.array([mu, s * np.cos(ph), s * np.sin(ph)])
    s2 = np.sin(th)**2
    Sig = r * r + a * a * np.cos(th)**2
    Del = r * r - 2 * M * r + a * a
    AK = (r * r + a * a)**2 - a * a * Del * s2
    al, om, sg = np.sqrt(Sig * Del / AK), 2 * M * a * r / AK, np.sqrt(AK / Sig * s2)
    et = np.array([1 / al, 0, 0, om / al])
    er = np.array([0, np.sqrt(Del / Sig), 0, 0])
    eth = np.array([0, 0, 1 / np.sqrt(Sig), 0])
    eph = np.array([0, 0, 0, 1 / sg])
    pu = et + n[0] * er + n[1] * eth + n[2] * eph       # forward momentum, contravariant
    pl = gmet(M, a, r, th) @ pu
    y0 = np.concatenate([[0, r, th, 0], -pl])          # reversed: trace backward
    rp = M + np.sqrt(M * M - a * a)
    cross = lambda lam, y: y[2] - np.pi / 2
    horizon = lambda lam, y: y[1] - 1.001 * rp
    escape = lambda lam, y: y[1] - 2000.0
    for e in (horizon, escape):
        e.terminal = True
    cross.terminal = True
    sol = solve_ivp(rhs(M, a), (0, 1e5), y0, events=(cross, horizon, escape), rtol=1e-11,
                    atol=1e-12, method="DOP853")
    if len(sol.t_events[0]):
        return sol.y_events[0][0][1]
    return None
