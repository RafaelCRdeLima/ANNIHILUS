"""Flavour conversion with species-dependent neutrinospheres: relativistic versus flat space.

Reads output/flavour_<case>[_<model>][_sT<s>|_pt]_a<spin>.npz (from flavour_runs.py).
Conversion at emission, with probability P(r_em, mu_em) = P0 * p(r_em, mu_em):
    f'_e = f_e - P f_e + P f_x,   f'_x = f_x + (P/2) f_e - (P/2) f_x   (per heavy flavour),
and the same for antineutrinos with Pbar0. Since the engine stores moments weighted by
1, rho, mu_em and rho mu_em (rho = exp(-(r_em - r_isco)/r_isco)), any p that is a combination of
these is exact. For each configuration, the conversion factor R = Edot_inf(P0)/Edot_inf(0) is
computed in the relativistic case (gr) and in flat space with static (newt) discs and discs
rotating at Newtonian Keplerian speeds (flatrot) or at the local speeds of the relativistic disc
(flatloc). Pbar = P, or Pbar fixed by conservation of the net electron-lepton number of the map,
P (N_e - N_x) = Pbar (N_ebar - N_x), over the range of P with Pbar <= 1, where delta is the exact
maximum over that range (maxdev_exact). A sweep over temperature and luminosity ratios, which only
rescales the moments, crossed with the conversion profiles, tests the robustness of R_gr/R_flat.
Writes output/flavour.json and output/flavour_sweep.json.
"""
import glob
import itertools
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
from annihilus import deposition as dp, physics as ph  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "output")
KT = {"e": 4.0, "ebar": 5.0, "x": 6.0}     # MeV at r_isco (power law) or at the flux peak (pt)
SPECIES = ("e", "ebar", "x")
# conversion profiles p(r_em, mu_em) as coefficients of the weight blocks (1, rho, mu, rho mu)
PROFILES = {"uniform": (1, 0, 0, 0), "inner": (0, 1, 0, 0), "outer": (1, -1, 0, 0),
            "normal": (0, 0, 1, 0), "grazing": (1, 0, -1, 0), "inner-grazing": (0, 1, 0, -1)}
PS = np.linspace(0, 2 / 3, 9)
FLAT = ("flatrot", "flatloc")     # flat-space references: Newtonian Keplerian, local speeds
REF = os.environ.get("ANNIHILUS_REF", "flatloc")   # the reference of delta in the paper
SWEEP = list(itertools.product((0.8, 1.0, 1.25, 1.5), (1.0, 1.25, 1.5), (0.1, 0.3, 1.0)))


def fname(case, model, spin, tag):
    m = "" if model == "default" else "_" + model
    return os.path.join(OUT, f"flavour_{case}{m}{tag}_a{spin:.2f}.npz")


def kT_ref(z, sp, kT):
    """Temperature at the engine's reference point for the species."""
    if "profile" in z and str(z["profile"]) == "pt":
        return kT[sp]
    ri, s = float(z["r_in"]), float(z["s_T"])
    rin_sp = z["r_in_sp"][SPECIES.index(sp)] if "r_in_sp" in z else ri
    return kT[sp] * (rin_sp / ri)**(-s)


class Case:
    def __init__(self, path):
        z = np.load(path)
        self.z = z
        M = float(z["M"])
        a = float(z["spin"]) if "spin" in z else 0.0
        rh = dp.horizon(M, a) if M > 0 else 0.0
        r, wr = dp.radial_nodes(z["edges"], int(z["nr"]), rh)
        th, wt = dp.polar_nodes(int(z["nth"]))
        R, TH = np.meshgrid(r, th, indexing="ij")
        al, om, sg, sqg, _ = dp.metric_factors(M, a, R, TH)
        self.W = 4 * np.pi * np.outer(wr, wt) / np.sin(TH) * sqg
        self.al, self.om_sg = al, om * sg
        self.mom = {sp: z["mom_" + sp] for sp in SPECIES}
        self.weighted = self.mom["e"].ndim == 5

    def moments(self, kT, grey, common=False):
        """Physical moments per species, shape (..., n_w, 2, 20). common: every species on
        the nu_e neutrinosphere, with its own temperature scale."""
        E = {}
        for sp in SPECIES:
            src = "e" if common else sp
            m = self.mom[src] if self.weighted else self.mom[src][..., None, :, :]
            T = kT_ref(self.z, "e", kT) * kT[sp] / kT["e"] if common else kT_ref(self.z, sp, kT)
            E[sp] = grey[sp] * ph.energy_moments(m, T)
        return E

    def rate(self, E, P0, Pb0, prof="uniform"):
        c = np.array(PROFILES[prof], float)
        if not self.weighted and np.any(c[1:]):
            return np.nan

        def conv(Ee, Ex, P):
            Pe = P * np.tensordot(Ee, c[:Ee.shape[-3]], axes=([-3], [0]))
            Px = P * np.tensordot(Ex, c[:Ex.shape[-3]], axes=([-3], [0]))
            return Ee[..., 0, :, :] - Pe + Px, Ex[..., 0, :, :] + 0.5 * (Pe - Px)
        e, x = conv(E["e"], E["x"], P0)
        eb, xb = conv(E["ebar"], E["x"], Pb0)
        De, Dx = ph.couplings(0.23)
        q = ph.qdot(e, eb, De) + 2 * ph.qdot(x, xb, Dx)
        p = ph.pdot(e, eb, De) + 2 * ph.pdot(x, xb, Dx)
        return np.sum(self.W * (self.al * q + self.om_sg * p[..., 2]))


def numbers(z, kT, grey=None):
    """Neutrinos of each species (one heavy flavour for x) emitted per unit Killing time, up to a
    common factor. A Fermi-Dirac surface at comoving temperature T emits a number flux proportional
    to T^3 per unit comoving area and proper time, and the world sheet of an equatorial disc has
    the invariant element r dt dr dphi in Kerr (sqrt(g_rr (g_tphi^2 - g_tt g_phph)) = r at the
    equator) as in flat space, whatever the rotation of the disc: N = int T^3 r dr. Temperatures
    as in the engine (Page-Thorne: normalised to the peak of the whole disc, as in flavour_runs.py).
    """
    ri, s = float(z["r_in"]), float(z["s_T"])
    rin = z["r_in_sp"] if "r_in_sp" in z else np.full(3, ri)
    rout = z["r_out"]
    pt = "profile" in z and str(z["profile"]) == "pt"
    if pt:
        from annihilus import page_thorne
        a = float(z["spin"]) if float(z["M"]) > 0 else 0.0
        t_norm = np.max(page_thorne(a, np.linspace(ri, 60.0, 20000)))
    N = {}
    for k, sp in enumerate(SPECIES):
        rr = np.linspace(rin[k], rout[k], 4001)
        if pt:
            T = kT[sp] * (np.clip(page_thorne(a, rr), 0.0, None) / t_norm)**0.25
        else:
            T = kT[sp] * (rr / ri)**(-s)
        N[sp] = (1.0 if grey is None else grey[sp]) * np.trapezoid(T**3 * rr, rr)
    return N


def lepton_kbar(N, tol=1e-12):
    """Pbar/P that conserves N_e - N_ebar under the map, with N_xbar = N_x (same disc and spectrum):
    Delta(N_e - N_ebar) = -P (N_e - N_x) + Pbar (N_ebar - N_x) = 0.

    k = 0 (N_e = N_x: only antineutrinos are left unconverted) is admissible. If both differences
    vanish, any Pbar conserves it and Pbar = P (k = 1) is returned. None when no map with P > 0 and
    Pbar >= 0 conserves it: k < 0, or N_ebar = N_x != N_e (then only P = 0 does)."""
    num, den = N["e"] - N["x"], N["ebar"] - N["x"]
    small = tol * max(abs(v) for v in N.values())
    if abs(den) <= small:
        return 1.0 if abs(num) <= small else None
    if abs(num) <= small:
        return 0.0
    k = num / den
    return k if k > 0 else None


def configurations():
    """(model, tag, spin) for which gr and flatrot files exist."""
    out = []
    for path in sorted(glob.glob(os.path.join(OUT, "flavour_gr*_a*.npz"))):
        m = re.match(r"flavour_gr(_xinner|_disjoint)?(_pt|_sT[0-9.]+)?_a([0-9.]+)\.npz$",
                     os.path.basename(path))
        if not m:
            continue
        model = (m.group(1) or "_default")[1:]
        tag, spin = m.group(2) or "", float(m.group(3))
        if os.path.exists(fname("flatrot", model, spin, tag)):
            out.append((model, tag, spin))
    return out


def factors(cases, kT, grey, prof, kbar=1.0, common=False, E=None):
    """Conversion factors for Pbar = kbar P on the grid PS of P or, if Pbar would exceed 1 there,
    on as many equally spaced values from 0 to P_max = 1/kbar (no physical map beyond).
    E: moments of each case, if already computed."""
    rows = []
    Ps = PS if kbar * PS[-1] <= 1.0 else np.linspace(0.0, 1.0 / kbar, len(PS))
    if E is None:
        E = {k: c.moments(kT, grey) for k, c in cases.items()}
    base = {k: c.rate(E[k], 0.0, 0.0) for k, c in cases.items()}
    if common:
        Ec = cases["gr"].moments(kT, grey, common=True)
        base_c = cases["gr"].rate(Ec, 0.0, 0.0)
    for P in Ps:
        Pb = kbar * P
        w = dict(P=P, Pbar=Pb)
        rate = {k: c.rate(E[k], P, Pb, prof) for k, c in cases.items()}
        for k in cases:
            w["R_" + k] = rate[k] / base[k]
        if common:
            w["R_common"] = cases["gr"].rate(Ec, P, Pb, prof) / base_c
        for k, g in (("flatrot", "gain_rot"), ("flatloc", "gain_loc"), ("newt", "gain_static")):
            if k in cases:
                w[g] = rate["gr"] / rate[k]
        rows.append(w)
    return rows


def maxdev(rows, ref):
    return max(abs(w["R_gr"] / w["R_" + ref] - 1) for w in rows)


def maxdev_exact(rows, ref):
    """Maximum of |R_gr/R_ref - 1| over the continuous range of P of the rows. Along Pbar = kbar P
    both factors are quadratic in P, so the rows fix them; the extrema of their ratio lie at the
    ends or where R_gr' R_ref - R_gr R_ref' = 0, a quadratic equation (the cubic terms cancel)."""
    P = np.array([w["P"] for w in rows])
    g = np.polyfit(P, [w["R_gr"] for w in rows], 2)
    f = np.polyfit(P, [w["R_" + ref] for w in rows], 2)
    c = [g[0] * f[1] - g[1] * f[0], 2 * (g[0] * f[2] - g[2] * f[0]), g[1] * f[2] - g[2] * f[1]]
    ends = [P[0], P[-1]]
    stat = [x.real for x in np.roots(c) if abs(x.imag) < 1e-12 and ends[0] < x.real < ends[1]]
    return max(abs(np.polyval(g, x) / np.polyval(f, x) - 1) for x in ends + stat)


def main():
    grey1 = {sp: 1.0 for sp in SPECIES}
    res, sweep = {}, {}
    for model, tag, spin in configurations():
        cases = {c: Case(fname(c, model, spin, tag)) for c in ("gr",) + FLAT + ("newt",)
                 if os.path.exists(fname(c, model, spin, tag))}
        flats = [c for c in FLAT if c in cases]
        key = f"{model}{tag}|{spin:.2f}"
        N = numbers(cases["gr"].z, KT)
        kbar = lepton_kbar(N)
        out = dict(model=model, tag=tag, spin=spin, N=N, kbar_lepton=kbar,
                   P_max_lepton=None if kbar is None else min(2 / 3, 1 / kbar) if kbar else 2 / 3,
                   weighted=cases["gr"].weighted)
        out["equal"] = factors(cases, KT, grey1, "uniform", common=True)
        out["lepton"] = None if kbar is None else factors(cases, KT, grey1, "uniform", kbar)
        profs = list(PROFILES) if cases["gr"].weighted else ["uniform"]
        if cases["gr"].weighted:
            out["profiles"] = {p: factors(cases, KT, grey1, p) for p in PROFILES}
        res[key] = out
        # sweep of temperature and luminosity ratios (P = Pbar) crossed with the profiles
        pair = {c: cases[c] for c in ["gr"] + flats}
        devs = []
        for tx, teb, gx in SWEEP:
            kT = {"e": 4.0, "ebar": 4.0 * teb, "x": 4.0 * tx}
            grey = {"e": 1.0, "ebar": 1.0, "x": gx}
            E = {k: c.moments(kT, grey) for k, c in pair.items()}
            rows = {p: factors(pair, kT, grey, p, E=E) for p in profs}
            devs.append(dict(Tx=tx, Teb=teb, grey_x=gx,
                             **{"dev" if f == "flatrot" else "dev_" + f[4:]:
                                maxdev(rows["uniform"], f) for f in flats},
                             prof={f: {p: maxdev(r, f) for p, r in rows.items()} for f in flats}))
        sweep[key] = devs
        gains = "  ".join(f"gain_{f[4:]} {out['equal'][0]['gain_' + f[4:]]:.3f}" for f in flats)
        if "newt" in cases:
            gains += f"  gain_static {out['equal'][0]['gain_static']:.3f}"
        lep = "--" if kbar is None else f"{kbar:.3f}"
        print(f"{key:28s} {gains}  Pbar/P (lepton) {lep}", flush=True)
        for f in flats:
            dl = "  --  " if kbar is None else f"{100 * maxdev_exact(out['lepton'], f):5.2f}%"
            line = (f"{'':28s} vs {f}: delta {100 * maxdev(out['equal'], f):5.2f}% "
                    f"(lepton {dl})")
            if "newt" in cases and f == flats[0]:
                line += f"  delta_total {100 * maxdev(out['equal'], 'newt'):5.2f}%"
            if "profiles" in out:
                line += "  profiles: " + " ".join(f"{p} {100 * maxdev(v, f):.2f}%"
                                                  for p, v in out["profiles"].items())
            d = np.array([v["dev" if f == "flatrot" else "dev_" + f[4:]] for v in devs]) * 100
            j = 100 * max(max(v["prof"][f].values()) for v in devs)
            line += (f"\n{'':28s}    sweep (36 cases): median {np.median(d):.2f}%  "
                     f"max {d.max():.2f}%;  x {len(profs)} profiles: max {j:.2f}%")
            print(line, flush=True)
    with open(os.path.join(OUT, "flavour.json"), "w") as f:
        json.dump(res, f, indent=1)
    with open(os.path.join(OUT, "flavour_sweep.json"), "w") as f:
        json.dump(sweep, f, indent=1)


if __name__ == "__main__":
    main()
