"""Paper figures: sphere enhancement, disc deposition maps and flavour ratios.

Writes PDFs to figures/ (or to $ANNIHILUS_FIG) and caches maps in output/.
Usage: figures.py [sphere] [maps] [layouts] [emission] [flavour] [robust] [reversal]
"""
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "python"))
import annihilus as an  # noqa: E402
from annihilus import physics as ph, plots, published  # noqa: E402
sys.path.insert(0, HERE)
import flavour as F  # noqa: E402

OUT = os.path.join(HERE, "..", "output")
FIG = os.environ.get("ANNIHILUS_FIG") or os.path.join(HERE, "..", "figures")
KT = {"e": 4.0, "ebar": 5.0}
REF = F.REF                      # flat-space reference of delta: flatloc or flatrot
FLATLAB = {"flatloc": "flat space, same local speeds", "flatrot": "flat space, Newtonian Keplerian"}


def fig_sphere():
    RM = np.geomspace(3.0, 30.0, 200)
    F = [ph.sphere_enhancement(x) for x in RM]
    pts = np.array([3.0, 3.5, 4.0, 5.0, 7.0, 10.0, 20.0])
    Fe = []
    for x in pts:
        cfg = an.Config(M=1.0, emitter=an.SPHERE, R=x, n_phi=4)
        Fe.append(ph.sphere_enhancement(
            x, qgeo=lambda r: ph.qdot_geometric(an.moments(cfg, r, np.ones_like(r))[0])))
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ax.plot(RM, F, color=plots.SERIES[0], label="analytic cone")
    ax.plot(pts, Fe, "o", color=plots.SERIES[0], mfc="white", mew=1.2, ms=5,
            label="ray tracing (this work)")
    sw = published.get("salmonson_wilson_1999")
    if sw:
        ax.plot(sw["R_over_M"], sw["F"], "s", color=plots.SERIES[1], ms=4,
                label="Salmonson & Wilson (1999)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$R/M$"); ax.set_ylabel(r"$F=\dot Q_{\rm GR}/\dot Q_{\rm Newt}$")
    ax.set_xticks([3, 5, 10, 20, 30]); ax.set_xticklabels(["3", "5", "10", "20", "30"])
    ax.set_yticks([1, 2, 5, 10, 20, 30]); ax.set_yticklabels(["1", "2", "5", "10", "20", "30"])
    ax.minorticks_off()
    ax.legend(loc="upper right")
    fig.savefig(os.path.join(FIG, "sphere_enhancement.pdf"))
    plt.close(fig)


def disc_map(M, n=56, L=40.0, mrot=0.0, law=0):
    name = "gr" if M > 0 else (("flatloc" if law else "flatrot") if mrot > 0 else "newt")
    path = os.path.join(OUT, f"map_{name}.npz")
    if os.path.exists(path):
        return dict(np.load(path))
    x = np.linspace(0.25, L, n)
    z = np.linspace(0.25, L, n)
    X, Z = np.meshgrid(x, z, indexing="ij")
    r, th = np.hypot(X, Z), np.arctan2(X, Z)
    mask = r > 2.0 * M * 1.02 + 1e-9
    q = np.full(X.shape, np.nan)
    pr = np.full(X.shape, np.nan)
    E = {}
    for sp, rout in (("e", 40.0), ("ebar", 25.0)):
        cfg = an.Config(M=M, emitter=an.DISK, r_in=6.0, r_out=rout,
                        keplerian=int(M > 0 or mrot > 0), t_index=0.5, m_rot=mrot, rot_law=law)
        mom, _ = an.moments(cfg, r[mask], th[mask])
        E[sp] = ph.energy_moments(mom, KT[sp])
    De, _ = ph.couplings(0.23)
    q[mask] = ph.qdot(E["e"], E["ebar"], De)
    pr[mask] = ph.pdot(E["e"], E["ebar"], De)[..., 0]
    out = dict(X=X, Z=Z, q=q, pr=pr)
    np.savez(path, **out)
    return out


def enclosing_levels(w, fracs):
    """Values of w above which the given fractions of sum(w) are contained."""
    v = np.sort(w[np.isfinite(w)].ravel())[::-1]
    c = np.cumsum(v) / v.sum()
    return [v[np.searchsorted(c, f)] for f in fracs]


def fig_maps():
    gr, nw = disc_map(1.0), disc_map(0.0, mrot=1.0, law=int(REF == "flatloc"))
    lq = np.log10(gr["q"])
    vmax = np.nanmax(lq); vmin = vmax - 5
    lr = np.log10(gr["q"] / nw["q"])
    X, Z = gr["X"], gr["Z"]
    A = 1 - 2 / np.hypot(X, Z)
    contrib = X * np.sqrt(np.clip(A, 0, None)) * gr["q"]      # dE_inf / (dvarpi dz), up to 2 pi
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 3.0), sharey=True)
    ax = axes[0]
    cs = ax.contourf(X, Z, np.clip(lq, vmin, vmax), levels=np.linspace(vmin, vmax, 21),
                     cmap=plots.BLUES)
    cb = fig.colorbar(cs, ax=ax, shrink=0.8, pad=0.24, orientation="horizontal")
    cb.set_label(r"$\log_{10}\dot q\ [{\rm MeV\,cm^{-3}\,s^{-1}}]$")
    cb.set_ticks(np.arange(np.ceil(vmin), vmax + 0.01, 1))
    ax.contour(X, Z, gr["pr"], levels=[0.0], colors=plots.SERIES[1], linewidths=1.0)
    ax.set_title("Schwarzschild", fontsize=9, color=plots.INK)
    ax = axes[1]
    lv = np.linspace(0.0, np.log10(5.0), 22)
    cs2 = ax.contourf(X, Z, np.clip(lr, 0.0, np.log10(5.0)), levels=lv, cmap=plots.REDS)
    ax.contour(X, Z, lr, levels=[np.log10(2.0)], colors=plots.INK2, linewidths=0.8)
    cb2 = fig.colorbar(cs2, ax=ax, shrink=0.8, pad=0.24, orientation="horizontal",
                       ticks=np.log10([1, 1.5, 2, 3, 5]))
    cb2.ax.set_xticklabels(["1", "1.5", "2", "3", "5"])
    cb2.set_label(r"$\dot q_{\rm GR}/\dot q_{\rm flat}$")
    ax.set_title("gravity alone", fontsize=9, color=plots.INK)
    ax = axes[2]
    lc = np.log10(contrib / np.nanmax(contrib))
    cs3 = ax.contourf(X, Z, np.clip(lc, -3, 0), levels=np.linspace(-3, 0, 19), cmap=plots.BLUES)
    lev = enclosing_levels(contrib, [0.9, 0.5])
    ax.contour(X, Z, contrib, levels=sorted(lev), colors=[plots.SERIES[1]] * 2,
               linewidths=[0.8, 1.2])
    cb3 = fig.colorbar(cs3, ax=ax, shrink=0.8, pad=0.24, orientation="horizontal",
                       ticks=[-3, -2, -1, 0])
    cb3.set_label(r"$\log_{10}$ contribution to $\dot E_\infty$ (rel.)")
    ax.set_title("where the energy is deposited", fontsize=9, color=plots.INK)
    for ax in axes:
        ax.plot([6, 40], [0.3, 0.3], color=plots.INK, lw=2.5, solid_capstyle="butt")
        ax.add_patch(plt.Circle((0, 0), 2.0, color=plots.INK))
        ax.set_aspect("equal"); ax.grid(False)
        ax.set_xlim(0, 40); ax.set_ylim(0, 40)
        ax.set_xlabel(r"$\varpi/M$")
    axes[0].set_ylabel(r"$z/M$")
    fig.savefig(os.path.join(FIG, "disc_maps.pdf"))
    plt.close(fig)


def fig_flavour():
    with open(os.path.join(OUT, "flavour.json")) as f:
        res = json.load(f)
    rows = res["default|0.00"]["equal"]
    P = np.array([w["P"] for w in rows])
    De, Dx = ph.couplings(0.23)
    Pf = np.linspace(0, 2 / 3, 100)
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ax.plot(P, [w["R_gr"] for w in rows], "-o", color=plots.SERIES[0], mfc="white", mew=1.2,
            label="Schwarzschild")
    ax.plot(P, [w["R_" + REF] for w in rows], "--", color=plots.SERIES[0], label=FLATLAB[REF])
    ax.plot(P, [w["R_common"] for w in rows], ":", color=plots.SERIES[2],
            label=r"common $\nu_e$ neutrinosphere")
    ax.plot(Pf, (1 - Pf)**2 + Dx / (2 * De) * Pf**2, color=plots.SERIES[1],
            label=r"no $\nu_x$ emission")
    ax.set_xlabel(r"$P=\bar P$"); ax.set_ylabel(r"$\dot E_\infty(P)\,/\,\dot E_\infty(0)$")
    ax.axvline(2 / 3, color=plots.GRID, lw=0.8)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=6.5)
    fig.savefig(os.path.join(FIG, "flavour_ratio.pdf"))
    plt.close(fig)


def fig_layouts():
    ri, rh = 2.321, 1.436
    lay = {"(i)": {"e": (ri, 40), "ebar": (ri, 25), "x": (ri, 12)},
           "(ii)": {"e": (ri, 40), "ebar": (ri, 25), "x": (ri, 2 * ri)},
           "(iii)": {"e": (2 * ri, 40), "ebar": (2 * ri, 25), "x": (ri, 2 * ri)}}
    lab = {"e": r"$\nu_e$", "ebar": r"$\bar\nu_e$", "x": r"$\nu_x$"}
    col = {"e": plots.SERIES[0], "ebar": plots.SERIES[1], "x": plots.SERIES[2]}
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    y = 0
    ticks = []
    for name, rng in lay.items():
        for sp in ("e", "ebar", "x"):
            a, b = rng[sp]
            ax.plot([a, b], [y, y], color=col[sp], lw=4, solid_capstyle="butt")
            if name == "(i)":
                ax.text(b * 1.08, y, lab[sp], va="center", fontsize=7, color=plots.INK)
            y -= 1
        ticks.append((y + 2, name))
        y -= 0.8
    ax.axvline(rh, color=plots.INK, lw=1.0)
    ax.axvline(ri, color=plots.INK2, lw=0.6, ls=":")
    ax.text(rh * 0.97, (y + 1.9) / 2, "horizon", fontsize=6.5, color=plots.INK, ha="right",
            va="center", rotation=90)
    ax.text(ri * 1.04, 1.0, "ISCO", fontsize=6.5, color=plots.INK2)
    ax.set_xscale("log"); ax.set_xlim(1.0, 70); ax.set_ylim(y + 0.3, 1.6)
    ax.set_xticks([1, 2, 5, 10, 20, 50]); ax.set_xticklabels(["1", "2", "5", "10", "20", "50"])
    ax.minorticks_off()
    ax.set_yticks([t for t, _ in ticks]); ax.set_yticklabels([n for _, n in ticks])
    ax.set_xlabel(r"$r/M$ (emitting range, $a=0.9$)")
    ax.grid(axis="y", visible=False)
    fig.savefig(os.path.join(FIG, "layouts.pdf"))
    plt.close(fig)


def fig_robustness():
    with open(os.path.join(OUT, "flavour.json")) as f:
        res = json.load(f)
    with open(os.path.join(OUT, "flavour_sweep.json")) as f:
        sw = json.load(f)
    prof = {"": r"$r^{-1/2}$", "_pt": "PT", "_sT0.25": r"$r^{-1/4}$", "_sT0.75": r"$r^{-3/4}$"}
    names = {"default": "(i)", "xinner": "(ii)", "disjoint": "(iii)"}
    colour = {"default": plots.SERIES[0], "xinner": plots.SERIES[1], "disjoint": plots.SERIES[2]}
    groups = [("overlapping: layout (i)", [("default", t) for t in ("", "_pt", "_sT0.25", "_sT0.75")]),
              ("segregated: layouts (ii), (iii)", [("xinner", ""), ("disjoint", ""),
                                                  ("disjoint", "_pt"), ("disjoint", "_sT0.25"),
                                                  ("disjoint", "_sT0.75")])]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.6), sharex=True)
    fig.subplots_adjust(wspace=0.55, bottom=0.25)
    for ax, (title, cfgs) in zip(axes, groups):
        rows = [(m, t, sp) for m, t in cfgs for sp in ("0.00", "0.90", "0.99")
                if f"{m}{t}|{sp}" in res and not (m == "xinner" and sp == "0.00")]
        ylab = []
        for v in (5, 10, 50):
            ax.axvline(v, color=plots.GRID, lw=1.0, zorder=0)
        for k, (m, t, sp) in enumerate(rows):
            key, y = f"{m}{t}|{sp}", len(rows) - k
            out = res[key]
            base = 100 * F.maxdev(out["equal"], REF)
            d = 100 * np.array([v["dev" if REF == "flatrot" else "dev_" + REF[4:]] for v in sw[key]])
            pmax = 100 * max(F.maxdev(v, REF) for v in out["profiles"].values())
            joint = 100 * max(max(v["prof"][REF].values()) for v in sw[key])
            c = colour[m]
            ax.plot([d.min(), d.max()], [y, y], color=c, lw=1.4)
            ax.plot(np.median(d), y, "D", color=c, ms=4)
            ax.plot(base, y, "o", color=c, mfc="white", mew=1.2, ms=5)
            ax.plot(pmax, y, "^", color=plots.INK2, ms=4)
            ax.plot(joint, y, "x", color=plots.INK2, ms=4, mew=1.0)
            ylab.append((y, f"{names[m]} {prof[t]}, $a={float(sp):g}$"))
        ax.set_yticks([y for y, _ in ylab]); ax.set_yticklabels([t for _, t in ylab], fontsize=7)
        ax.set_ylim(0.3, len(rows) + 0.7)
        ax.set_xscale("log"); ax.set_xlim(0.08, 250)
        ax.set_xticks([0.1, 1, 10, 100]); ax.set_xticklabels(["0.1", "1", "10", "100"])
        ax.set_xlabel(r"$\delta$ [%]")
        ax.set_title(title, fontsize=9, color=plots.INK)
        ax.grid(axis="y", visible=False)
    axes[0].plot([], [], "o", color=plots.INK2, mfc="white", label="reference temperatures")
    axes[0].plot([], [], "D", color=plots.INK2, ms=4, label="36 ratios: median, range")
    axes[0].plot([], [], "^", color=plots.INK2, ms=4, label="non-uniform $P$ (largest)")
    axes[0].plot([], [], "x", color=plots.INK2, ms=4, mew=1.0,
                 label=r"ratios $\times$ profiles (largest)")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=4, fontsize=6.5,
               bbox_to_anchor=(0.5, 0.0))
    fig.savefig(os.path.join(FIG, "flavour_robustness.pdf"))
    plt.close(fig)


def fig_emission():
    with open(os.path.join(OUT, "delta_vs_grey.json")) as f:
        d = json.load(f)[REF]
    lab = {"default": "(i)", "xinner": "(ii)", "disjoint": "(iii)"}
    col = {"default": plots.SERIES[0], "xinner": plots.SERIES[1], "disjoint": plots.SERIES[2]}
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for key, rows in d.items():
        model, spin = key.split("|")
        e = np.array([r[0] for r in rows[1:]])
        v = np.array([r[1] for r in rows[1:]])
        ls = "-" if spin == "0.99" else "--"
        ax.plot(e, v, ls, marker="o", ms=3, color=col[model])
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"heavy-lepton emission factor $\epsilon_x$")
    ax.set_ylabel(r"$\delta$ [%]")
    ax.set_ylim(0.05, 300)
    hd = [plt.Line2D([], [], color=col[m], marker="o", ms=3, label=lab[m]) for m in lab]
    hd += [plt.Line2D([], [], color=plots.INK2, ls=ls, label=f"$a={a}$")
           for ls, a in (("-", "0.99"), ("--", "0.9"))]
    ax.legend(handles=hd, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=5, fontsize=6.5,
              handlelength=1.8, columnspacing=1.0)
    fig.savefig(os.path.join(FIG, "delta_emission.pdf"))
    plt.close(fig)


def fig_reversal():
    with open(os.path.join(OUT, "decompose.json")) as f:
        dec = json.load(f)
    kT = {"e": 4.0, "ebar": 6.0, "x": 4.0}
    grey = {sp: 1.0 for sp in F.SPECIES}
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), gridspec_kw=dict(width_ratios=[1, 1.1]))
    fig.subplots_adjust(wspace=0.35)
    ax = axes[0]
    for model, ls in (("disjoint", "-"), ("default", "--")):
        cases = {c: F.Case(F.fname(c, model, 0.99, "")) for c in ("gr", REF)}
        rr = F.factors(cases, kT, grey, "uniform")
        P = [w["P"] for w in rr]
        lab = "(iii)" if model == "disjoint" else "(i)"
        ax.plot(P, [w["R_gr"] for w in rr], ls, color=plots.SERIES[0], label=f"{lab} Kerr")
        ax.plot(P, [w["R_" + REF] for w in rr], ls, color=plots.SERIES[1],
                label=f"{lab} flat")
    ax.axhline(1, color=plots.INK2, lw=0.6)
    ax.set_xlabel(r"$P=\bar P$")
    ax.set_ylabel(r"$\mathcal{R}=\dot E_\infty(P)/\dot E_\infty(0)$".replace("\\mathcal{R}", "R"))
    ax.legend(loc="lower left", fontsize=6.5)
    ax.set_title(r"$a=0.99$, $T_x=T_e$, $T_{\bar e}=1.5\,T_e$", fontsize=8, color=plots.INK)
    ax = axes[1]
    keys = [("oo", "outer-outer"), ("ii", "inner-inner"), ("cross", "inner-outer")]
    width = 0.2
    x = np.arange(len(keys))
    for k, (case, c) in enumerate((("gr", plots.SERIES[0]), (REF, plots.SERIES[1]))):
        d = dec[case]
        for j, (sh, hatch) in enumerate((("share0", None), ("share1", "////"))):
            vals = [d[sh]["oo"], d[sh]["ii"], d[sh]["oi"] + d[sh]["io"]]
            pos = x + (2 * k + j - 1.5) * width
            ax.bar(pos, vals, width * 0.92, color=c if hatch is None else "white", edgecolor=c,
                   hatch=hatch, lw=1.0)
    ax.set_xticks(x); ax.set_xticklabels([n for _, n in keys], fontsize=8)
    ax.set_ylabel(r"contribution / $\dot E_\infty(0)$")
    from matplotlib.patches import Patch
    hd = [Patch(facecolor=plots.SERIES[0], edgecolor=plots.SERIES[0], label="Kerr, $P=0$"),
          Patch(facecolor="white", edgecolor=plots.SERIES[0], hatch="////", label="Kerr, $P=2/3$"),
          Patch(facecolor=plots.SERIES[1], edgecolor=plots.SERIES[1], label="flat, $P=0$"),
          Patch(facecolor="white", edgecolor=plots.SERIES[1], hatch="////", label="flat, $P=2/3$")]
    ax.legend(handles=hd, loc="upper left", fontsize=6.5, ncol=2)
    ax.set_ylim(0, 1.15)
    ax.grid(axis="x", visible=False)
    ax.set_title("layout (iii): pairs by emission region", fontsize=8, color=plots.INK)
    fig.savefig(os.path.join(FIG, "flavour_reversal.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    plots.style()
    os.makedirs(FIG, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)
    which = sys.argv[1:] or ["sphere", "maps", "flavour", "robust", "reversal", "layouts", "emission"]
    if "sphere" in which:
        fig_sphere()
    if "maps" in which:
        fig_maps()
    if "flavour" in which:
        fig_flavour()
    if "robust" in which:
        fig_robustness()
    if "reversal" in which:
        fig_reversal()
    if "layouts" in which:
        fig_layouts()
    if "emission" in which:
        fig_emission()
