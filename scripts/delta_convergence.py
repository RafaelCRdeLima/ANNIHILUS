"""Convergence of delta = max_P |R_gr/R_flat - 1|: base versus refined quadrature.

Reads the base and refined (flavour_runs.py --refine, files *_ref_*) runs for the smallest, an
intermediate and the largest delta of Table 3, and compares, for each flat-space
reference (flatrot, flatloc), the rates without conversion, delta at the reference temperatures,
over the 36-case sweep and for the non-uniform profiles. Writes output/delta_convergence.json.
"""
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import flavour as fl  # noqa: E402
OUT = fl.OUT
confs = [("default", "_sT0.75", 0.9), ("xinner", "", 0.9), ("disjoint", "", 0.99)]
res = {}
for model, tag, spin in confs:
    row = {}
    for lab, t in (("base", tag), ("ref", tag + "_ref")):
        cases = {c: fl.Case(fl.fname(c, model, spin, t)) for c in ("gr",) + fl.FLAT}
        z = cases["gr"].z
        g1 = {s: 1.0 for s in fl.SPECIES}
        r = {}
        r["nphi_ngl_nr_nth"] = [int(z[k]) if k in z else None for k in ("n_phi", "n_gl", "nr", "nth")]
        for c in cases:
            r["E0_" + c] = float(cases[c].rate(cases[c].moments(fl.KT, g1), 0, 0))
        rows_ref = fl.factors(cases, fl.KT, g1, "uniform")
        rows_case = fl.factors(cases, {"e": 4.0, "ebar": 6.0, "x": 4.0}, g1, "uniform")
        sweep = [fl.factors(cases, {"e": 4.0, "ebar": 4.0 * teb, "x": 4.0 * tx},
                            {"e": 1, "ebar": 1, "x": gx}, "uniform")
                 for tx, teb, gx in itertools.product((0.8, 1.0, 1.25, 1.5), (1.0, 1.25, 1.5),
                                                      (0.1, 0.3, 1.0))]
        prof = {p: fl.factors(cases, fl.KT, g1, p) for p in fl.PROFILES} if cases["gr"].weighted else {}
        for f in fl.FLAT:
            r[f] = dict(d_ref=fl.maxdev(rows_ref, f), d_sweepcase=fl.maxdev(rows_case, f),
                        R_sweepcase=[rows_case[-1]["R_gr"], rows_case[-1]["R_" + f]],
                        d_sweep=[fl.maxdev(w, f) for w in sweep],
                        d_prof={p: fl.maxdev(w, f) for p, w in prof.items()})
        row[lab] = r
    b, f = row["base"], row["ref"]
    print(f"{model}{tag} a={spin}: grid base {b['nphi_ngl_nr_nth']} ref {f['nphi_ngl_nr_nth']}")
    print("  " + "  ".join(f"E0_{c} rel {f['E0_' + c] / b['E0_' + c] - 1:+.2e}"
                           for c in ("gr",) + fl.FLAT))
    for ref in fl.FLAT:
        B, F = b[ref], f[ref]
        ds_b, ds_r = np.array(B["d_sweep"]), np.array(F["d_sweep"])
        print(f"  vs {ref}: d_ref {100 * B['d_ref']:.4f}% -> {100 * F['d_ref']:.4f}%;  "
              f"d_sweepcase {100 * B['d_sweepcase']:.3f}% -> {100 * F['d_sweepcase']:.3f}% "
              f"(R(2/3) gr {B['R_sweepcase'][0]:.4f} -> {F['R_sweepcase'][0]:.4f}, "
              f"flat {B['R_sweepcase'][1]:.4f} -> {F['R_sweepcase'][1]:.4f})")
        print(f"     sweep max {100 * ds_b.max():.3f}% -> {100 * ds_r.max():.3f}%; max |diff| "
              f"{100 * np.abs(ds_r - ds_b).max():.2e} pp; max rel "
              f"{np.max(np.abs(ds_r - ds_b) / np.maximum(ds_b, 1e-12)):.2e}")
        if B["d_prof"]:
            dp_ = max(abs(F["d_prof"][p] - B["d_prof"][p]) for p in fl.PROFILES)
            print(f"     profiles max |diff| {100 * dp_:.2e} pp; max prof "
                  f"{100 * max(B['d_prof'].values()):.3f}% -> {100 * max(F['d_prof'].values()):.3f}%")
    res[f"{model}{tag}|{spin}"] = row
json.dump(res, open(os.path.join(OUT, "delta_convergence.json"), "w"), indent=1)
