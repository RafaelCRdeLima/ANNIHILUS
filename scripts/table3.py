"""Rows of Table 3 (LaTeX) from output/flavour.json and output/flavour_sweep.json.

Columns: layout, T(r), a, gain over the flat reference and over static flat discs; delta for the
reference temperatures (in parentheses with Pbar from lepton-number conservation), versus static
discs, largest over the profiles, median and largest over the 36 ratios, largest over ratios x
profiles, and largest over the 36 ratios with the other flat reference.
Usage: table3.py [ref], ref = flatloc (default) or flatrot, the flat-space reference of delta.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import flavour as F  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "output")
ORDER = [("default", "", s) for s in ("0.00", "0.90", "0.99")] + \
        [("default", "_pt", s) for s in ("0.00", "0.90", "0.99")] + \
        [("default", "_sT0.25", "0.90"), ("default", "_sT0.75", "0.90"),
         ("xinner", "", "0.90"), ("xinner", "", "0.99")] + \
        [("disjoint", "", s) for s in ("0.00", "0.90", "0.99")] + \
        [("disjoint", "_pt", s) for s in ("0.00", "0.90", "0.99")] + \
        [("disjoint", "_sT0.25", "0.90"), ("disjoint", "_sT0.75", "0.90")]
LAY = {"default": "(i)", "xinner": "(ii)", "disjoint": "(iii)"}
TPROF = {"": r"$r^{-1/2}$", "_pt": "PT", "_sT0.25": r"$r^{-1/4}$", "_sT0.75": r"$r^{-3/4}$"}


def dev(rows, ref):
    return 100 * max(abs(w["R_gr"] / w["R_" + ref] - 1) for w in rows)


def f(x):
    return f"{x:.1f}" if x < 9.95 else f"{x:.0f}"


def main(ref="flatloc"):
    res = json.load(open(os.path.join(OUT, "flavour.json")))
    sw = json.load(open(os.path.join(OUT, "flavour_sweep.json")))
    dkey = "dev" if ref == "flatrot" else "dev_" + ref[4:]
    for m, t, s in ORDER:
        key = f"{m}{t}|{s}"
        o, v = res[key], sw[key]
        e0 = o["equal"][0]
        gl = e0["gain_" + ref[4:]]
        other = "flatrot" if ref == "flatloc" else "flatloc"
        okey = "dev" if other == "flatrot" else "dev_" + other[4:]
        omax = f(100 * max(x[okey] for x in v))
        gs = f"{e0['gain_static']:.2f}" if "gain_static" in e0 else "--"
        d = dev(o["equal"], ref)
        lep = "--" if o["lepton"] is None else f(100 * F.maxdev_exact(o["lepton"], ref))
        tot = f(dev(o["equal"], "newt")) if "R_newt" in e0 else "--"
        prof = f(max(dev(r, ref) for r in o["profiles"].values()))
        ds = 100 * np.array([x[dkey] for x in v])
        joint = 100 * max(max(x["prof"][ref].values()) for x in v)
        print(f"{LAY[m]} & {TPROF[t]} & {float(s):g} & {gl:.2f} & {gs} & {f(d)} ({lep}) & {tot} "
              f"& {prof} & {f(np.median(ds))} & {f(ds.max())} & {f(joint)} & {omax}\\\\")


if __name__ == "__main__":
    main(*sys.argv[1:])
