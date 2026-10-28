"""Gravitational change of the conversion factor as the heavy-lepton emission is switched on.

delta = max_P |R_gr/R_flat - 1| (flat: flatrot and flatloc) at the reference temperatures, for
heavy-lepton emission scaled by eps_x from 0 (exact separability limit, delta = 0) to 1. Reads the
flavour_*.npz files of flavour_runs.py (layouts default, xinner, disjoint at a = 0.9 and 0.99).
Writes output/delta_vs_grey.json, {reference: {layout|spin: [(eps_x, delta)]}} (Fig. 4).
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import flavour as F  # noqa: E402

GREY = (0.0, 0.001, 0.01, 0.03, 0.1, 0.3, 1.0)


def main():
    out = {ref: {} for ref in F.FLAT}
    for model, spin in (("default", 0.99), ("xinner", 0.99), ("disjoint", 0.99),
                        ("disjoint", 0.9), ("default", 0.9)):
        cases = {c: F.Case(F.fname(c, model, spin, "")) for c in ("gr",) + F.FLAT}
        rows = [F.factors(cases, F.KT, {"e": 1.0, "ebar": 1.0, "x": gx}, "uniform") for gx in GREY]
        for ref in F.FLAT:
            row = [(gx, 100 * F.maxdev(r, ref)) for gx, r in zip(GREY, rows)]
            out[ref][f"{model}|{spin}"] = row
            print(ref, model, spin, " ".join(f"eps={gx:g}:{d:.2f}%" for gx, d in row), flush=True)
    with open(os.path.join(F.OUT, "delta_vs_grey.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
