# ANNIHILUS

Code and results of the paper *Neutrino pair annihilation in curved spacetime: a fast moment method
and the geometry of flavour conversion*, by R. C. R. de Lima, L. de Oliveira Junior and P. M. Bailer.

ANNIHILUS computes the energy and momentum deposited by neutrino pair annihilation,
νν̄ → e⁺e⁻, around compact objects. Neutrinos are traced backward from each point in
Schwarzschild or Kerr spacetime to a sphere or a thin disc. The annihilation kernel factorises
exactly into angular moments (ranks 0 to 3) of the redshifted energy moments of each species. A C
engine returns these moments, and Python combines them into rates for any set of species, spectra
and flavour conversion and integrates them over volume.

## Requirements

- Linux (or another Unix) with `make` and a C compiler with OpenMP, such as `gcc`;
- Python 3.9 or newer with numpy 2.0 or newer, scipy and matplotlib:

      pip install -r requirements.txt

## Running

    git clone https://github.com/RafaelCRdeLima/ANNIHILUS.git
    cd ANNIHILUS
    make help         # list the targets
    make              # build the C engine
    make test         # unit tests; all must end with "OK"
    make quick        # tests, validation and Figs. 1-3 (~1 h 10 min on 12 cores)
    make paper        # the whole paper (~5 h on 12 cores)

Results go to `output/` and figures to `figures/`. If a run is interrupted, repeat the command:
`make` resumes after the last finished run. To limit the number of cores:

    OMP_NUM_THREADS=4 make paper

## Contents

| path | content |
|---|---|
| `src/` | C engine: backward ray tracing in Schwarzschild (`raytrace.c`) and Kerr (`kerr.c`), emission (`emitter.c`), edge-adapted quadrature and angular moments (`moments.c`) |
| `python/annihilus/` | interface to the engine (`engine.py`), annihilation and flavour physics (`physics.py`), volume integrals (`deposition.py`), figure style (`plots.py`), optional published values (`published.py`) |
| `scripts/` | one program for each result of the paper |
| `tests/` | unit tests |
| `results/` | the results we obtained, for comparison |
| `data/` | description of the published values used for comparison (not distributed) |
| `Makefile` | runs everything; its comments say what each target computes |
| `LICENSE` | MIT license |

## Where each result appears in the paper

| in the paper | make target | program | output |
|---|---|---|---|
| Fig. 1, Table 1 | `validation` | `validate.py` | `validation.json` |
| Table 2, Section 8.1 | `validation` | `birkl_kerr.py`, `einf_convergence.py`, `birkl_static_convergence.py` | `birkl_kerr.json`, `einf_convergence.log`, `birkl_static_convergence.log` |
| Appendix A | `validation` | `einf_convergence.py` | `einf_convergence.log` |
| Fig. 2 | `figures` | `figures.py maps` | `figures/disc_maps.pdf` |
| Fig. 3 | `figures` | `figures.py layouts` | `figures/layouts.pdf` |
| Table 3, Figs. 5 and 6 | `flavour` | `flavour_runs.py`, `flavour.py`, `table3.py` | `flavour.json`, `flavour_sweep.json`, `flavour_table.txt`, `table3.tex` |
| Fig. 4 | `flavour` | `delta_vs_grey.py` | `delta_vs_grey.json` |
| Fig. 7 | `flavour` | `decompose.py` | `decompose.json` |
| Section 7 (narrow images) | `convergence` | `narrow_images.py` | `narrow_images.json` |
| Appendix B (cost, Table B.4) | `convergence` | `benchmark.py` | `benchmark.json` |
| Appendix B (convergence) | `convergence` | `convergence.py`, `delta_convergence.py`, `check_pmax.py` | `convergence.json`, `delta_convergence.json`, `check_pmax.log` |

All figures are made by `scripts/figures.py` (`make figures`).

## Comparing with our results

The files in `output/` should agree with those in `results/` up to rounding: the engine is
compiled with `-march=native`, so the last digits depend on the processor. The times in
`benchmark.json` depend on the machine. The moment files of the flavour runs
(`output/flavour_*.npz`, about 250 MB) are not included; `make runs` produces them.

## Published values

The validation compares our rates with values published by Salmonson & Wilson (1999) and Birkl
et al. (2007). These values are not distributed here; `data/README.md` gives their source and
how to supply them. Without them all programs run and write all of their own results.

## Units

Inside the engine G = c = 1 and lengths are in units of the mass M of the black hole. In
`physics.py` energies are in MeV, lengths in cm and times in s. The moments are given per unit of
(g_s/h³) Γ(n+1) η(n+1) (kT_ref)^(n+1); `physics.energy_moments` restores the physical scale. The
statistical weight is g_s = 1 (Section 2 of the paper).

## License

MIT; see `LICENSE`.
