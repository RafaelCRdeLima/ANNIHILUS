# =============================================================================
#  ANNIHILUS: neutrino pair annihilation in curved spacetime
#
#  Reproduces the numbers, tables and figures of the paper
#    "Neutrino pair annihilation in curved spacetime: a fast moment method and
#     the geometry of flavour conversion".
#
#    make help         list the targets
#    make              build the C engine (python/annihilus/libannihilus.so)
#    make test         run the 25 unit tests (~10 s)
#    make quick        build, test, validate and make Figs. 1-3 (~1 h 10 min)
#    make paper        reproduce the whole paper (~5 h on 12 cores)
#
#    output/    results (.json, .txt, .log, .tex and the .npz moment files)
#    figures/   the figures of the paper, in PDF
#    results/   the results we obtained, for comparison
#
#  The engine uses OpenMP and all cores; to use 4, for example:
#    OMP_NUM_THREADS=4 make paper
#  The times below were measured on 12 cores. make only redoes what is missing:
#  if a run is interrupted, the same command resumes after the last finished run.
# =============================================================================

PY      ?= python3
CC      ?= gcc
CFLAGS  ?= -O3 -march=native -fopenmp -fPIC -Wall -Wextra -std=c11 -D_DEFAULT_SOURCE
LDFLAGS ?= -fopenmp -lm

OUT = output
FIG = figures
DONE = $(OUT)/.done
LIB = python/annihilus/libannihilus.so
SRC = src/raytrace.c src/kerr.c src/emitter.c src/moments.c
OBJ = $(SRC:.c=.o)

export ANNIHILUS_FIG := $(CURDIR)/$(FIG)

# a program that fails inside "| tee" stops make, which deletes the incomplete file
SHELL := /bin/bash
.SHELLFLAGS := -eo pipefail -c
.DELETE_ON_ERROR:

all: $(LIB)

$(OUT) $(DONE):
	mkdir -p $@


# -----------------------------------------------------------------------------
# 1. C engine -> python/annihilus/libannihilus.so (called through ctypes)
#      src/raytrace.c   backward ray tracing in Schwarzschild spacetime (Sections 4, 5)
#      src/kerr.c       Kerr geodesics in Mino time (Section 5.3)
#      src/emitter.c    frequency ratio and temperature at emission (sphere, Keplerian
#                       disc, zero-angular-momentum disc, flat-space discs, Page-Thorne)
#      src/moments.c    edge-adapted quadrature on the local sky and angular moments
#                       (Sections 3, 7)
# -----------------------------------------------------------------------------
$(LIB): $(OBJ)
	$(CC) -shared -o $@ $(OBJ) $(LDFLAGS)

src/%.o: src/%.c src/annihilus.h
	$(CC) $(CFLAGS) -c $< -o $@


# -----------------------------------------------------------------------------
# 2. Unit tests (25 tests, ~10 s)
#      tests/test_sphere.py   sphere: moments against the analytic cone, rates
#      tests/test_disc.py     flat annulus against direct integration, symmetries,
#                             weight blocks, Doppler shifts of the flat-space discs
#      tests/test_flavour.py  lepton-number condition and exact maximum of delta in P
#      tests/test_kerr.py     Kerr: first integrals, oblate flat space and geodesics
#                             from Hamilton's equations (tests/kerr_hamilton.py)
# -----------------------------------------------------------------------------
test: $(LIB)
	cd python && $(PY) -m unittest discover -s ../tests -v


# -----------------------------------------------------------------------------
# 3. Validation (Section 8.1, Tables 1 and 2, Fig. 1, Appendix A), ~1 h
# -----------------------------------------------------------------------------
# sphere of Salmonson & Wilson and models D, DN, S, SN of Birkl et al. (~1 min)
$(OUT)/validation.json: $(LIB) scripts/validate.py | $(OUT)
	$(PY) -u scripts/validate.py | tee $(OUT)/validation.log

# spin series of Birkl et al. in their convention (static observers outside the
# ergosphere), ZAMO rates and the exact energy at infinity (~25 min)
$(OUT)/birkl_kerr.json: $(LIB) scripts/birkl_kerr.py | $(OUT)
	$(PY) -u scripts/birkl_kerr.py | tee $(OUT)/birkl_kerr.log

# integrals for a = 0.9999 as functions of the inner boundary (~25 min)
$(OUT)/einf_convergence.log: $(LIB) scripts/einf_convergence.py | $(OUT)
	$(PY) -u scripts/einf_convergence.py | tee $@

# integrals in the convention of Birkl et al. on a refined volume grid (~20 min)
$(OUT)/birkl_static_convergence.log: $(LIB) scripts/birkl_static_convergence.py | $(OUT)
	$(PY) -u scripts/birkl_static_convergence.py | tee $@

validation: $(OUT)/validation.json $(OUT)/birkl_kerr.json $(OUT)/einf_convergence.log \
            $(OUT)/birkl_static_convergence.log


# -----------------------------------------------------------------------------
# 4. Flavour runs (Section 8.3), ~1 h 45 min in total
#
#    scripts/flavour_runs.py computes on an (r, theta) grid the angular moments
#    of each species (nu_e, nubar_e, nu_x), with four weight blocks for
#    non-uniform conversion, for the cases
#      gr        general relativity (Schwarzschild or Kerr), Keplerian disc
#      flatloc   flat space, discs moving with the local speeds of the
#                relativistic disc (the reference of delta)
#      flatrot   flat space, discs at Newtonian Keplerian speeds
#      newt      flat space, static discs
#    and writes output/flavour_<case>[_<layout>][_pt|_sT<s>]_a<spin>.npz.
#    Layouts (Fig. 3): default (i), xinner (ii), disjoint (iii). Temperature
#    profiles: T ~ r^(-s) (s = 0.5 unless --sT) or Page-Thorne (--profile pt).
#    Each target is one run, marked as finished by a file in output/.done/.
# -----------------------------------------------------------------------------
# (i), T ~ r^-1/2, a = 0, 0.9, 0.99 (~11 min)
$(DONE)/i: $(LIB) scripts/flavour_runs.py | $(DONE)
	$(PY) -u scripts/flavour_runs.py --cases gr,newt,flatrot,flatloc 0 0.9 0.99 | tee $@.log
	touch $@

# (iii), T ~ r^-1/2, a = 0, 0.9, 0.99 (~23 min)
$(DONE)/iii: $(LIB) scripts/flavour_runs.py | $(DONE)
	$(PY) -u scripts/flavour_runs.py --model disjoint --cases gr,newt,flatrot,flatloc 0 0.9 0.99 | tee $@.log
	touch $@

# (ii), T ~ r^-1/2, a = 0.9, 0.99 (~10 min); at a = 0, (ii) coincides with (i)
$(DONE)/ii: $(LIB) scripts/flavour_runs.py | $(DONE)
	$(PY) -u scripts/flavour_runs.py --model xinner --cases gr,newt,flatrot,flatloc 0.9 0.99 | tee $@.log
	touch $@

# (i) and (iii), Page-Thorne profile (~10 and ~21 min)
$(DONE)/i_pt: $(LIB) scripts/flavour_runs.py | $(DONE)
	$(PY) -u scripts/flavour_runs.py --profile pt --cases gr,flatrot,flatloc 0 0.9 0.99 | tee $@.log
	touch $@

$(DONE)/iii_pt: $(LIB) scripts/flavour_runs.py | $(DONE)
	$(PY) -u scripts/flavour_runs.py --model disjoint --profile pt --cases gr,flatrot,flatloc 0 0.9 0.99 | tee $@.log
	touch $@

# (i) and (iii), T ~ r^-1/4 and r^-3/4, a = 0.9 (~4 min each for (i), ~10 min for (iii))
$(DONE)/i_s025: $(LIB) scripts/flavour_runs.py | $(DONE)
	$(PY) -u scripts/flavour_runs.py --sT 0.25 --cases gr,flatrot,flatloc 0.9 | tee $@.log
	touch $@

$(DONE)/i_s075: $(LIB) scripts/flavour_runs.py | $(DONE)
	$(PY) -u scripts/flavour_runs.py --sT 0.75 --cases gr,flatrot,flatloc 0.9 | tee $@.log
	touch $@

$(DONE)/iii_s025: $(LIB) scripts/flavour_runs.py | $(DONE)
	$(PY) -u scripts/flavour_runs.py --model disjoint --sT 0.25 --cases gr,flatrot,flatloc 0.9 | tee $@.log
	touch $@

$(DONE)/iii_s075: $(LIB) scripts/flavour_runs.py | $(DONE)
	$(PY) -u scripts/flavour_runs.py --model disjoint --sT 0.75 --cases gr,flatrot,flatloc 0.9 | tee $@.log
	touch $@

RUNS = $(DONE)/i $(DONE)/ii $(DONE)/iii $(DONE)/i_pt $(DONE)/iii_pt \
       $(DONE)/i_s025 $(DONE)/i_s075 $(DONE)/iii_s025 $(DONE)/iii_s075

runs: $(RUNS)


# -----------------------------------------------------------------------------
# 5. Flavour analysis (Section 8.3, Table 3, data of Figs. 4-7)
# -----------------------------------------------------------------------------
# conversion factors, delta (Eq. 19) for both flat-space references, gains,
# lepton-number-conserving Pbar, conversion profiles, the 36 temperature and
# luminosity ratios and their combination with the profiles (~20 min);
# scripts/table3.py writes the rows of Table 3 to output/table3.tex
$(OUT)/flavour.json: $(RUNS) scripts/flavour.py scripts/table3.py
	$(PY) -u scripts/flavour.py | tee $(OUT)/flavour_table.txt
	$(PY) scripts/table3.py > $(OUT)/table3.tex

# delta as a function of the heavy-lepton emission epsilon_x (Fig. 4)
$(OUT)/delta_vs_grey.json: $(DONE)/i $(DONE)/ii $(DONE)/iii scripts/delta_vs_grey.py scripts/flavour.py
	$(PY) -u scripts/delta_vs_grey.py | tee $(OUT)/delta_vs_grey.log

# decomposition of the sign reversal by pairs of emission regions (Fig. 7, right)
$(OUT)/decompose.json: $(DONE)/iii scripts/decompose.py scripts/flavour.py
	$(PY) -u scripts/decompose.py | tee $(OUT)/decompose.log

flavour: $(OUT)/flavour.json $(OUT)/delta_vs_grey.json $(OUT)/decompose.json


# -----------------------------------------------------------------------------
# 6. Cost and convergence (Section 7, Appendix B)
# -----------------------------------------------------------------------------
# moment contraction against the direct double sum (Table B.4) (~1 min)
$(OUT)/benchmark.json: $(LIB) scripts/benchmark.py | $(OUT)
	$(PY) -u scripts/benchmark.py | tee $(OUT)/benchmark.log

# convergence of the global rates with each numerical parameter (~18 min)
$(OUT)/convergence.json: $(LIB) scripts/convergence.py | $(OUT)
	$(PY) -u scripts/convergence.py | tee $(OUT)/convergence.log

# narrow images of the disc against brute-force sampling (Section 7) (a few minutes)
$(OUT)/narrow_images.json: $(LIB) scripts/narrow_images.py | $(OUT)
	$(PY) -u scripts/narrow_images.py | tee $(OUT)/narrow_images.log

# refined quadrature (doubled azimuths and Gauss-Legendre nodes, finer volume grid)
# for the smallest, an intermediate and the largest delta of Table 3
# (~20, ~23 and ~42 min); writes output/flavour_*_ref_*.npz
$(DONE)/ref_i_s075: $(LIB) scripts/flavour_runs.py | $(DONE)
	$(PY) -u scripts/flavour_runs.py --refine --sT 0.75 --cases gr,flatrot,flatloc 0.9 | tee $@.log
	touch $@

$(DONE)/ref_ii: $(LIB) scripts/flavour_runs.py | $(DONE)
	$(PY) -u scripts/flavour_runs.py --refine --model xinner --cases gr,flatrot,flatloc 0.9 | tee $@.log
	touch $@

$(DONE)/ref_iii: $(LIB) scripts/flavour_runs.py | $(DONE)
	$(PY) -u scripts/flavour_runs.py --refine --model disjoint --cases gr,flatrot,flatloc 0.99 | tee $@.log
	touch $@

$(OUT)/delta_convergence.json: $(DONE)/ref_i_s075 $(DONE)/ref_ii $(DONE)/ref_iii \
                               $(DONE)/i_s075 $(DONE)/ii $(DONE)/iii scripts/delta_convergence.py
	$(PY) -u scripts/delta_convergence.py | tee $(OUT)/delta_convergence.log

# maximum of delta over continuous P against the nine-point grid (~30 min)
$(OUT)/check_pmax.log: $(RUNS) scripts/check_pmax.py scripts/flavour.py
	$(PY) -u scripts/check_pmax.py | tee $@

convergence: $(OUT)/benchmark.json $(OUT)/convergence.json $(OUT)/narrow_images.json \
             $(OUT)/delta_convergence.json $(OUT)/check_pmax.log


# -----------------------------------------------------------------------------
# 7. Figures of the paper -> figures/ (scripts/figures.py)
# -----------------------------------------------------------------------------
# Fig. 1: relativistic enhancement around a sphere
$(FIG)/sphere_enhancement.pdf: $(LIB) scripts/figures.py
	$(PY) scripts/figures.py sphere

# Fig. 2: deposition maps; cached in output/map_gr.npz and output/map_flatloc.npz (~5 min)
$(FIG)/disc_maps.pdf: $(LIB) scripts/figures.py | $(OUT)
	$(PY) scripts/figures.py maps

# Fig. 3: the three layouts
$(FIG)/layouts.pdf: $(LIB) scripts/figures.py
	$(PY) scripts/figures.py layouts

# Fig. 4: delta against epsilon_x
$(FIG)/delta_emission.pdf: $(OUT)/delta_vs_grey.json scripts/figures.py
	$(PY) scripts/figures.py emission

# Fig. 5: conversion factor of layout (i) at a = 0
$(FIG)/flavour_ratio.pdf: $(OUT)/flavour.json scripts/figures.py
	$(PY) scripts/figures.py flavour

# Fig. 6: robustness of delta
$(FIG)/flavour_robustness.pdf: $(OUT)/flavour.json scripts/figures.py
	$(PY) scripts/figures.py robust

# Fig. 7: sign reversal and decomposition by pairs
$(FIG)/flavour_reversal.pdf: $(OUT)/decompose.json $(DONE)/i $(DONE)/iii scripts/figures.py
	$(PY) scripts/figures.py reversal

FIGURES = $(FIG)/sphere_enhancement.pdf $(FIG)/disc_maps.pdf $(FIG)/layouts.pdf \
          $(FIG)/delta_emission.pdf $(FIG)/flavour_ratio.pdf $(FIG)/flavour_robustness.pdf \
          $(FIG)/flavour_reversal.pdf

figures: $(FIGURES)


# -----------------------------------------------------------------------------
# 8. Shortcuts
# -----------------------------------------------------------------------------
quick: test validation $(FIG)/sphere_enhancement.pdf $(FIG)/disc_maps.pdf $(FIG)/layouts.pdf

paper: test validation flavour convergence figures
	@echo "Done: results in $(OUT)/, figures in $(FIG)/; compare with results/."

help:
	@echo "make               build the C engine"
	@echo "make test          unit tests (~10 s)"
	@echo "make validation    Tables 1 and 2, Fig. 1, Appendix A (~1 h)"
	@echo "make runs          flavour runs (~1 h 45 min)"
	@echo "make flavour       runs and analysis: Table 3, data of Figs. 4-7"
	@echo "make convergence   cost and convergence, Section 7 and Appendix B (~1 h 50 min)"
	@echo "make figures       the seven figures of the paper in $(FIG)/"
	@echo "make quick         tests, validation and Figs. 1-3 (~1 h 10 min)"
	@echo "make paper         the whole paper (~5 h on 12 cores)"
	@echo "make clean         remove the compiled files"
	@echo "make distclean     remove also $(OUT)/ and $(FIG)/"

clean:
	rm -f $(OBJ) $(LIB)

distclean: clean
	rm -rf $(OUT) $(FIG)

.PHONY: all test validation runs flavour convergence figures quick paper help clean distclean
