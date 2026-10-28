# Published values used for comparison

Some results of the paper are compared with values published by other authors. Those values are
not distributed with this code. They come from:

- **Salmonson & Wilson (1999)**, J. D. Salmonson and J. R. Wilson, *General relativistic
  augmentation of neutrino pair annihilation energy deposition near neutron stars*, ApJ 517, 859
  ([doi:10.1086/307232](https://doi.org/10.1086/307232), arXiv:astro-ph/9908017). The
  relativistic enhancement F(M/R) of the energy deposition around a neutrinosphere, as quoted in
  the text for R/M = 10, 5 and 3. They appear as squares in Fig. 1.
- **Birkl et al. (2007)**, R. Birkl, M. A. Aloy, H.-Th. Janka and E. Müller, *Neutrino pair
  annihilation near accreting, stellar-mass black holes*, A&A 463, 51
  ([doi:10.1051/0004-6361:20066293](https://doi.org/10.1051/0004-6361:20066293),
  arXiv:astro-ph/0608543). Table 1: the energy deposition rates of models D, DN, S and SN (Table 1
  of the paper) and D, DA1, REF, A.5 and A1 (Table 2 of the paper).

The paper quotes these values in Fig. 1 and Tables 1 and 2. Every script runs without them and
writes all of its own results. To print the published values next to ours and to plot them in
Fig. 1, create `data/published_values.json` with this structure:

```json
{
  "salmonson_wilson_1999": {"R_over_M": [10, 5, 3], "F": [F10, F5, F3]},
  "birkl_2007_table1": {
    "D":   [E_tot, E_up, E_tot_inf, E_up_inf],
    "DN":  [E_tot, E_up, E_tot_inf, E_up_inf],
    "S":   [E_tot],
    "SN":  [E_tot],
    "DA1": [E_tot, E_up, E_tot_inf, E_up_inf],
    "REF": [E_tot, E_up, E_tot_inf, E_up_inf],
    "A.5": [E_tot, E_up, E_tot_inf, E_up_inf],
    "A1":  [E_tot, E_up, E_tot_inf, E_up_inf]
  }
}
```

The rates are in units of 10^49 erg/s, in the order of the columns of Table 1 of Birkl et al.:
the deposition over the whole volume and over the region of outgoing radial momentum, measured
locally (E_tot, E_up) and at infinity (E_tot_inf, E_up_inf). The file is listed in `.gitignore`.
