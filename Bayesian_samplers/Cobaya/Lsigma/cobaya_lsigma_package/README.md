# Cobaya + CLASS likelihood for the HII-galaxy L-sigma relation

This package translates the supplied MultiNest/Astropy workflow into a Cobaya external likelihood using CLASS as the cosmological theory provider.

## Files

- `lsigma_hiig_likelihood.py`: external Cobaya likelihood class.
- `yaml/H0/`: H0-only conditional background model (Omega_m fixed to 0.30).
- `yaml/wCDM/`: H0 + Omega_m + constant w_DE.
- `yaml/w0waCDM/`: H0 + Omega_m + CPL (w0, wa).

Each model directory contains:

1. `hiig_cepheids.yaml`
2. `hiig_trgb.yaml`
3. `planck.yaml`
4. `planck_hiig_cepheids.yaml`
5. `planck_hiig_trgb.yaml`

Thus there are 15 ready-to-edit run files.

## Expected data files

Put these next to `lsigma_hiig_likelihood.py`, or replace them by absolute paths in the YAML:

- `Compilation2026.csv`
- `GEHR_Cepheids.csv`
- `GEHR_TRGB.csv`

The L-sigma catalogue is expected to contain the columns used by the supplied sampler:

- `GEHR_id`
- `origin_id` (`0` = GEHR anchor, non-zero = HIIG)
- `z_or_mu`
- `e_z_or_e_mu`
- `log_sigma`
- `e_log_sigma`
- `log_f_Hbeta`
- `e_log_f_Hbeta`

The anchor table is expected to contain one homogenized distance per host:

- `Galaxia`
- `mu_w`
- `sigma_w`

If your columns have different names, all of them can be overridden in the likelihood block.

## Statistical model

The relation is

```text
log10 L(Hbeta) = alpha_ls + beta_ls log10(sigma)
```

and the inferred distance modulus is

```text
mu_Lsigma = 2.5(alpha_ls + beta_ls log_sigma)
            - 2.5 log_f_Hbeta
            - 100.19477738511641.
```

For anchors, the comparison modulus is the independent `mu_w` in the anchor table.
For HIIG, CLASS provides `D_A(z)` and the code evaluates

```text
D_L(z) = (1+z)^2 D_A(z)
mu(z)  = 5 log10[D_L/Mpc] + 25.
```

For the redshift error, in the flat models supplied here,

```text
dmu/dz = (5/ln 10) [ 1/(1+z) + c/(H(z) D_C(z)) ]
D_C=(1+z)D_A.
```

The L-sigma measurement variance is

```text
sigma_mu,Lsigma^2 = 2.5^2 [sigma_logf^2 + beta_ls^2 sigma_logsigma^2]
                    + (2.5 sigma_int)^2.
```

`sigma_int` is in dex in log10 L and is fixed to zero in the supplied YAMLs, matching the no-intrinsic-scatter setup of the original code.

## Two important switches

The supplied YAMLs use the statistically recommended settings:

```yaml
include_logdet: true
correlate_host_mu: true
```

`include_logdet: true` uses the fully normalized Gaussian likelihood. This matters because the variance depends on the fitted slope `beta_ls`.

`correlate_host_mu: true` treats the distance-modulus uncertainty as a host-level error shared by all GEHR belonging to the same galaxy. If several regions share one Cepheid/TRGB distance, the same distance information is therefore not counted independently several times.

To reproduce the algebra of the original sampler as closely as possible, use

```yaml
include_logdet: false
correlate_host_mu: false
sigma_int: 0.0
```

## Optional cross-host anchor covariance

If you have a covariance matrix between host distance moduli (for example from a shared Cepheid/TRGB zero point), provide

```yaml
anchor_cov_file: GEHR_Cepheids_cov.csv
```

The CSV must be a square covariance matrix in mag^2, with host names as both row labels and column names. If this file is absent, the code assumes different host moduli are independent but still correlates multiple GEHR inside each host.

## Intrinsic scatter

The supplied runs fix it to zero. To sample it, replace

```yaml
sigma_int: 0.0
```

with for example

```yaml
sigma_int:
  prior: {min: 0.0, max: 0.5}
  ref: {dist: norm, loc: 0.15, scale: 0.03}
  proposal: 0.01
  latex: \sigma_{\rm int}
```

The units are dex in log10 L(Hbeta).

## Cosmological models

### `yaml/H0/`

- sampled target: `H0`
- fixed: `Omega_m = 0.30`, flat LambdaCDM background

This is a **conditional H0 constraint**. In the Planck-only H0 YAML, the primordial/recombination parameters needed by the CMB are still varied, but Omega_m is fixed. It is therefore not identical to the usual six-parameter Planck LambdaCDM H0 posterior.

### `yaml/wCDM/`

- sampled targets: `H0`, `Omega_m`, `w0_fld`
- `wa_fld = 0`
- `Omega_Lambda = 0`, so CLASS uses the dark-energy fluid
- `use_ppf = yes`

`w0_fld` is the constant `w_DE` parameter.

### `yaml/w0waCDM/`

- sampled targets: `H0`, `Omega_m`, `w0_fld`, `wa_fld`
- CPL form: `w(a)=w0_fld + wa_fld(1-a)`
- `Omega_Lambda = 0`
- `use_ppf = yes`

## Planck choice

The Planck YAMLs use a Planck 2018 combination:

```yaml
planck_2018_lowl.TT: null
planck_2018_lowl.EE: null
planck_2018_highl_plik.TTTEEE_lite_native: null
planck_2018_lensing.native: null
```

The high-l likelihood is the nuisance-marginalized Plik-lite native implementation, which keeps the example compact. Cobaya adds any likelihood-specific nuisance parameters that are still required.

The CMB runs additionally vary `logA/A_s`, `n_s`, `omega_b`, and `tau_reio`, and fix one massive neutrino to 0.06 eV. These parameters are necessary to predict the CMB even when H0 is the only late-time target of interest.

## Running

From this package directory:

```bash
cobaya-run yaml/H0/hiig_cepheids.yaml --test
```

If initialization succeeds:

```bash
mpirun -np 4 cobaya-run yaml/H0/hiig_cepheids.yaml
```

For a combined CPL run:

```bash
cobaya-run yaml/w0waCDM/planck_hiig_cepheids.yaml --test
mpirun -np 4 cobaya-run yaml/w0waCDM/planck_hiig_cepheids.yaml
```

If your CLASS/Planck packages are not in Cobaya's default package directory, add near the top of each YAML:

```yaml
packages_path: /your/path/to/cobaya_packages
```

Install the theory/likelihood data using your normal `cobaya-install` workflow before running Planck.

## Recommended validation sequence

1. Run `--test` on `yaml/H0/hiig_cepheids.yaml`.
2. Temporarily set `include_logdet: false` and `correlate_host_mu: false` and compare the log-likelihood at a fixed `(alpha_ls,beta_ls,H0,Omega_m,w)` against the old MultiNest code.
3. Restore the recommended covariance/log-determinant treatment.
4. Run HIIG Cepheids and HIIG TRGB separately.
5. Run Planck separately.
6. Only then run the combined likelihoods.

This makes any discrepancy between the old Astropy implementation and the new CLASS/Cobaya implementation easy to isolate.
