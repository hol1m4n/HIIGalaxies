# Setup de Cobaya para la relación L-sigma (HIIG) + Planck

## Archivos

- `lsigma_hiig_likelihood.py` — clase de likelihood externa de Cobaya, port
  directo de la física de `five_par_nested_sampling.py` (misma relación
  L-sigma, mismo tratamiento de anclas GEHR vs. muestra HIIG cosmología-
  dependiente). Parámetros libres propios: `alpha_ls`, `beta_ls`.
- `lik_hiig_cepheids.yaml` — likelihood HIIG anclada con distancias Cepheidas.
- `lik_hiig_trgb.yaml` — likelihood HIIG anclada con distancias TRGB.
- `lik_planck.yaml` — likelihoods Planck 2018 (TT/EE lowl, TTTEEE highl lite,
  lensing).
- `params_h0_only.yaml` — Parametrización 1: solo H0 libre.
- `params_h0_om_wde.yaml` — Parametrización 2: H0, Omega_m, w_DE (constante).
- `params_h0_om_w0wa.yaml` — Parametrización 3: H0, Omega_m, w0, wa (CPL).
- `sampler_mcmc.yaml` — configuración común del muestreador MCMC.

Cobaya permite pasar **varios** archivos yaml a `cobaya-run`; los va
fusionando en orden (el que se declara después puede sobreescribir claves
del anterior). Así puedes armar cualquiera de tus 5 combinaciones de datos ×
3 parametrizaciones sin duplicar nada, con 15 líneas de comando distintas:

```bash
# Solo HIIG_Cepheids -----------------------------------------------------
cobaya-run lik_hiig_cepheids.yaml params_h0_only.yaml     sampler_mcmc.yaml -o chains/cepheids_h0
cobaya-run lik_hiig_cepheids.yaml params_h0_om_wde.yaml   sampler_mcmc.yaml -o chains/cepheids_wcdm
cobaya-run lik_hiig_cepheids.yaml params_h0_om_w0wa.yaml  sampler_mcmc.yaml -o chains/cepheids_cpl

# Solo HIIG_TRGB ----------------------------------------------------------
cobaya-run lik_hiig_trgb.yaml params_h0_only.yaml     sampler_mcmc.yaml -o chains/trgb_h0
cobaya-run lik_hiig_trgb.yaml params_h0_om_wde.yaml   sampler_mcmc.yaml -o chains/trgb_wcdm
cobaya-run lik_hiig_trgb.yaml params_h0_om_w0wa.yaml  sampler_mcmc.yaml -o chains/trgb_cpl

# Solo Planck ---------------------------------------------------------------
cobaya-run lik_planck.yaml params_h0_only.yaml     sampler_mcmc.yaml -o chains/planck_h0
cobaya-run lik_planck.yaml params_h0_om_wde.yaml   sampler_mcmc.yaml -o chains/planck_wcdm
cobaya-run lik_planck.yaml params_h0_om_w0wa.yaml  sampler_mcmc.yaml -o chains/planck_cpl

# Planck + HIIG_Cepheids ------------------------------------------------------
cobaya-run lik_planck.yaml lik_hiig_cepheids.yaml params_h0_only.yaml     sampler_mcmc.yaml -o chains/planck_cepheids_h0
cobaya-run lik_planck.yaml lik_hiig_cepheids.yaml params_h0_om_wde.yaml   sampler_mcmc.yaml -o chains/planck_cepheids_wcdm
cobaya-run lik_planck.yaml lik_hiig_cepheids.yaml params_h0_om_w0wa.yaml  sampler_mcmc.yaml -o chains/planck_cepheids_cpl

# Planck + HIIG_TRGB ----------------------------------------------------------
cobaya-run lik_planck.yaml lik_hiig_trgb.yaml params_h0_only.yaml     sampler_mcmc.yaml -o chains/planck_trgb_h0
cobaya-run lik_planck.yaml lik_hiig_trgb.yaml params_h0_om_wde.yaml   sampler_mcmc.yaml -o chains/planck_trgb_wcdm
cobaya-run lik_planck.yaml lik_hiig_trgb.yaml params_h0_om_w0wa.yaml  sampler_mcmc.yaml -o chains/planck_trgb_cpl
```

Corre `cobaya-run` desde el directorio donde viven `lsigma_hiig_likelihood.py`
y tus CSV (`Compilation2026.csv`, `GEHR_Cepheids.csv`, `GEHR_TRGB.csv`), o
agrega esa ruta a tu `PYTHONPATH`.

## Notas importantes / cosas que debes ajustar

1. **Nombre de columnas de anclas**: en `lik_hiig_trgb.yaml` puse
   `error_kind: sigma_TRGB` como placeholder — cámbialo al nombre real de la
   columna de error en tu tabla `GEHR_TRGB.csv` (igual que `sigma_w` en tu
   script original para Cefeidas).

2. **Instalación de likelihoods de Planck**: necesitas correr
   `cobaya-install planck_2018_highl_plik.TTTEEE_lite_native planck_2018_lowl.TT planck_2018_lowl.EE planck_2018_lensing.clik`
   una vez, antes de usar `lik_planck.yaml`. Si prefieres el likelihood
   "full" de `plik` (con nuisance de foregrounds), cambia esa línea, pero
   entonces deberás añadir sus parámetros de nuisance al bloque `params`.

3. **omega_cdm dinámico**: en las parametrizaciones 2 y 3, `omega_cdm` no se
   muestrea directamente sino que se recalcula en cada paso a partir de
   `Omega_m`, `omega_b` y `H0` (`value: 'lambda ...'`), para que `Omega_m`
   tenga el significado físico que usas en tu tesis. Esto es el equivalente
   Cobaya de fijar `Om` como parámetro libre en `FlatwCDM` dentro de tu
   `lnlike` original.

4. **w0_fld / wa_fld y CLASS**: `Omega_Lambda: 0.0` + `use_ppf: 'yes'` le
   dicen a CLASS que reemplace la constante cosmológica por un fluido con
   ecuación de estado libre (necesario para permitir que `w` cruce -1). Si
   tu instalación de `classy` no trae PPF, tendrás que fijar el prior de
   `w0_fld`/`wa_fld` de modo que nunca crucen -1, o compilar CLASS con esa
   opción.

5. **La likelihood L-sigma solo usa el ángulo `angular_diameter_distance`,
   `Hubble` y `comoving_radial_distance` en el redshift de tus HIIG** (no
   pide nada del espectro de potencias), así que corre igual de rápido
   combinada con Planck que sola — el costo extra al combinarlas viene solo
   de Planck.

6. Los priors numéricos (rangos de `H0`, `Omega_m`, `w0_fld`, `wa_fld`,
   `alpha_ls`, `beta_ls`) son puntos de partida razonables tomados de tu
   `prior_transform` original — ajústalos si tu problema los necesita más
   anchos/angostos.
