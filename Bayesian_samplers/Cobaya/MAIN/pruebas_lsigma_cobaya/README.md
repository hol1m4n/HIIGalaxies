# Reproducción y auditoría de L–sigma en Cobaya

## Qué se comprobó

Se revisaron los siete `Lensing_Model*.py` y el likelihood adjunto
`lsigma_hiig_lensing_likelihood(2).py`. Las comprobaciones numéricas usan una
muestra SINTÉTICA de 11 objetos (3 GEHR y 8 HIIG, z=0.01...8), no tus CSV reales.
No se ejecutaron cadenas de producción ni se validaron los resultados científicos
de tu muestra. Versiones utilizadas: Cobaya 3.6.2, Astropy 8.0.3, CLASS 3.4.3.

- Con idénticos fondos Astropy, los likelihoods originales planos (1,2,4,6)
  coinciden a aproximadamente 1e-13 en lnL en los puntos examinados.
- Los originales curvos (3,5,7) difieren: MultiNest usa D_C radial y Cobaya usa
  D_M transversal en una expresión que solamente es correcta en geometría plana.
  Al anular e_z, la diferencia desaparece a precisión numérica.
- `_diag_gaussian_loglike` en el archivo original omite el determinante también
  cuando `include_logdet=True`. La rama de lensing sí lo incluye: es inconsistente.
- Modelo 1, a través de Cobaya+CLASS con el fondo igualado: diferencias de lnL
  menores de 3e-6 respecto a Astropy en los tres puntos ensayados.
- El módulo corregido en modo `legacy_multinest` reproduce los siete originales
  a ~1e-13 en la comparación directa, y a ~2e-14 en los puntos ensayados a través
  del pipeline Cobaya+Astropy.
- La propagación corregida de curvatura concuerda con una derivada numérica
  independiente del módulo de distancia: diferencia en lnL menor de 2e-11 en
  los puntos sintéticos probados.

Los JSON incluidos contienen los números, parámetros y discrepancias. No son
estimaciones de incertidumbre ni un análisis de posterior.

## Contenido

- `lsigma_hiig_lensing_audited.py`: copia alternativa del likelihood, con la
  normalización diagonal corregida, propagación general FLRW y un modo legado.
- `astropy_background.py`: proveedor de distancias/H(z) para aislar el cambio de
  muestreador manteniendo el mismo calculador cosmológico que tus scripts.
- `replica_astropy_model1.yaml` ... `replica_astropy_model7.yaml`: priors y fondos
  correspondientes a tus siete modelos; usan explícitamente el modo legado.
- `reproducir_modelo1.yaml`: prueba CLASS para el likelihood ORIGINAL, con priors
  y fondo del Modelo 1 igualados. Asume el nombre de módulo habitual sin `(2)`.
- `reproducir_modelo1_corregido_CLASS.yaml`: variante CLASS con el módulo corregido.
- `validar_lsigma.py`: evalúa varios puntos sin importar ni ejecutar el sampler
  PyMultiNest. Extrae únicamente los métodos necesarios de los originales por AST.
- `covmat_desde_multinest.py`: convierte muestras de igual peso de MultiNest a una
  covarianza de propuesta para Cobaya; convierte h a H0.

## Cómo empezar

1. Descomprime estos archivos en una carpeta de pruebas. Coloca allí los CSV y
   los siete scripts originales, o usa rutas absolutas para los datos.
2. Ajusta `anchor_host_column`, `anchor_mu_column`, `anchor_error_column` en los YAML.
   Usé `Galaxia`, `mu_w`, `sigma_w`, como en tus scripts MultiNest. El módulo original
   de Cobaya espera por defecto `galaxy`, `mu_W`, `sigma_W`. Son sensibles a mayúsculas.
3. No ejecutes los originales mediante import normal: lanzan muestreos al importarse.
   El validador evita esas llamadas de nivel superior.
4. Ejecuta el validador en tu entorno donde estén instalados Astropy y Cobaya:

```bash
python validar_lsigma.py \
  --models . --likelihood 'lsigma_hiig_lensing_likelihood(2).py' \
  --data Compilation2026_main.csv --anchors GEHR_base.csv \
  --anchor-host Galaxia --anchor-mu mu_w --anchor-error sigma_w \
  --output auditoria_datos_reales.json
```

Para repetir con las correcciones, cambia el argumento del likelihood y añade:

```text
--likelihood lsigma_hiig_lensing_audited.py --redshift-error-method curved
```

Para reproducir exactamente la fórmula antigua, incluso en los modelos curvos:

```text
--likelihood lsigma_hiig_lensing_audited.py --redshift-error-method legacy_multinest
```

`--synthetic` sustituye ambos CSV por la muestra sintética. `--class-check` añade
la comparación completa Cobaya+CLASS del Modelo 1. Si CLASS no está instalado
globalmente, usa `--class-path /ruta/a/class` con la ruta que acepte tu instalación
de Cobaya. Esta opción CLASS prueba solo el Modelo 1, no los otros seis.

Después de comprobar los puntos con datos reales:

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
mpirun -np 4 cobaya-run replica_astropy_model1.yaml
```

Repite con el YAML del modelo deseado. Las siete configuraciones Astropy se
validaron mediante `get_model` y evaluaciones de likelihood, no mediante corridas
MCMC completas. Sus referencias/propuestas son puntos de partida conservadores.
Usa un prefijo de salida nuevo para cada cambio de modelo, datos o likelihood.

## Parámetros y modelos

En todos: alpha=[32.5,34.5], beta=[4.5,5.5], H0=[50,100]. Cada prior es uniforme.
La conversión lineal H0=100h conserva la forma uniforme del prior original.

| Modelo | Fondo Astropy | Otros parámetros muestreados |
|---|---|---|
| 1 | FlatLambdaCDM | ninguno; Om0=0.3 |
| 2 | FlatLambdaCDM | Om0=[0,1] |
| 3 | LambdaCDM | Om0=[0,1], Ode0=[0,1] |
| 4 | FlatwCDM | Om0=[0,1], w0=[-2,0] |
| 5 | wCDM | Om0=[0,1], Ode0=[0,1], w0=[-2,0] |
| 6 | Flatw0waCDM | Om0=[0,1], w0=[-2,0], wa=[-2,2] |
| 7 | w0waCDM | Om0=[0,1], Ode0=[0,1], w0=[-2,0], wa=[-2,2] |

En 3,5,7, Omega_k=1-Om0-Ode0-Omega_r(H0). El prior rectangular sobre Om0/Ode0
no es un prior rectangular independiente sobre Om0/Omega_k. No cambiar esa
parametrización durante una prueba de reproducción.

Los fondos originales usan Tcmb0=2.7255 K, Neff=3.04 y neutrinos sin masa.
No son idénticos al antiguo YAML CLASS con un neutrino de 0.06 eV y densidades
físicas omega_b/omega_cdm fijas. Fijar omega_m mantiene Omega_m*h^2, no Omega_m.

## Qué significa la corrección

El módulo alternativo tiene `redshift_error_method`:

- `curved` (por defecto): usa la derivada correcta en FLRW, también en el caso plano.
- `legacy_multinest`: reproduce la fórmula anterior con D_C radial. Es útil para
  identificar diferencias y repetir los resultados viejos; conserva deliberadamente
  el error de propagación de la curvatura y no debe confundirse con su corrección.

Para CLASS con `curved`, declara `omegak` dentro de `params` (fijo o dependiente/
muestreado). No lo dejes solamente como `Omega_k` en `extra_args`: el likelihood
necesita poder obtenerlo del provider. El YAML corregido del Modelo 1 ya lo hace.

Con D_H=c/H0 y chi=D_C/D_H:

    dmu/dz = 5/ln(10) * [1/(1+z) + c*C_k(chi)/(H(z)*D_M)]

C_k=cosh(sqrt(Omega_k)*chi) para Omega_k>0; C_k=1 para cero;
C_k=cos(sqrt(-Omega_k)*chi) para Omega_k<0. D_M=(1+z)D_A.
El módulo conserva el signo de cos para la geometría cerrada.

También se corrigió `_diag_gaussian_loglike` para que `include_logdet=True`
incluya -0.5*sum(log(2*pi*V)); con False conserva el comportamiento antiguo.
La máscara de lensing ahora usa z<lensing_zmax, como los originales.
No se modificaron los archivos adjuntos originales.

Para reproducción, mantener:

```yaml
include_logdet: false
correlate_host_mu: false
sigma_v_pec_kms: 0.0
sigma_sys_mag: 0.0
lensing_method: gh
gh_order: 20
zmax: null
lensing_zmax: 20.0
# sigma_int: 0.0 dentro de params
```

Los scripts originales no cortan la muestra en z=20: solo dejan de aplicar
lensing allí. El validador rechaza objetos exactamente en z=20 para que la
diferencia original entre `<` y `<=` no contamine la prueba.

Normalizar, correlacionar GEHR del mismo host o liberar sigma_int son cambios
del modelo estadístico. Háganse después, como experimentos separados. La varianza
depende de beta, así que omitir log(V) no equivale a omitir una constante. No usar
la evidencia del likelihood no normalizado como evidencia de un modelo generativo
gaussiano normalizado. El MCMC ordinario de Cobaya no estima la evidencia.

## Covarianza a partir de tus resultados

```bash
python covmat_desde_multinest.py Lens_FlatLCDM_post_equal_weights.dat \
  --names alpha_ls beta_ls H0 --output model1.covmat
```

Para Modelo 7 con estos YAML, usar `--names alpha_ls beta_ls H0 Om0 Ode0 w0 wa`.
Solo sirve para `post_equal_weights.dat`; no para el archivo de pesos desiguales.
Luego `covmat: model1.covmat`, `learn_proposal: true`, y ensayar
`proposal_scale: 1.9`. La covarianza debe corresponder al mismo problema.
Los `proposal` individuales no reemplazan las entradas presentes en la covarianza.

Al comparar posteriores: convertir h a H0, usar los pesos de Cobaya (no tomar
la media simple de sus filas), retirar el transitorio de cada cadena, comprobar
trazas, R-1, ESS y errores Monte Carlo. Usar distintos puntos iniciales. La
convergencia no se decide por la aceptación ni por el número bruto de filas.

## Antes de combinar con CMB/BAO

AstropyBackground es exclusivamente un puente para validar HIIG. Reemplazarlo
por CLASS/CAMB para las corridas conjuntas; no activar dos proveedores de las
mismas distancias. El módulo HIIG pide observables al provider y no necesita
codificar otra cosmología internamente.

Al usar CMB/BAO, liberar las densidades físicas y parámetros pertinentes de ese
análisis; el reparto fijo Omega_b/Omega_cdm del ejemplo CLASS es solo para igualar
el fondo del Modelo 1. En wCDM/CPL, especificar la energía oscura de CLASS como
fluido (Omega_Lambda=0, w0_fld, wa_fld; Omega_fld determinada por cierre) y revisar
las condiciones de perturbaciones/PPF y la región de prior admitida por CLASS.
La reproducción de distancias con Astropy no valida perturbaciones ni r_drag.

El lensing de HIIG es una marginalización sobre magnificación de fuentes. No es
la likelihood de lensing del CMB. En el código su amplitud depende solo de z y
un coeficiente fijo: no incorpora respuesta a As/ns/crecimiento. La validez física
de esa receta y las posibles covarianzas entre datos requieren evaluación aparte.
La suma de log-likelihoods supone independencia condicional de los conjuntos.

## Referencias de implementación

- https://cobaya.readthedocs.io/en/stable/sampler_mcmc.html
- https://cobaya.readthedocs.io/en/stable/params_prior.html
- https://cobaya.readthedocs.io/en/stable/theories_and_dependencies.html
- https://cobaya.readthedocs.io/en/stable/theory_class.html
- https://docs.astropy.org/en/stable/api/astropy.cosmology.FlatLambdaCDM.html
- https://ned.ipac.caltech.edu/level5/Hogg/Hogg5.html
