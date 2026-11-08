from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd
from cobaya.likelihood import Likelihood
from numpy.polynomial.hermite import hermgauss
from scipy.special import logsumexp


_C_KM_S = 299792.458
_LENS_DM_COEFF = 2.5 / np.log(10.0)


class LSigmaHIIGLensing(Likelihood):
    """Cobaya likelihood for GEHR + HIIG using the L(Hbeta)-sigma relation,
    including the redshift-dependent weak-lensing marginalization used in
    Chavez et al. (2026, arXiv:2607.14254).

    Relation
    --------
        log10 L(Hbeta) = alpha_ls + beta_ls log10(sigma)

    Hence
        mu_Lsigma = 2.5(alpha_ls + beta_ls log_sigma)
                    - 2.5 log_f_Hbeta - mu_constant.

    GEHR anchors are compared to independent host distance moduli. HIIG are
    compared to CLASS luminosity distances through D_L=(1+z)^2 D_A.

    Weak lensing
    ------------
    For each lensed HIIG,
        t = ln(M) ~ Normal(U,S^2),
        sigma_eff(z) = lensing_sigma_coeff * z,
        S^2 = ln(1 + sigma_eff^2),
        U = -S^2/2,
    so that <M>=1. The distance-modulus shift is
        Delta mu_lens = -b t,  b=2.5/ln(10),
    and the residual at fixed t is R + b t.

    The marginalization can be evaluated either exactly (``analytic``) or by
    Gauss-Hermite quadrature (``gh``). The latter is provided as a direct
    validation route against the user's previous MultiNest implementation.
    """

    # Sampled/fixed input parameters are configured in YAML.
    params = {
        "alpha_ls": None,
        "beta_ls": None,
        "sigma_int": None,
    }

    # ------------------------- data configuration -------------------------
    data_file: str = "Compilation2026_main.csv"
    anchor_file: str = "GEHR_base.csv"
    anchor_cov_file: Optional[str] = None

    data_host_column: str = "GEHR_id"
    origin_column: str = "origin_id"
    z_or_mu_column: str = "z_or_mu"
    z_or_mu_error_column: str = "e_z_or_e_mu"
    log_sigma_column: str = "log_sigma"
    e_log_sigma_column: str = "e_log_sigma"
    log_flux_column: str = "log_f_Hbeta"
    e_log_flux_column: str = "e_log_f_Hbeta"

    anchor_host_column: str = "Galaxia"
    anchor_mu_column: str = "mu_w"
    anchor_error_column: str = "sigma_w"

    mu_constant: float = 100.19477738511641
    zmax: Optional[float] = 14.0

    # ---------------------- Gaussian-core configuration -------------------
    # Recommended for inference because beta_ls and sigma_int can change the
    # variance. Set False only when reproducing the older unnormalised chi2.
    include_logdet: bool = True

    # If several GEHR live in one host, their distance-modulus uncertainty is
    # shared. With True this is represented as a host-level covariance block.
    correlate_host_mu: bool = True

    # Optional extra terms in magnitudes.
    sigma_v_pec_kms: float = 0.0
    sigma_sys_mag: float = 0.0
    sigma_sys_column: Optional[str] = None

    # ------------------------- lensing configuration ----------------------
    apply_lensing: bool = True
    lensing_sigma_coeff: float = 0.088
    lensing_zmin: float = 0.0
    lensing_zmax: Optional[float] = 14.0
    lensing_method: str = "analytic"  # "analytic" or "gh"
    gh_order: int = 20

    def initialize(self):
        self._data_path = self._resolve_file(self.data_file)
        self._anchor_path = self._resolve_file(self.anchor_file)
        self._anchor_cov_path = (
            self._resolve_file(self.anchor_cov_file)
            if self.anchor_cov_file not in (None, "", "null")
            else None
        )

        data = pd.read_csv(self._data_path, comment="#", index_col=False)
        anchors = pd.read_csv(self._anchor_path, comment="#", index_col=False)

        required_data = [
            self.data_host_column,
            self.origin_column,
            self.z_or_mu_column,
            self.z_or_mu_error_column,
            self.log_sigma_column,
            self.e_log_sigma_column,
            self.log_flux_column,
            self.e_log_flux_column,
        ]
        if self.sigma_sys_column not in (None, "", "null"):
            required_data.append(str(self.sigma_sys_column))

        required_anchor = [
            self.anchor_host_column,
            self.anchor_mu_column,
            self.anchor_error_column,
        ]
        self._require_columns(data, required_data, self._data_path)
        self._require_columns(anchors, required_anchor, self._anchor_path)

        data = data.copy()
        anchors = anchors.copy()
        data[self.data_host_column] = data[self.data_host_column].astype(str).str.strip()
        anchors[self.anchor_host_column] = anchors[self.anchor_host_column].astype(str).str.strip()

        # One homogenized distance per anchor host is expected.
        dup = anchors[self.anchor_host_column].duplicated(keep=False)
        if dup.any():
            duplicated = sorted(anchors.loc[dup, self.anchor_host_column].unique().tolist())
            raise ValueError(
                "anchor_file contains more than one row for at least one host. "
                "Provide one homogenized mu per host (or use a hierarchical latent-mu model). "
                f"Duplicated hosts: {duplicated[:10]}"
            )

        origin_all = pd.to_numeric(data[self.origin_column], errors="raise").to_numpy(float)
        is_anchor_all = np.isclose(origin_all, 0.0)
        is_hiig_all = ~is_anchor_all

        anchor_hosts = set(anchors[self.anchor_host_column].tolist())
        host_data = data[self.data_host_column].to_numpy(str)

        # Keep all HIIG and only GEHR whose host is present in the anchor table.
        keep = is_hiig_all | (is_anchor_all & np.isin(host_data, list(anchor_hosts)))
        if self.zmax is not None:
            z_or_mu_all = pd.to_numeric(data[self.z_or_mu_column], errors="raise").to_numpy(float)
            keep &= is_anchor_all | (z_or_mu_all <= float(self.zmax))

        data = data.loc[keep].reset_index(drop=True)
        if len(data) == 0:
            raise ValueError("No L-sigma objects remain after anchor selection and zmax cut.")

        origin = pd.to_numeric(data[self.origin_column], errors="raise").to_numpy(float)
        self._is_anchor = np.isclose(origin, 0.0)
        self._is_hiig = ~self._is_anchor
        self._hosts = data[self.data_host_column].astype(str).str.strip().to_numpy()
        self._z_or_mu = pd.to_numeric(data[self.z_or_mu_column], errors="raise").to_numpy(float)
        self._e_z_or_e_mu = pd.to_numeric(
            data[self.z_or_mu_error_column], errors="raise"
        ).to_numpy(float)
        self._log_sigma = pd.to_numeric(data[self.log_sigma_column], errors="raise").to_numpy(float)
        self._e_log_sigma = pd.to_numeric(
            data[self.e_log_sigma_column], errors="raise"
        ).to_numpy(float)
        self._log_flux = pd.to_numeric(data[self.log_flux_column], errors="raise").to_numpy(float)
        self._e_log_flux = pd.to_numeric(
            data[self.e_log_flux_column], errors="raise"
        ).to_numpy(float)

        # Optional per-object systematic magnitude uncertainty.
        if self.sigma_sys_column not in (None, "", "null"):
            self._sigma_sys = pd.to_numeric(
                data[str(self.sigma_sys_column)], errors="raise"
            ).to_numpy(float)
        else:
            self._sigma_sys = np.full(len(data), float(self.sigma_sys_mag), dtype=float)
        if np.any(~np.isfinite(self._sigma_sys)) or np.any(self._sigma_sys < 0):
            raise ValueError("Systematic magnitude uncertainties must be finite and >= 0.")

        # Map host distances into each GEHR row.
        anchor_mu_map = anchors.set_index(self.anchor_host_column)[self.anchor_mu_column].astype(float)
        anchor_err_map = anchors.set_index(self.anchor_host_column)[self.anchor_error_column].astype(float)
        self._anchor_mu = np.full(len(data), np.nan, dtype=float)
        self._anchor_muerr = np.zeros(len(data), dtype=float)
        if np.any(self._is_anchor):
            anchor_host_series = pd.Series(self._hosts[self._is_anchor])
            mu = anchor_host_series.map(anchor_mu_map)
            err = anchor_host_series.map(anchor_err_map)
            if mu.isna().any() or err.isna().any():
                missing = sorted(anchor_host_series[mu.isna() | err.isna()].unique().tolist())
                raise ValueError(f"Missing anchor mu/error for hosts: {missing}")
            self._anchor_mu[self._is_anchor] = mu.to_numpy(float)
            self._anchor_muerr[self._is_anchor] = err.to_numpy(float)

        self._z_hiig = self._z_or_mu[self._is_hiig]
        self._ez_hiig = self._e_z_or_e_mu[self._is_hiig]
        if np.any(~np.isfinite(self._z_hiig)) or np.any(self._z_hiig <= 0):
            raise ValueError("All HIIG redshifts must be finite and > 0.")
        if np.any(~np.isfinite(self._ez_hiig)) or np.any(self._ez_hiig < 0):
            raise ValueError("HIIG redshift uncertainties must be finite and >= 0.")

        self._host_cov = self._load_host_covariance(anchors)

        # Pre-compute redshift-only lensing prior parameters.
        self._prepare_lensing_cache()

        method = str(self.lensing_method).lower().strip()
        if method not in {"analytic", "gh"}:
            raise ValueError("lensing_method must be 'analytic' or 'gh'.")
        self._lensing_method = method
        if int(self.gh_order) < 4:
            raise ValueError("gh_order must be >= 4.")
        if self._lensing_method == "gh":
            self._gh_x, self._gh_w = hermgauss(int(self.gh_order))

        self.log.info(
            "Loaded L-sigma+lensing sample: %d objects = %d GEHR anchors + %d HIIG; "
            "zmax=%s; lensed HIIG=%d/%d; method=%s",
            len(data),
            int(self._is_anchor.sum()),
            int(self._is_hiig.sum()),
            str(self.zmax),
            int(self._lens_mask_hiig.sum()),
            int(self._is_hiig.sum()),
            self._lensing_method,
        )

    def get_requirements(self):
        """Ask CLASS only for the background quantities needed by the HIIG block."""
        if len(self._z_hiig) == 0:
            return {}
        z = self._z_hiig.tolist()
        return {
            "angular_diameter_distance": {"z": z},
            "Hubble": {"z": z},
        }

    def logp(self, alpha_ls, beta_ls, sigma_int, **params_values):
        alpha_ls = float(alpha_ls)
        beta_ls = float(beta_ls)
        sigma_int = float(sigma_int)
        if sigma_int < 0:
            return -np.inf

        # ------------------------------------------------------------------
        # 1) L-sigma distance estimator and Gaussian-core variance.
        # ------------------------------------------------------------------
        mu_lsigma = (
            2.5 * (beta_ls * self._log_sigma + alpha_ls)
            - 2.5 * self._log_flux
            - float(self.mu_constant)
        )

        # sigma_int is in dex(log L), hence 2.5*sigma_int in magnitudes.
        var_lsigma = (2.5 ** 2) * (
            self._e_log_flux ** 2 + (beta_ls ** 2) * self._e_log_sigma ** 2
        ) + (2.5 * sigma_int) ** 2
        var_lsigma += self._sigma_sys ** 2

        if np.any(~np.isfinite(var_lsigma)) or np.any(var_lsigma <= 0):
            return -np.inf

        # ------------------------------------------------------------------
        # 2) GEHR anchor likelihood. Lensing is NEVER applied to anchors.
        # ------------------------------------------------------------------
        loglike_anchor = self._anchor_loglike(mu_lsigma, var_lsigma)
        if not np.isfinite(loglike_anchor):
            return -np.inf

        # No cosmological HIIG rows: return anchor contribution only.
        if not np.any(self._is_hiig):
            return float(loglike_anchor)

        # ------------------------------------------------------------------
        # 3) Cosmological distance modulus for HIIG from CLASS.
        # ------------------------------------------------------------------
        z = self._z_hiig
        da = np.asarray(self.provider.get_angular_diameter_distance(z), dtype=float)
        hubble = np.asarray(
            self.provider.get_Hubble(z, units="km/s/Mpc"), dtype=float
        )
        if da.shape != z.shape or hubble.shape != z.shape:
            return -np.inf

        dl = (1.0 + z) ** 2 * da
        if np.any(~np.isfinite(dl)) or np.any(dl <= 0):
            return -np.inf
        mu_cosmo = 5.0 * np.log10(dl) + 25.0

        # Exact flat-FLRW redshift-error propagation:
        # D_C=(1+z)D_A,
        # dmu/dz = 5/ln10 [1/(1+z) + c/(H(z) D_C)].
        dc = (1.0 + z) * da
        if np.any(dc <= 0) or np.any(hubble <= 0):
            return -np.inf
        dmu_dz = (5.0 / np.log(10.0)) * (
            1.0 / (1.0 + z) + _C_KM_S / (hubble * dc)
        )

        sigma_z = self._ez_hiig.copy()
        if float(self.sigma_v_pec_kms) > 0:
            sigma_z_pec = (1.0 + z) * float(self.sigma_v_pec_kms) / _C_KM_S
            sigma_z = np.sqrt(sigma_z ** 2 + sigma_z_pec ** 2)
        var_z = (dmu_dz * sigma_z) ** 2

        mu_obs_hiig = mu_lsigma[self._is_hiig]
        var_hiig = var_lsigma[self._is_hiig] + var_z
        residual_hiig = mu_obs_hiig - mu_cosmo

        if np.any(~np.isfinite(var_hiig)) or np.any(var_hiig <= 0):
            return -np.inf

        # ------------------------------------------------------------------
        # 4) HIIG likelihood, with latent magnification marginalized out.
        # ------------------------------------------------------------------
        lens_mask = self._lens_mask_hiig
        loglike_hiig = 0.0

        # Objects outside the lensing mask use the ordinary Gaussian core.
        if np.any(~lens_mask):
            loglike_hiig += self._diag_gaussian_loglike(
                residual_hiig[~lens_mask], var_hiig[~lens_mask]
            )

        if np.any(lens_mask):
            R = residual_hiig[lens_mask]
            V = var_hiig[lens_mask]
            U = self._lens_U_hiig[lens_mask]
            S = self._lens_S_hiig[lens_mask]

            if self._lensing_method == "analytic":
                loglike_hiig += self._lensing_loglike_analytic(R, V, U, S)
            else:
                loglike_hiig += self._lensing_loglike_gh(R, V, U, S)

        if not np.isfinite(loglike_hiig):
            return -np.inf
        return float(loglike_anchor + loglike_hiig)

    # ======================================================================
    # LENSING
    # ======================================================================

    def _prepare_lensing_cache(self):
        """Pre-compute U(z), S(z) for HIIG; anchors are excluded by design."""
        n = len(self._z_hiig)
        self._lens_mask_hiig = np.zeros(n, dtype=bool)
        self._lens_U_hiig = np.zeros(n, dtype=float)
        self._lens_S_hiig = np.zeros(n, dtype=float)

        if not bool(self.apply_lensing) or n == 0:
            return

        z = self._z_hiig
        mask = np.isfinite(z) & (z > float(self.lensing_zmin))
        if self.lensing_zmax is not None:
            mask &= z <= float(self.lensing_zmax)
        self._lens_mask_hiig = mask

        if not np.any(mask):
            return

        sigma_eff = np.clip(float(self.lensing_sigma_coeff) * z[mask], 0.0, None)
        S2 = np.log1p(sigma_eff ** 2)
        self._lens_S_hiig[mask] = np.sqrt(S2)
        self._lens_U_hiig[mask] = -0.5 * S2

    def _lensing_loglike_analytic(self, residual, variance, U, S):
        """Exact marginalization over t=ln(M).

        Because t is Gaussian and the residual is linear in t, the integral
        written in Eq. (23) of Chavez et al. (2026) is a Gaussian convolution.

        Normalized form:
            V_eff = variance + b^2 S^2
            R_eff = residual + b U
            ln L = -1/2 [R_eff^2/V_eff + ln(2*pi*V_eff)].

        If include_logdet=False, the normalization is chosen so that S->0
        reproduces exactly the old unnormalized -chi2/2 convention.
        """
        V0 = np.asarray(variance, dtype=float)
        R = np.asarray(residual, dtype=float)
        U = np.asarray(U, dtype=float)
        S = np.asarray(S, dtype=float)

        Veff = V0 + (_LENS_DM_COEFF * S) ** 2
        Reff = R + _LENS_DM_COEFF * U
        if np.any(Veff <= 0) or np.any(~np.isfinite(Veff)):
            return -np.inf

        if bool(self.include_logdet):
            return float(-0.5 * np.sum(Reff ** 2 / Veff + np.log(2.0 * np.pi * Veff)))

        # Legacy unnormalised core: integral of exp[-(R+b t)^2/(2 V0)]
        # over the normalized lens prior.
        return float(
            np.sum(-0.5 * Reff ** 2 / Veff + 0.5 * np.log(V0 / Veff))
        )

    def _lensing_loglike_gh(self, residual, variance, U, S):
        """Gauss-Hermite evaluation matching the previous MultiNest approach."""
        R = np.asarray(residual, dtype=float)
        sigma = np.sqrt(np.asarray(variance, dtype=float))
        U = np.asarray(U, dtype=float)
        S = np.asarray(S, dtype=float)
        sigma = np.clip(sigma, 1.0e-15, None)

        t = U[:, None] + np.sqrt(2.0) * S[:, None] * self._gh_x[None, :]
        arg = (R[:, None] + _LENS_DM_COEFF * t) / sigma[:, None]
        log_terms = np.log(self._gh_w)[None, :] - 0.5 * arg ** 2
        log_expectation = -0.5 * np.log(np.pi) + logsumexp(log_terms, axis=1)

        if bool(self.include_logdet):
            # Gaussian-core normalization. The previous MultiNest code omitted
            # the additive constant -0.5 ln(2*pi); keeping it here makes the
            # likelihood fully normalized.
            log_expectation -= np.log(sigma) + 0.5 * np.log(2.0 * np.pi)

        return float(np.sum(log_expectation))

    # ======================================================================
    # ANCHORS AND ORDINARY GAUSSIAN PIECES
    # ======================================================================

    def _anchor_loglike(self, mu_lsigma, var_lsigma):
        if not np.any(self._is_anchor):
            return 0.0

        idx = np.where(self._is_anchor)[0]
        residual = mu_lsigma[idx] - self._anchor_mu[idx]
        base_var = var_lsigma[idx]

        if bool(self.correlate_host_mu):
            anchor_hosts = self._hosts[idx]
            unique_hosts = list(dict.fromkeys(anchor_hosts.tolist()))
            B = np.zeros((len(idx), len(unique_hosts)), dtype=float)
            host_to_col = {h: j for j, h in enumerate(unique_hosts)}
            for r, h in enumerate(anchor_hosts):
                B[r, host_to_col[h]] = 1.0

            if self._host_cov is not None:
                host_cov = self._host_cov.loc[unique_hosts, unique_hosts].to_numpy(float)
            else:
                host_sigmas = np.array([
                    self._anchor_muerr[idx[np.where(anchor_hosts == h)[0][0]]]
                    for h in unique_hosts
                ])
                host_cov = np.diag(host_sigmas ** 2)

            cov = np.diag(base_var) + B @ host_cov @ B.T
            try:
                L = np.linalg.cholesky(cov)
                y = np.linalg.solve(L, residual)
            except np.linalg.LinAlgError:
                return -np.inf

            chi2 = float(y @ y)
            if bool(self.include_logdet):
                logdet = 2.0 * float(np.sum(np.log(np.diag(L))))
                return -0.5 * (chi2 + logdet + len(residual) * np.log(2.0 * np.pi))
            return -0.5 * chi2

        # Legacy independent-row treatment of host distance errors.
        variance = base_var + self._anchor_muerr[idx] ** 2
        return self._diag_gaussian_loglike(residual, variance)

    def _diag_gaussian_loglike(self, residual, variance):
        R = np.asarray(residual, dtype=float)
        V = np.asarray(variance, dtype=float)
        if np.any(V <= 0) or np.any(~np.isfinite(V)):
            return -np.inf
        if bool(self.include_logdet):
            return float(-0.5 * np.sum(R ** 2 / V + np.log(2.0 * np.pi * V)))
        return float(-0.5 * np.sum(R ** 2 / V))

    # ======================================================================
    # I/O HELPERS
    # ======================================================================

    def _load_host_covariance(self, anchors: pd.DataFrame):
        if self._anchor_cov_path is None:
            return None

        cov = pd.read_csv(self._anchor_cov_path, index_col=0)
        cov.index = cov.index.astype(str).str.strip()
        cov.columns = cov.columns.astype(str).str.strip()
        hosts = anchors[self.anchor_host_column].astype(str).str.strip().tolist()
        missing_rows = sorted(set(hosts) - set(cov.index))
        missing_cols = sorted(set(hosts) - set(cov.columns))
        if missing_rows or missing_cols:
            raise ValueError(
                "anchor_cov_file does not cover all hosts. "
                f"Missing rows={missing_rows}, missing cols={missing_cols}"
            )
        sub = cov.loc[hosts, hosts].astype(float)
        arr = sub.to_numpy(float)
        if not np.allclose(arr, arr.T, rtol=1e-10, atol=1e-12):
            raise ValueError("anchor_cov_file must be symmetric.")
        return sub

    @staticmethod
    def _require_columns(df: pd.DataFrame, columns, path: str):
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in {path}: {missing}")

    @staticmethod
    def _resolve_file(path: str) -> str:
        path = os.path.expanduser(str(path))
        if os.path.isabs(path) and os.path.exists(path):
            return path
        candidates = [
            os.path.abspath(path),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), path),
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
        raise FileNotFoundError(
            f"Could not find '{path}'. Tried: {candidates}. "
            "Use an absolute path in the Cobaya YAML if needed."
        )
