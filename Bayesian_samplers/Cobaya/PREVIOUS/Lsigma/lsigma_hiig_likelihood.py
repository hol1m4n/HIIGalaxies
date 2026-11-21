from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd
from cobaya.likelihood import Likelihood


_C_KM_S = 299792.458


class LSigmaHIIG(Likelihood):
    """Cobaya likelihood for the HII-galaxy/GEHR L(Hbeta)-sigma relation.

    Relation used
    -------------
    log10 L(Hbeta) = alpha_ls + beta_ls * log10(sigma)

    which implies

    mu_Lsigma = 2.5 * (alpha_ls + beta_ls*log_sigma)
                - 2.5*log_f_Hbeta - mu_constant.

    For GEHR anchors, mu_model is read from ``anchor_file``.
    For HIIGs, mu_model is computed from CLASS through D_A(z), using
    D_L=(1+z)^2 D_A.

    ``sigma_int`` is an optional intrinsic scatter in dex in log10 L.
    Set sigma_int=0 to reproduce the no-intrinsic-scatter setup.
    """

    # Likelihood input parameters. Priors/fixed values are set in the run YAML.
    params = {
        "alpha_ls": None,
        "beta_ls": None,
        "sigma_int": None,
    }

    # User-configurable options (overridable from the run YAML).
    data_file: str = "Compilation2026.csv"
    anchor_file: str = "GEHR_Cepheids.csv"
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

    # Same constant as in the user's MultiNest implementation.
    mu_constant: float = 100.19477738511641

    zmax: Optional[float] = 7.0
    include_logdet: bool = True
    correlate_host_mu: bool = True
    sigma_v_pec_kms: float = 0.0

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

        # The anchor table should already contain one homogenized distance per host.
        dup = anchors[self.anchor_host_column].duplicated(keep=False)
        if dup.any():
            duplicated = sorted(anchors.loc[dup, self.anchor_host_column].unique().tolist())
            raise ValueError(
                "anchor_file contains more than one row for at least one host. "
                "Provide one homogenized mu per host (or extend the likelihood to a latent-mu model). "
                f"Duplicated hosts: {duplicated[:10]}"
            )

        origin = pd.to_numeric(data[self.origin_column], errors="raise").to_numpy(float)
        is_anchor_all = np.isclose(origin, 0.0)
        is_hiig_all = ~is_anchor_all

        anchor_hosts = set(anchors[self.anchor_host_column].tolist())
        host_data = data[self.data_host_column].to_numpy(str)

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
        if np.any(self._z_hiig <= 0):
            raise ValueError("All HIIG redshifts must be > 0.")
        if np.any(self._ez_hiig < 0):
            raise ValueError("HIIG redshift uncertainties must be >= 0.")

        self._host_cov = self._load_host_covariance(anchors)

        self.log.info(
            "Loaded L-sigma sample: %d objects = %d GEHR anchors + %d HIIG; zmax=%s",
            len(data), int(self._is_anchor.sum()), int(self._is_hiig.sum()), str(self.zmax)
        )

    def get_requirements(self):
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

        # L-sigma distance modulus, identical algebra to the supplied sampler.
        mu_lsigma = (
            2.5 * (beta_ls * self._log_sigma + alpha_ls)
            - 2.5 * self._log_flux
            - float(self.mu_constant)
        )

        # Measurement variance. sigma_int is in dex(log L), hence 2.5*sigma_int in mag.
        var_lsigma = (2.5 ** 2) * (
            self._e_log_flux ** 2 + (beta_ls ** 2) * self._e_log_sigma ** 2
        ) + (2.5 * sigma_int) ** 2

        mu_model = np.empty_like(mu_lsigma)
        extra_diag_var = np.zeros_like(mu_lsigma)

        # GEHR anchors: independent external distance modulus.
        if np.any(self._is_anchor):
            mu_model[self._is_anchor] = self._anchor_mu[self._is_anchor]

        # HIIG: CLASS background distance.
        if np.any(self._is_hiig):
            z = self._z_hiig
            da = np.asarray(self.provider.get_angular_diameter_distance(z), dtype=float)
            hubble = np.asarray(self.provider.get_Hubble(z, units="km/s/Mpc"), dtype=float)

            dl = (1.0 + z) ** 2 * da
            if np.any(~np.isfinite(dl)) or np.any(dl <= 0):
                return -np.inf

            mu_z = 5.0 * np.log10(dl) + 25.0
            mu_model[self._is_hiig] = mu_z

            # For a flat FLRW background:
            # D_C=(1+z)D_A and dmu/dz = 5/ln10 [1/(1+z) + c/(H D_C)].
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

            extra_diag_var[self._is_hiig] = (dmu_dz * sigma_z) ** 2

        residual = mu_lsigma - mu_model
        diag_var = var_lsigma + extra_diag_var

        if np.any(~np.isfinite(diag_var)) or np.any(diag_var <= 0):
            return -np.inf

        if bool(self.correlate_host_mu) and np.any(self._is_anchor):
            cov = np.diag(diag_var)
            anchor_indices = np.where(self._is_anchor)[0]
            anchor_hosts = self._hosts[self._is_anchor]

            # B maps host-level distance covariance into the region-level covariance.
            unique_hosts = list(dict.fromkeys(anchor_hosts.tolist()))
            B = np.zeros((len(anchor_indices), len(unique_hosts)), dtype=float)
            host_to_col = {h: j for j, h in enumerate(unique_hosts)}
            for r, h in enumerate(anchor_hosts):
                B[r, host_to_col[h]] = 1.0

            if self._host_cov is not None:
                host_cov = self._host_cov.loc[unique_hosts, unique_hosts].to_numpy(float)
            else:
                host_sigmas = np.array([
                    self._anchor_muerr[anchor_indices[np.where(anchor_hosts == h)[0][0]]]
                    for h in unique_hosts
                ])
                host_cov = np.diag(host_sigmas ** 2)

            cov_anchor = B @ host_cov @ B.T
            cov[np.ix_(anchor_indices, anchor_indices)] += cov_anchor

            try:
                sign, logdet = np.linalg.slogdet(cov)
                if sign <= 0 or not np.isfinite(logdet):
                    return -np.inf
                solved = np.linalg.solve(cov, residual)
            except np.linalg.LinAlgError:
                return -np.inf

            chi2 = float(residual @ solved)
            if bool(self.include_logdet):
                return -0.5 * (chi2 + logdet + len(residual) * np.log(2.0 * np.pi))
            return -0.5 * chi2

        # Legacy/diagonal treatment: each GEHR row receives the host mu uncertainty independently.
        diag_var = diag_var.copy()
        if np.any(self._is_anchor):
            diag_var[self._is_anchor] += self._anchor_muerr[self._is_anchor] ** 2

        chi2 = float(np.sum(residual ** 2 / diag_var))
        if bool(self.include_logdet):
            return -0.5 * np.sum(
                residual ** 2 / diag_var + np.log(2.0 * np.pi * diag_var)
            )
        return -0.5 * chi2

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
