"""
Cobaya likelihood: L-sigma relation for HII galaxies (HIIG), anchored with
independent distance moduli for a subsample of Giant Extragalactic HII
Regions (GEHR).

This is a direct port of the physical model in `five_par_nested_sampling.py`
(the PyMultiNest / nested-sampling version) into Cobaya's Likelihood API, so
that the cosmology (H0, Omega_m, w0, wa, ...) is supplied by a Cobaya theory
code (CLASS/classy) instead of being sampled by hand inside the likelihood.

The class is generic with respect to the *anchor* catalogue: point
`group_file` / `error_kind` at a Cepheid-calibrated or TRGB-calibrated GEHR
table to get the "HIIG_Cepheids" or "HIIG_TRGB" likelihood respectively.

Free parameters of THIS likelihood (must be declared in the Cobaya `params`
block, see lik_hiig_cepheids.yaml / lik_hiig_trgb.yaml):
    alpha_ls : zero point of the L-sigma relation
    beta_ls  : slope of the L-sigma relation

Cosmological parameters are requested from the theory code via
`get_requirements` and are NOT parameters of this likelihood.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from astropy.table import Table
from astropy import constants as const
import astropy.units as u

from cobaya.likelihood import Likelihood

C_KM_S = const.c.to(u.km / u.s).value


class LSigmaHIIG(Likelihood):

    # ---- options, settable per-instance from the yaml file ----
    data_file: str = "Compilation2026.csv"
    group_file: str = "GEHR_base.csv"
    error_kind: str = "sigma_w"
    zmax: float = 7.0
    hbeta_zero_point: float = 100.19477738511641

    def initialize(self):
        df = pd.read_csv(self.data_file, comment="#", index_col=False,
                          dtype={"GEHR_id": str})
        tab = Table.from_pandas(df)
        tab = self._select_redshift_cut(tab, self.zmax)
        tab = self._group_reader(tab, self.group_file, self.error_kind)

        origin = np.asarray(tab["origin_id"], dtype=float)
        self.is_anchor = origin == 0.0
        self.is_hiig = ~self.is_anchor

        self.log_sigma = np.asarray(tab["log_sigma"], dtype=float)
        self.e_log_sigma = np.asarray(tab["e_log_sigma"], dtype=float)
        self.log_f_hbeta = np.asarray(tab["log_f_Hbeta"], dtype=float)
        self.e_log_f_hbeta = np.asarray(tab["e_log_f_Hbeta"], dtype=float)

        z_or_mu = np.asarray(tab["z_or_mu"], dtype=float)
        e_z_or_e_mu = np.asarray(tab["e_z_or_e_mu"], dtype=float)

        # anchors: z_or_mu / e_z_or_e_mu already hold an external distance
        # modulus (Cepheid- or TRGB-based) and its error -> no cosmology
        self.mu_anchor = z_or_mu[self.is_anchor]
        self.mu_anchor_err = e_z_or_e_mu[self.is_anchor]

        # HIIG: z_or_mu / e_z_or_e_mu hold redshift and its error
        self.z_hiig = z_or_mu[self.is_hiig]
        self.z_hiig_err = e_z_or_e_mu[self.is_hiig]

        self.log.info(
            "L-sigma likelihood '%s': %d anchors, %d HIIG (z<=%.2f)",
            self.group_file, self.mu_anchor.size, self.z_hiig.size, self.zmax,
        )

    # ---------------------------------------------------------------
    # data handling (ported verbatim in spirit from
    # five_par_nested_sampling.py: select_redshift_cut / group_reader)
    # ---------------------------------------------------------------
    @staticmethod
    def _select_redshift_cut(tab, zmax):
        origin = np.asarray(tab["origin_id"], dtype=float)
        if zmax is None:
            mask = np.ones(len(tab), dtype=bool)
        else:
            z_or_mu = np.asarray(tab["z_or_mu"], dtype=float)
            mask = (origin == 0.0) | ((origin != 0.0) & (z_or_mu <= zmax))
        return tab[mask]

    @staticmethod
    def _group_reader(tab, group_file, error_kind):
        df_pd = tab.to_pandas()
        df_group = pd.read_csv(group_file)
        galaxies_in_group = df_group["Galaxia"].unique()
        filtro = df_pd[
            df_pd["GEHR_id"].isin(galaxies_in_group) | (df_pd["origin_id"] > 0.0)
        ].copy()
        filtro = filtro.merge(
            df_group[["Galaxia", "mu_w", error_kind]],
            left_on="GEHR_id", right_on="Galaxia", how="left",
        )
        coincide = filtro["mu_w"].notna()
        filtro.loc[coincide, "z_or_mu"] = filtro.loc[coincide, "mu_w"]
        filtro.loc[coincide, "e_z_or_e_mu"] = filtro.loc[coincide, error_kind]
        filtro = filtro.drop(columns=["Galaxia", "mu_w", error_kind])
        return Table.from_pandas(filtro)

    # ---------------------------------------------------------------
    # Cobaya interface
    # ---------------------------------------------------------------
    def get_requirements(self):
        # Everything needed to build mu(z) and its cosmology-dependent
        # error term for the non-anchor (HIIG) subsample. This plays the
        # role of the "High" branch of D_L / dmu_dz in the original code.
        return {
            "H0": None,
            "angular_diameter_distance": {"z": self.z_hiig},
            "Hubble": {"z": self.z_hiig},
            "comoving_radial_distance": {"z": self.z_hiig},
        }

    def logp(self, **params_values):
        alpha = params_values["alpha_ls"]
        beta = params_values["beta_ls"]

        # --- observed side of the L-sigma relation ---
        Mu = 2.5 * (beta * self.log_sigma + alpha) - 2.5 * self.log_f_hbeta \
            - self.hbeta_zero_point
        MuErr2 = 2.5 ** 2 * (
            self.e_log_f_hbeta ** 2 + beta ** 2 * self.e_log_sigma ** 2
        )

        # --- "true" distance modulus side ---
        Mum = np.empty_like(Mu)
        MumErr2 = np.empty_like(Mu)

        # anchors: external, cosmology-independent distance moduli
        Mum[self.is_anchor] = self.mu_anchor
        MumErr2[self.is_anchor] = self.mu_anchor_err ** 2

        # HIIG: theory-side luminosity distance from CLASS (via Cobaya)
        if self.z_hiig.size:
            H0 = self.provider.get_param("H0")
            d_A = self.provider.get_angular_diameter_distance(self.z_hiig)
            d_L = d_A * (1.0 + self.z_hiig) ** 2  # Mpc

            Hz = self.provider.get_Hubble(self.z_hiig)  # km/s/Mpc
            Ez = Hz / H0
            Dc = self.provider.get_comoving_radial_distance(self.z_hiig)  # Mpc
            Iz = Dc * H0 / C_KM_S

            mu_hiig = 5.0 * np.log10(d_L) + 25.0
            dmu_dz = np.abs(
                (5.0 / np.log(10.0)) * (1.0 / (1.0 + self.z_hiig) + 1.0 / (Ez * Iz))
            )
            mu_hiig_err = dmu_dz * self.z_hiig_err

            Mum[self.is_hiig] = mu_hiig
            MumErr2[self.is_hiig] = mu_hiig_err ** 2

        R = Mu - Mum
        W = 1.0 / (MuErr2 + MumErr2)
        return -0.5 * np.sum(R ** 2 * W)
