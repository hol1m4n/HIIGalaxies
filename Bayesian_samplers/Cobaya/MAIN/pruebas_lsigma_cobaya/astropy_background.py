"""Background-only Cobaya theory matching the seven supplied Astropy models.

For HIIG replication only. Does not provide CMB spectra, perturbations or r_drag.
Use CLASS/CAMB for the subsequent joint CMB/BAO analysis.
"""
import numpy as np
from astropy.cosmology import (FlatLambdaCDM, LambdaCDM, FlatwCDM, wCDM,
                               Flatw0waCDM, w0waCDM)
from cobaya.theory import Theory


class AstropyBackground(Theory):
    model_index: int = 1

    def initialize(self):
        if int(self.model_index) not in range(1, 8):
            raise ValueError('model_index must be 1...7')
        self._requests = {}

    def get_can_support_params(self):
        return ['H0', 'Om0', 'Ode0', 'w0', 'wa']

    def get_can_provide_params(self):
        return ['omegak']

    def must_provide(self, **requirements):
        for name, request in requirements.items():
            if name == 'omegak':
                continue
            if name not in ('Hubble', 'angular_diameter_distance', 'comoving_radial_distance'):
                raise ValueError(f'Unsupported background product {name}')
            self._requests[name] = np.unique(np.r_[self._requests.get(name, []), request['z']])

    def calculate(self, state, want_derived=True, **params_values_dict):
        number = int(self.model_index)
        cls = {1: FlatLambdaCDM, 2: FlatLambdaCDM, 3: LambdaCDM,
               4: FlatwCDM, 5: wCDM, 6: Flatw0waCDM, 7: w0waCDM}[number]
        args = dict(H0=params_values_dict['H0'], Om0=params_values_dict['Om0'],
                    Tcmb0=2.7255, Neff=3.04, m_nu=0.0)
        if number in (3, 5, 7):
            args['Ode0'] = params_values_dict['Ode0']
        if number in (4, 5, 6, 7):
            args['w0'] = params_values_dict['w0']
        if number in (6, 7):
            args['wa'] = params_values_dict['wa']
        try:
            cosmo = cls(**args)
            functions = dict(Hubble=cosmo.H,
                             angular_diameter_distance=cosmo.angular_diameter_distance,
                             comoving_radial_distance=cosmo.comoving_distance)
            for name, z in self._requests.items():
                values = functions[name](z).value
                if np.any(~np.isfinite(values)):
                    return False
                state[name] = values
        except (ValueError, FloatingPointError, OverflowError):
            return False
        state['derived'] = {'omegak': float(cosmo.Ok0)}

    def _at(self, name, z):
        query = np.atleast_1d(z)
        grid = self._requests[name]
        idx = np.searchsorted(grid, query)
        if np.any(idx >= len(grid)) or not np.array_equal(grid[idx], query):
            raise ValueError(f'Unrequested redshift in {name}')
        return self.current_state[name][idx]

    def get_Hubble(self, z, units='km/s/Mpc'):
        result = self._at('Hubble', z)
        if units == '1/Mpc':
            return result/299792.458
        if units != 'km/s/Mpc':
            raise ValueError(units)
        return result

    def get_angular_diameter_distance(self, z):
        return self._at('angular_diameter_distance', z)

    def get_comoving_radial_distance(self, z):
        return self._at('comoving_radial_distance', z)
