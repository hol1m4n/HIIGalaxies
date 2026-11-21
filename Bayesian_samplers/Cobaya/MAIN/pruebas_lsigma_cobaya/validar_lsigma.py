#!/usr/bin/env python3
"""Compare the supplied likelihood implementations without running MultiNest.

Run from the folder containing the eight original Python files:
    python validar_lsigma.py --synthetic --output auditoria_sintetica.json
or use actual data:
    python validar_lsigma.py --data Compilation2026_main.csv \
        --anchors GEHR_base.csv --anchor-host Galaxia \
        --anchor-mu mu_w --anchor-error sigma_w --output auditoria_real.json

Add --class-check to evaluate Model 1 through the real Cobaya/CLASS pipeline.
No sampling is launched. Original Python files are never modified.
Requires numpy, scipy, pandas, astropy and cobaya; classy for --class-check.
"""
from __future__ import annotations

import argparse
import ast
import importlib.util
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd
import astropy
from astropy import constants as const, units as u
from astropy.cosmology import (FlatLambdaCDM, LambdaCDM, FlatwCDM, wCDM,
                               Flatw0waCDM, w0waCDM)
from scipy.special import logsumexp

B = 2.5 / np.log(10.0)
COSMO_CLASSES = {1: FlatLambdaCDM, 2: FlatLambdaCDM, 3: LambdaCDM,
                 4: FlatwCDM, 5: wCDM, 6: Flatw0waCDM, 7: w0waCDM}
THETAS = {
    1: [33.2, 5.0, 0.7],
    2: [33.2, 5.0, 0.7, 0.28],
    3: [33.2, 5.0, 0.7, 0.28, 0.55],
    4: [33.2, 5.0, 0.7, 0.28, -0.9],
    5: [33.2, 5.0, 0.7, 0.28, 0.55, -0.9],
    6: [33.2, 5.0, 0.7, 0.28, -0.9, 0.2],
    7: [33.2, 5.0, 0.7, 0.28, 0.55, -0.9, 0.2],
}


def cosmology(model, theta):
    args = dict(H0=theta[2] * 100, Om0=0.3 if model == 1 else theta[3],
                Tcmb0=2.7255 * u.K)
    if model in (3, 5, 7):
        args['Ode0'] = theta[4]
    if model in (4, 6):
        args['w0'] = theta[4]
    if model in (5, 7):
        args['w0'] = theta[5]
    if model in (6, 7):
        args['wa'] = theta[-1]
    return COSMO_CLASSES[model](**args)


def load_original_model(path):
    """Extract ONLY likelihood methods; never execute imports or top-level runs."""
    tree = ast.parse(Path(path).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name == 'Lsig_Ho_sampler')
    names = {'lens_sigma_eff', 'lens_lognormal_params', 'prepare_lensing_cache',
             'lensing_loglike_gh', 'lnlike'}
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if len(cls.body) != len(names):
        raise ValueError(f'Missing required methods in {path}')
    namespace = dict(np=np, u=u, const=const, logsumexp=logsumexp,
                     LENS_DM_COEFF=B, LENS_GH_ORDER=20)
    namespace.update({c.__name__: c for c in set(COSMO_CLASSES.values())})
    namespace['LENS_GH_X'], namespace['LENS_GH_W'] = np.polynomial.hermite.hermgauss(20)
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['Lsig_Ho_sampler']()


def load_cobaya_class(path):
    spec = importlib.util.spec_from_file_location('likelihood_under_test', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.LSigmaHIIGLensing


class AstropyProvider:
    """Supply identical background distances to isolate likelihood differences."""
    def __init__(self, cosmo):
        self.cosmo = cosmo

    def get_angular_diameter_distance(self, z):
        return self.cosmo.angular_diameter_distance(z).value

    def get_Hubble(self, z, units='km/s/Mpc'):
        if units != 'km/s/Mpc':
            raise ValueError(units)
        return self.cosmo.H(z).value

    def get_comoving_radial_distance(self, z):
        return self.cosmo.comoving_distance(z).value

    def get_param(self, name):
        if name != 'omegak':
            raise ValueError(name)
        return self.cosmo.Ok0


def synthetic_data():
    cosmo = cosmology(1, THETAS[1])
    z = np.array([0.01, 0.05, 0.2, 0.6, 1., 2., 4., 8.])
    mu = np.r_[29., 29., 31., cosmo.distmod(z).value]
    x = np.linspace(1.1, 1.9, len(mu))
    residual = np.array([.07, -.10, .03, .12, -.15, .08, -.12, .11, -.08, .17, -.1])
    flux = 33.2 + 5.0*x - (mu + residual + 100.19477738511641)/2.5
    frame = pd.DataFrame({
        'GEHR_id': ['A', 'A', 'B'] + [f'H{i}' for i in range(len(z))],
        'origin_id': [0]*3 + [1]*len(z),
        'z_or_mu': np.r_[29., 29., 31., z],
        'e_z_or_e_mu': np.r_[.08, .08, .12, 0.002*(1+z)],
        'log_sigma': x, 'e_log_sigma': .025,
        'log_f_Hbeta': flux, 'e_log_f_Hbeta': .04})
    anchors = pd.DataFrame({'Galaxia': ['A', 'B'], 'mu_w': [29., 31.],
                            'sigma_w': [.08, .12]})
    return frame, anchors


def prepare_same_sample(data, anchors, args):
    for col in [args.anchor_host, args.anchor_mu, args.anchor_error]:
        if col not in anchors:
            raise ValueError(f'Missing anchor column {col!r}; set --anchor-* options')
    hosts = anchors[args.anchor_host].astype(str).str.strip()
    if hosts.duplicated().any():
        raise ValueError('Duplicate anchor hosts: would duplicate rows in original merge')
    data = data.copy()
    data['GEHR_id'] = data['GEHR_id'].astype(str).str.strip()
    origin = data['origin_id'].to_numpy(float)
    if not np.isfinite(origin).all() or np.any(origin < 0) or np.any(origin != np.round(origin)):
        raise ValueError('Use finite non-negative integer origin IDs to match both selections')
    # The old group_reader also updates matching HIIG rows: fail explicitly.
    if ((origin != 0) & data['GEHR_id'].isin(hosts)).any():
        raise ValueError('HIIG ID matches an anchor host: original group_reader overwrites its redshift')
    keep = (origin != 0) | data['GEHR_id'].isin(hosts)
    if args.zmax is not None:
        keep &= (origin == 0) | (data['z_or_mu'] <= args.zmax)
    data = data.loc[keep].reset_index(drop=True)
    if np.any((data.origin_id != 0) & (data.z_or_mu == 20.)):
        raise ValueError('z=20 lies on a different lensing-mask boundary in the two originals')
    anchors = anchors.copy()
    anchors[args.anchor_host] = hosts
    old = data.copy()
    mask = old.origin_id == 0
    mapping = anchors.set_index(args.anchor_host)
    old.loc[mask, 'z_or_mu'] = old.loc[mask, 'GEHR_id'].map(mapping[args.anchor_mu])
    old.loc[mask, 'e_z_or_e_mu'] = old.loc[mask, 'GEHR_id'].map(mapping[args.anchor_error])
    return data, anchors, old


def derivative_audit(cosmo, z):
    z = np.asarray(z, float)
    dc = cosmo.comoving_distance(z).value
    dm = cosmo.comoving_transverse_distance(z).value
    h = cosmo.H(z).value
    dh = const.c.to_value(u.km/u.s) / cosmo.H0.value
    ok = cosmo.Ok0
    if ok > 0:
        ck = np.cosh(np.sqrt(ok)*dc/dh)
    elif ok < 0:
        ck = np.cos(np.sqrt(-ok)*dc/dh)
    else:
        ck = np.ones_like(z)
    a = 5/np.log(10.)
    c = const.c.to_value(u.km/u.s)
    exact = a*(1/(1+z) + c*ck/(h*dm))
    original_multinest = a*(1/(1+z) + c/(h*dc))
    original_cobaya = a*(1/(1+z) + c/(h*dm))
    dz = 1.e-5*z
    finite_difference = (cosmo.distmod(z+dz).value - cosmo.distmod(z-dz).value)/(2*dz)
    return dict(z=z.tolist(), exact=exact.tolist(), multinest=original_multinest.tolist(),
                cobaya=original_cobaya.tolist(), finite_difference=finite_difference.tolist(),
                max_exact_fd_abs=float(np.max(np.abs(exact-finite_difference))))


def run(args):
    if args.synthetic:
        data, anchors = synthetic_data()
    else:
        if not args.data or not args.anchors:
            raise ValueError('Specify --synthetic or both --data and --anchors')
        data = pd.read_csv(args.data, comment='#', dtype={'GEHR_id': str})
        anchors = pd.read_csv(args.anchors, comment='#', dtype={args.anchor_host: str})
    data, anchors, old = prepare_same_sample(data, anchors, args)
    cls = load_cobaya_class(args.likelihood)
    report = dict(synthetic=args.synthetic, astropy_version=astropy.__version__,
                  counts=dict(total=len(data), anchors=int((data.origin_id==0).sum()),
                              hiig=int((data.origin_id!=0).sum())), models={})
    with tempfile.TemporaryDirectory(prefix='lsigma_audit_') as temp:
        data_path, anchor_path = Path(temp)/'data.csv', Path(temp)/'anchors.csv'
        data.to_csv(data_path, index=False)
        anchors.to_csv(anchor_path, index=False)
        options = dict(data_file=str(data_path), anchor_file=str(anchor_path),
                       anchor_host_column=args.anchor_host, anchor_mu_column=args.anchor_mu,
                       anchor_error_column=args.anchor_error, zmax=None,
                       include_logdet=False, correlate_host_mu=False,
                       sigma_v_pec_kms=0., sigma_sys_mag=0., apply_lensing=True,
                       lensing_zmin=0., lensing_zmax=20., lensing_method='gh', gh_order=20)
        if args.redshift_error_method:
            options['redshift_error_method'] = args.redshift_error_method
        like = cls(options)
        for number, fiducial in THETAS.items():
            previous = load_original_model(Path(args.models)/f'Lensing_Model{number}.py')
            cache = previous.prepare_lensing_cache(old, zmax=20.)
            results = []
            for shift in (0., -1., 1.):
                theta = np.array(fiducial, dtype=float)
                theta[:3] += shift*np.array([.02, -.01, .02])
                cosmo = cosmology(number, theta)
                like.initialize_with_provider(AstropyProvider(cosmo))
                old_value = float(previous.lnlike(theta=theta, DF=old, lens_cache=cache))
                new_value = like.logp(alpha_ls=theta[0], beta_ls=theta[1], sigma_int=0.)
                saved = like._ez_hiig.copy()
                zero_old = old.copy()
                zero_old.loc[zero_old.origin_id != 0, 'e_z_or_e_mu'] = 0.
                like._ez_hiig[:] = 0.
                zero_new_value = like.logp(alpha_ls=theta[0], beta_ls=theta[1], sigma_int=0.)
                zero_old_value = float(previous.lnlike(theta=theta, DF=zero_old, lens_cache=cache))
                like._ez_hiig[:] = saved
                like._lensing_method = 'analytic'
                analytic = like.logp(alpha_ls=theta[0], beta_ls=theta[1], sigma_int=0.)
                like._lensing_method = 'gh'
                # Independent numerical derivative of Astropy's distance modulus.
                hi = old.origin_id.to_numpy() != 0
                zz = old.loc[hi, 'z_or_mu'].to_numpy(float)
                step = 1.e-5*zz
                derivative = (cosmo.distmod(zz+step).value-cosmo.distmod(zz-step).value)/(2*step)
                mu_model = old.z_or_mu.to_numpy(float).copy()
                mu_model[hi] = cosmo.distmod(zz).value
                err = old.e_z_or_e_mu.to_numpy(float).copy()
                err[hi] *= derivative
                mu_obs = (2.5*(theta[0]+theta[1]*old.log_sigma.to_numpy(float)
                              -old.log_f_Hbeta.to_numpy(float))-100.19477738511641)
                var = 6.25*(old.e_log_f_Hbeta.to_numpy(float)**2
                            + theta[1]**2*old.e_log_sigma.to_numpy(float)**2)+err**2
                reference = float(previous.lensing_loglike_gh(mu_obs-mu_model, np.sqrt(var), cache,
                                                              normalized=False))
                results.append(dict(theta=theta.tolist(), lnL_multinest=old_value,
                                    lnL_cobaya_same_astropy=new_value, delta=new_value-old_value,
                                    delta_with_zero_ez=zero_new_value-zero_old_value,
                                    delta_vs_finite_difference=new_value-reference,
                                    delta_analytic_minus_gh20=analytic-new_value))
            report['models'][str(number)] = results
        # Directly expose the normalization branch bug in the submitted file.
        R, V = np.array([0.1, -0.2]), np.array([0.09, 0.25])
        like.include_logdet = True
        got = like._diag_gaussian_loglike(R, V)
        expected = float(-.5*np.sum(R**2/V+np.log(2*np.pi*V)))
        report['normalization'] = dict(actual=got, expected=expected, delta=got-expected)
        like.include_logdet = False
        # Mathematical equality of analytic and GH with a broader synthetic core.
        U = np.array([-.005, -.02]); S = np.sqrt(-2*U)
        report['lensing_kernel_delta_analytic_minus_gh20'] = (
            like._lensing_loglike_analytic(R, V, U, S)-like._lensing_loglike_gh(R, V, U, S))
        report['curvature_derivatives'] = derivative_audit(cosmology(3, THETAS[3]), [.1, 1., 3.])
        if args.class_check:
            from cobaya.model import get_model
            info = dict(likelihood={'hiig': {'external': cls, **options}},
                        theory={'classy': {'path': args.class_path,
                                'extra_args': {'Omega_b': .048, 'Omega_cdm': .252,
                                    'N_ncdm': 0, 'N_ur': 3.04,
                                    'T_cmb': 2.7255}}},
                        params={'H0': {'prior': {'min': 50., 'max': 100.}},
                                'alpha_ls': {'prior': {'min': 32.5, 'max': 34.5}},
                                'beta_ls': {'prior': {'min': 4.5, 'max': 5.5}},
                                'sigma_int': 0., 'omegak': 0.}, stop_at_error=True)
            with get_model(info) as model:
                comparisons=[]
                for row in report['models']['1']:
                    a,b,h = row['theta']
                    val=float(model.loglikes({'H0':100*h, 'alpha_ls':a, 'beta_ls':b},
                                             return_derived=False)[0])
                    comparisons.append(dict(theta=row['theta'], lnL_class=val,
                                            delta_class_minus_astropy=val-row['lnL_multinest']))
                report['class_check_model1']=comparisons
    return report


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--models', default='.', help='Directory containing Lensing_Model1.py ... 7.py')
    parser.add_argument('--likelihood', default='lsigma_hiig_lensing_likelihood.py')
    parser.add_argument('--data'); parser.add_argument('--anchors')
    parser.add_argument('--anchor-host', default='Galaxia')
    parser.add_argument('--anchor-mu', default='mu_w')
    parser.add_argument('--anchor-error', default='sigma_w')
    parser.add_argument('--zmax', type=float, default=None)
    parser.add_argument('--synthetic', action='store_true')
    parser.add_argument('--class-check', action='store_true')
    parser.add_argument('--class-path', default='global')
    parser.add_argument('--redshift-error-method', choices=['curved','legacy_multinest'])
    parser.add_argument('--output', default='auditoria_lsigma.json')
    args=parser.parse_args()
    result=run(args)
    Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print('Saved:', args.output)
    for n, rows in result['models'].items():
        print('Model',n,'max |delta lnL|=',max(abs(r['delta']) for r in rows),
              'zero ez=',max(abs(r['delta_with_zero_ez']) for r in rows))
    print('Normalization test:',result['normalization'])
