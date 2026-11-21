#!/usr/bin/env python3
"""Convert a MultiNest *post_equal_weights.dat file to a Cobaya proposal covmat.

Example (Model 1):
  python covmat_desde_multinest.py Lens_FlatLCDM_post_equal_weights.dat \
      --names alpha_ls beta_ls H0 --output model1.covmat

Model 7 with the bundled Astropy configs:
  --names alpha_ls beta_ls H0 Om0 Ode0 w0 wa

Requires EQUAL-WEIGHT posterior samples in the original parameter order.
It is NOT suitable for the weighted .txt file. Third column is h and will be
multiplied by 100. Only the final log-likelihood column is discarded.
"""
import argparse
from pathlib import Path
import numpy as np

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('samples')
    p.add_argument('--names', nargs='+', required=True)
    p.add_argument('--output', default='multinest_proposal.covmat')
    a=p.parse_args()
    n=len(a.names)
    if n<3 or a.names[:3]!=['alpha_ls','beta_ls','H0'] or len(set(a.names))!=n:
        raise ValueError('Names must be unique and begin alpha_ls beta_ls H0')
    raw=np.loadtxt(a.samples,ndmin=2)
    if raw.shape[1]!=n+1:
        raise ValueError(f'Expected {n} parameter columns plus final log-likelihood; got {raw.shape[1]}')
    x=raw[:,:n].copy()
    if len(x)<=n or not np.isfinite(x).all():
        raise ValueError('Insufficient or non-finite posterior samples')
    if np.any((x[:,2]<.5)|(x[:,2]>1.)):
        raise ValueError('Third column is not h in the supplied [0.5, 1] prior')
    x[:,2]*=100.
    cov=np.cov(x,rowvar=False,ddof=1)
    np.linalg.cholesky(cov)  # Reject singular covariance instead of silently changing it.
    if Path(a.output).exists():
        raise FileExistsError(f'Choose a new --output; {a.output} already exists')
    np.savetxt(a.output,cov,header=' '.join(a.names),fmt='%.16e')
    widths=1/np.sqrt(np.diag(np.linalg.inv(cov)))
    print('Saved',a.output)
    print('Approximate conditional widths (Gaussian approximation):')
    for name,width in zip(a.names,widths):
        print(f'{name}: {width:.8g}')
