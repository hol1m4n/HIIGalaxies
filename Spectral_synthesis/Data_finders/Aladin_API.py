import os
import sys
import glob
import numpy as np
import  scipy.optimize as op
import matplotlib.pyplot as plt
import urllib.request
from astropy.io import fits as fits
from astropy.table import Table
import pandas as pd
import urllib.request
import requests
import astropy.units as u
from astropy.coordinates import SkyCoord
from pathlib import Path
from numpy import savetxt
from PyAstronomy import pyasl
from scipy.interpolate import interp1d
#from dustmaps.sfd import SFDQuery
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
from astropy.table import Table
from astropy.cosmology import FlatwCDM
import emcee
import numpy as np
import corner
import scipy.optimize as op
import pandas as pd
from astropy.io import ascii
import urllib.request
import requests


from astropy.coordinates import Angle, SkyCoord
from ipyaladin import Aladin, Marker
from pathlib import Path


from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.wcs import WCS
from astroquery.simbad import Simbad
from astroquery.vizier import Vizier
import matplotlib.pyplot as plt
import numpy as np

from ipyaladin import Aladin


aladin = Aladin(
    #target=SkyCoord(coord.ra.degree, coord.dec.degree, unit="deg"),
    target = "NGC 1073",
    fov=Angle(0.1, "deg"),
    height =1000,
    width = 500,
    reticle_size=20,
    coo_frame = "ICRSd",
    reticle_color="#08fd00",
    #overlay_survey = "SDSS9",
    #overlay_survey_opacity = 1,
    survey = "SDSS9"
)

fitsT = aladin.get_view_as_fits()

wcsDATA = WCS(fitsT[0].header)

r_data = fitsT[0].data[0, :,:]
g_data = fitsT[0].data[1, :,:]
b_data = fitsT[0].data[2, :,:]

rgb_composite = np.dstack((r_data, g_data, b_data))

fig = plt.figure(figsize=(20,10))
ax = fig.add_subplot(1, 1, 1, projection=wcsDATA, slices=('x', 'y', 3))
ax.imshow(rgb_composite)
ax.plot(T, 'o', color='white', label='Data Points')
fig.savefig('AlaTEST.png')