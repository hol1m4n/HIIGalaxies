import re
import os
import numpy as np
from astropy.io import ascii
from astropy.io import fits
from astropy.table import Table
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO
from matplotlib.ticker import AutoMinorLocator
from astropy.constants import L_sun
from matplotlib import gridspec
import astropy.units as u
from math import pi

storage_dir = os.path.join(os.path.expanduser("~"),'Data_storageHII/')
os.makedirs(storage_dir, exist_ok=True)
tex_folder = storage_dir + '/specTx_Tab5ID/'
spec_list = os.listdir(tex_folder)
#spec_list.remove('DOWNspec_listDR7.txt')
spec_list = sorted(spec_list)
len(spec_list)

BZ_libs = ['BZ19Stndr','BZ19Step1','BZ19Step5','BZ19Step5Z10','BZ19Stp5Z10.M300','BZ19Stp5Z10.M600']


for LIB_CASE in BZ_libs:
	tex_folder = '/home/hollman/Data_storageHII/specTx_Tab5ID/'
	n_spec = '1'
	base_dir = f'/home/holman/BASES/Base_{LIB_CASE}/' # cambiar 
	obs_dir = f'{tex_folder}'.replace('//','/')
	mask_dir = '/home/hollman/StarLightv05/merged_masks/'
	out_dir = f'/home/holman/StarLightv05/outputs/{LIB_CASE}/' # cambiar 
	input_folder = f'/home/holman/StarLightv05/inputs/{LIB_CASE}/' # cambiar
	os.makedirs(out_dir , exist_ok=True)
	os.makedirs(input_folder, exist_ok=True)
	phone_number = '-2007200'
	llow_SN = '4570.0'
	lupp_SN = '4650.0'
	Olsyn_ini = '3400.0'
	Olsyn_fin = '8500.0'
	Odlsyn = '1.0'
	fscale_chi2 = '1.0'
	fit_method = 'FIT'
	errors = '0'
	flags = '0'
	ext_laws = ['GD1']
	config_file = f'FR4020.config'
	BASE = f'Base_{LIB_CASE}' # cambiar 


	for i in range(len(spec_list)):

	    with open(f"{input_folder}/grid_{LIB_CASE}{spec_list[i][0:3]}.in", 'w') as file:
                file.write(f"{n_spec:<50} [Number of fits to run]\n")
                file.write(f"{base_dir:<50} [base_dir]\n")
                file.write(f"{obs_dir:<50} [obs_dir]\n")
                file.write(f"{mask_dir:<50} [mask_dir]\n")
                file.write(f"{out_dir:<50} [out_dir]\n")
                file.write(f"{phone_number:<50} [your phone number]\n")
                file.write(f"{llow_SN:<50} [llow_SN] lower-lambda of S/N window\n")
                file.write(f"{lupp_SN:<50} [lupp_SN] upper-lambda of S/N window\n")
                file.write(f"{Olsyn_ini:<50} [Olsyn_ini] lower-lambda for fit\n")
                file.write(f"{Olsyn_fin:<50} [Olsyn_fin] upper-lambda for fit\n")
                file.write(f"{Odlsyn:<50} [Odlsyn] delta-lambda for fit\n")
                file.write(f"{fscale_chi2:<50} [fscale_chi2] fudge-factor for chi2\n")
                file.write(f"{fit_method:<50} [FIT/FXK] Fit or Fix kinematics\n")
                file.write(f"{errors:<50} [IsErrSpecAvailable] 1/0 = Yes/No\n")
                file.write(f"{flags:<50} [IsFlagSpecAvailable] 1/0 = Yes/No\n")

                n_spec = int(n_spec)

                for aux in range(n_spec):
                    file.write(f"{spec_list[i]}  {config_file}  {BASE}  BestMask_{spec_list[i][0:3]}.txt  {ext_laws[aux]}  0.0  150.0  {spec_list[i][0:4]+ext_laws[aux]+'.'+str(LIB_CASE)}.out\n")

