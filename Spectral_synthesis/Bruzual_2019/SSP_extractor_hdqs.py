import os
from pathlib import Path
import shutil
from astropy.io import fits
import argparse
import subprocess
import numpy as np
import warnings
from astropy.io import ascii
from astropy.table import Table
import datetime, platform
import urllib.request
import requests



def check_link(path): # Function that takes care of seeing if the file exists on the web or not
    r = requests.head(path)
    status = r.status_code == requests.codes.ok
    if (status == True):
        return True
    else:
        return False
    

def library_creator(Path = os.getcwd(),
                  IMF='Chabrier',
                  Mup=100,
                  Z=[],
                  Ages=[],
                  lib_name = '',
                  obs = ''):

    warnings.filterwarnings('ignore')
    print("Working at: ",Path)

    main_link = 'https://www.bruzual.org/CB19/'
    metallicities = ['0000',
                     '0001',
                     '0002',
                     '0005',
                     '001',
                     '002',
                     '004',
                     '006',
                     '008',
                     '010',
                     '014',
                     '017',
                     '020',
                     '030',
                     '040',
                     '060',
                     ]
    metal_code = ['Z01',
                'Z02',
                'Z03',
                'Z04',
                'Z05',
                'Z06',
                'Z07',
                'Z08',
                'Z09',
                'Z10',
                'Z11',
                'Z12',
                'Z13',
                'Z14',
                'Z15',
                'Z16',
                ]
    SSP_ages = np.arange(1,222,1).tolist()

    
    while (IMF != 'Chabrier' and IMF != 'Kroupa' and IMF != 'Salpeter'):
        IMF = input('Invalid IMF. Available options: \n - Chabrier \n - Kroupa \n - Salpeter')
    while (Mup != 100 and Mup != 300 and Mup != 600):
        Mup = input('Invalid Mup value. Available options: \n - 100 \n - 300 \n - 600')

    Ages = np.unique(Ages).tolist()
    proper_ones_Z = [f for f in Z if f in metallicities]
    rejected_Z = [f for f in Z if f not in metallicities]
    proper_ones_Age = [f for f in Ages if f in SSP_ages]
    rejected_Age = [f for f in Ages if f not in SSP_ages]

    print('Selected IMF:',IMF)
    print('Selected Mup:',Mup)
    print('Selected Z list:', proper_ones_Z)
    print('Selected Age list:', proper_ones_Age)

    if len(rejected_Z) >= 1:
        print('Not found values for Z:',rejected_Z)
    if len(rejected_Age) >= 1:
        print('Not found values for Age:',rejected_Age,'\n')

    if IMF == 'Chabrier':
        tmp_imf = 'chab'
    if IMF == 'Kroupa':
        tmp_imf = 'kroup'
    if IMF == 'Salpeter':
        tmp_imf = 'salp'
    if Mup == 100:
        tmp_mup = ''
    if Mup == 300:
        tmp_mup = '_MU300'
    if Mup == 600:
        tmp_mup = '_MU600'


    # Obtener fits separados por IMF, Mup y metalicidad.

    FITS_list = [[],[]]

    for i in range(len(proper_ones_Z)):

        if os.path.exists(f"{Path}/cb2019_z{Z[i]}_{tmp_imf}{tmp_mup}_hr_xmilesi_ssp.fits"):
            print(f"Already on folder: cb2019_z{Z[i]}_{tmp_imf}{tmp_mup}_hr_xmilesi_ssp.fits\n")
        else:
            if check_link(f"{main_link}{IMF}IMF/Mup{Mup}/cb2019_z{Z[i]}_{tmp_imf}{tmp_mup}_hr_xmilesi_ssp.fits"):
                print(f"Getting: cb2019_z{Z[i]}_{tmp_imf}{tmp_mup}_hr_xmilesi_ssp.fits")
                download = urllib.request.urlretrieve(
                    f"{main_link}{IMF}IMF/Mup{Mup}/cb2019_z{Z[i]}_{tmp_imf}{tmp_mup}_hr_xmilesi_ssp.fits",
                    f"{Path}/cb2019_z{Z[i]}_{tmp_imf}{tmp_mup}_hr_xmilesi_ssp.fits")
            else:
                print(f"File ...{IMF}IMF/Mup{Mup}/cb2019_z{Z[i]}_{tmp_imf}{tmp_mup}_hr_xmilesi_ssp.fits not found")

        FITS_list[0].append(f"cb2019_z{Z[i]}_{tmp_imf}{tmp_mup}_hr_xmilesi_ssp.fits")
        FITS_list[1].append(f"{Z[i]}")









    # Extraer las edades, por archivo de metalicidad.
    print(proper_ones_Age)
    #Age_rearr = np.where(np.array(proper_ones_Age) < 2,-1,np.array(proper_ones_Age))
    #print(Age_rearr)
    #age_list_mod = f'"{Age_rearr}"'.replace('[','')
    #age_list_mod = age_list_mod.replace(']','')
    #print(age_list_mod)


    library_name = f'Base_bruz2019_{lib_name}'

    with open(library_name, 'w') as f:
        f.write(f'{len(proper_ones_Z)*len(proper_ones_Age)}                       [N_base]\n')


    for Z in range(len(FITS_list[0])):

        fits_file_perZ = fits.open(FITS_list[0][Z])
        SED_per_age = Table.read(fits_file_perZ[1]) # [1:221] 0 -> Wavelength
        Params_per_age = Table.read(fits_file_perZ[2]) # [0:220]
        Ages_in_fits_file = Table.read(fits_file_perZ[5]) # [0:220]

        lambda_range = list(SED_per_age.columns.values())[0].tolist()

        for A in range(len(proper_ones_Age)):
            if Mup == 100:
                tmp_mup = '100'

            age_code = f'{proper_ones_Age[A]}'
            if proper_ones_Age[A] < 100:
                if proper_ones_Age[A] < 10:
                    age_code = f"00{proper_ones_Age[A]}"
                else:
                    age_code = f"0{proper_ones_Age[A]}"


            output_file_name = f"bruz2019_{tmp_imf}_{tmp_mup.replace('_MU','')}_{metal_code[metallicities.index(FITS_list[1][Z])]}_{age_code}_ssp.sed"
            input_file_name = FITS_list[0][Z]


            age_in_sed = Ages_in_fits_file['age-yr'][proper_ones_Age[A]-1]
            age_e6 = age_in_sed / 1e6
            age_formatted = f"{age_e6:.5f}e6"


            m_stars_per_age = Params_per_age['Mstars'][proper_ones_Age[A]-1]

            #if Params_per_age['logage'][proper_ones_Age[A]-1] == 0:
            #    age_in_sed = 0
            #    age_e6 = age_in_sed / 1e6
            #    age_formatted = f"{age_e6:.6f}e6"
            #else:
            #    age_in_sed = round(10 ** Params_per_age['logage'][proper_ones_Age[A]-1],0)
            #    age_e6 = (10 ** Params_per_age['logage'][proper_ones_Age[A]-1]) / 1e6
            #    age_formatted = f"{age_e6:.6f}e6"


            age_in_sed = int(age_in_sed)
            column_size = 2
            lambda_flux = [lambda_range, list(SED_per_age.columns.values())[proper_ones_Age[A]].tolist()]

            with open(output_file_name, 'w') as f:
                f.write(f'# Output file name = {output_file_name}\n')
                f.write(f'# Input file name  = {input_file_name}\n')
                f.write(f'# Column           {column_size}\n')
                f.write(f'# Record           {proper_ones_Age[A]}\n')
                f.write(f'# Age (yr)      {age_in_sed}\n')
                f.write(f'# Lambda(A)       Flux   \n')
                ascii.write(lambda_flux, f, format="no_header", delimiter=" ", comment="#")


            with open(library_name, 'a') as f:
                f.write(
                    f"{output_file_name:<25}        "
                    f"{age_formatted:<15}      "
                    f"0.{FITS_list[1][Z]:<15}     "
                    f"{f'age{age_code}_{metal_code[metallicities.index(FITS_list[1][Z])]}':<15}      "
                    f"{f'{m_stars_per_age}':<12}     "
                    f"{'0':<6}     "
                    f"0.0000\n"
                )

    with open(library_name, 'a') as f:
        f.write(f"# spec-names                             age [yr]             met                 SSP_code             M_star          ---         alpha_Fe\n")
        f.write(f"#\n")
        f.write(f"# Library name: {lib_name}\n")
        f.write(f"#       - IMF: {IMF}\n")
        f.write(f"#       - Mup: {Mup}\n")
        f.write(f"# Base with N_Z = {len(proper_ones_Z)} metallicities and N_t = {len(proper_ones_Age)} ages.\n")
        f.write(f"# Obs:{obs}\n")
        f.write(f"# From Bruzual2019 SPPs. Output for HIIG_scatter_analysis Holman Q.S, Juan T.P, Cesar C. Ricardo C.M.\n")
        f.write(f"# Created at {platform.node()} {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.\n")

    os.makedirs(f'{Path}/Base_{lib_name}', exist_ok=True)


    new_file_list = [f for f in os.listdir(Path) if 'bruz2019' in f.lower() or 'sed' in f.lower()]
    for archivo in new_file_list:
        if os.path.exists(archivo):
            #os.remove(f'{Path}/Base_{lib_name}/{archivo}')
            shutil.move(archivo, f'{Path}/Base_{lib_name}',)
        else:
            print(f"El archivo no se encontró: {archivo}")


#fits_download(IMF='Chabrier',
#            Mup=100,
#            Z = ['0001','0005','004','008','020','060'],
#            Ages=[35,57,75,81,87,90,99,110,125,131,136,140,145,150,158,162,163,174,182,190,195,205,217,219,221],
#            lib_name = 'standard',
#            obs = 'Intentando replicar el mismo rango de edades y metalicidades que en Base_MILESM.Sbmm. El rango de edades difiere un poco respecto a la edad maxima y los intervalos entre edades. La metalicidad de 0004 se cambio a 0005 y de 050 a 060.',
#            )

tabla_edades = """ N    age(yr)        N    age(yr)        N    age(yr)        N    age(yr)        N    age(yr)        N    age(yr)        N    age(yr)        N    age(yr)      
1:  0.000E+00      32:  8.500E+05      63:  3.800E+06      94:  1.200E+07     125:  4.000E+07     156:  8.000E+08     187:  5.500E+09     218:  1.325E+10     
2:  1.500E+04      33:  9.000E+05      64:  3.900E+06      95:  1.250E+07     126:  4.250E+07     157:  8.500E+08     188:  5.750E+09     219:  1.350E+10     
3:  2.000E+04      34:  9.500E+05      65:  4.000E+06      96:  1.300E+07     127:  4.500E+07     158:  9.000E+08     189:  6.000E+09     220:  1.375E+10     
4:  3.000E+04      35:  1.000E+06      66:  4.100E+06      97:  1.350E+07     128:  4.750E+07     159:  1.000E+09     190:  6.250E+09     221:  1.400E+10     
5:  4.000E+04      36:  1.100E+06      67:  4.200E+06      98:  1.400E+07     129:  5.000E+07     160:  1.100E+09     191:  6.500E+09     
6:  5.000E+04      37:  1.200E+06      68:  4.300E+06      99:  1.450E+07     130:  5.350E+07     161:  1.200E+09     192:  6.750E+09     
7:  6.000E+04      38:  1.300E+06      69:  4.400E+06     100:  1.500E+07     131:  5.700E+07     162:  1.300E+09     193:  7.000E+09     
8:  7.000E+04      39:  1.400E+06      70:  4.500E+06     101:  1.600E+07     132:  6.000E+07     163:  1.400E+09     194:  7.250E+09     
9:  8.000E+04      40:  1.500E+06      71:  4.600E+06     102:  1.700E+07     133:  7.000E+07     164:  1.500E+09     195:  7.500E+09     
10:  9.000E+04      41:  1.600E+06      72:  4.700E+06     103:  1.800E+07     134:  8.000E+07     165:  1.600E+09     196:  7.750E+09     
11:  1.000E+05      42:  1.700E+06      73:  4.800E+06     104:  1.900E+07     135:  9.000E+07     166:  1.700E+09     197:  8.000E+09     
12:  1.150E+05      43:  1.800E+06      74:  4.900E+06     105:  2.000E+07     136:  1.000E+08     167:  1.800E+09     198:  8.250E+09     
13:  1.300E+05      44:  1.900E+06      75:  5.000E+06     106:  2.100E+07     137:  1.100E+08     168:  1.900E+09     199:  8.500E+09     
14:  1.450E+05      45:  2.000E+06      76:  5.250E+06     107:  2.200E+07     138:  1.200E+08     169:  2.000E+09     200:  8.750E+09     
15:  1.600E+05      46:  2.100E+06      77:  5.500E+06     108:  2.300E+07     139:  1.400E+08     170:  2.100E+09     201:  9.000E+09     
16:  1.750E+05      47:  2.200E+06      78:  5.750E+06     109:  2.400E+07     140:  1.600E+08     171:  2.200E+09     202:  9.250E+09     
17:  1.800E+05      48:  2.300E+06      79:  6.000E+06     110:  2.500E+07     141:  1.800E+08     172:  2.300E+09     203:  9.500E+09     
18:  1.900E+05      49:  2.400E+06      80:  6.250E+06     111:  2.600E+07     142:  2.000E+08     173:  2.400E+09     204:  9.750E+09     
19:  2.000E+05      50:  2.500E+06      81:  6.500E+06     112:  2.700E+07     143:  2.250E+08     174:  2.500E+09     205:  1.000E+10     
20:  2.500E+05      51:  2.600E+06      82:  6.750E+06     113:  2.800E+07     144:  2.500E+08     175:  2.600E+09     206:  1.025E+10     
21:  3.000E+05      52:  2.700E+06      83:  7.000E+06     114:  2.900E+07     145:  2.750E+08     176:  2.800E+09     207:  1.050E+10     
22:  3.500E+05      53:  2.800E+06      84:  7.250E+06     115:  3.000E+07     146:  3.000E+08     177:  3.000E+09     208:  1.075E+10     
23:  4.000E+05      54:  2.900E+06      85:  7.500E+06     116:  3.100E+07     147:  3.500E+08     178:  3.250E+09     209:  1.100E+10     
24:  4.500E+05      55:  3.000E+06      86:  8.000E+06     117:  3.200E+07     148:  4.000E+08     179:  3.500E+09     210:  1.125E+10     
25:  5.000E+05      56:  3.100E+06      87:  8.500E+06     118:  3.300E+07     149:  4.500E+08     180:  3.750E+09     211:  1.150E+10     
26:  5.500E+05      57:  3.200E+06      88:  9.000E+06     119:  3.400E+07     150:  5.000E+08     181:  4.000E+09     212:  1.175E+10     
27:  6.000E+05      58:  3.300E+06      89:  9.500E+06     120:  3.500E+07     151:  5.500E+08     182:  4.250E+09     213:  1.200E+10     
28:  6.500E+05      59:  3.400E+06      90:  1.000E+07     121:  3.600E+07     152:  6.000E+08     183:  4.500E+09     214:  1.225E+10     
29:  7.000E+05      60:  3.500E+06      91:  1.050E+07     122:  3.700E+07     153:  6.500E+08     184:  4.750E+09     215:  1.250E+10     
30:  7.500E+05      61:  3.600E+06      92:  1.100E+07     123:  3.800E+07     154:  7.000E+08     185:  5.000E+09     216:  1.275E+10     
31:  8.000E+05      62:  3.700E+06      93:  1.150E+07     124:  3.900E+07     155:  7.500E+08     186:  5.250E+09     217:  1.300E+10"""

tabla_metalicidades = ['0000',    #1
                     '0001',  #2
                     '0002',  #3
                     '0005',  #4
                     '001',  #5
                     '002',  #6
                     '004',  #7
                     '006',  #8
                     '008',  #9
                     '010',  #10
                     '014',  #11
                     '017',  #12
                     '020',  #13
                     '030',  #14
                     '040',  #15
                     '060',  #16
                     ]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--Path', type=str, default=os.getcwd(), help='Ruta donde se guardaran los archivos y la libreria extraida')
    parser.add_argument('--IMF', type=str, help='IMF seleccionada: Chabrier, Kroupa o Salpeter')
    parser.add_argument('--Mup', type=int, help='Mup seleccionada: 100, 300 o 600')
    
    parser.add_argument('--Z', nargs='+', type=str, help=f'Lista de strings con las metalicidades seleccionadas: {tabla_metalicidades}')
    parser.add_argument('--Ages', nargs='+', type=int, help=f'Lista de enteros con las edades seleccionadas (1-221): {tabla_edades}')
    
    parser.add_argument('--lib_name', type=str, help='Keyword de la libreria')
    parser.add_argument('--obs', type=str, help='Observaciones de esta libreria')
    
    args = parser.parse_args()
    
    library_creator(
        Path=args.Path, 
        IMF=args.IMF, 
        Mup=args.Mup, 
        Z=args.Z, 
        Ages=args.Ages, 
        lib_name=args.lib_name, 
        obs=args.obs
    )