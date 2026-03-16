import os
from astropy.table import Table


def h2gdat(dpath):
    tpath = dpath + 'indat/h2data.dat'
    data = Table.read(tpath, format='ascii')

    vx = data['col4']
    vxErr = data['col5']
    vy = data['col6']
    vyErr = data['col7']
    vz = data['col2']
    vzErr = data['col3']
    return vx, vxErr, vy, vyErr, vz, vzErr


# Main Code
mcd = os.path.dirname(os.path.abspath(__file__))
dpath = mcd + '/dat/'

# Reading Data
vx, vxErr, vy, vyErr, vz, vzErr = h2gdat(dpath)

for i in range(0, len(vx)):
    print i, vx[i], vxErr[i], vy[i], vyErr[i], vz[i], vzErr[i]
