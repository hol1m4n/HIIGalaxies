# -*- coding: utf-8 -*-
# hiigs main module on mraos
# Modified: Fri 27 Apr 2018 @ lus1/ana

__all__ = ['main']
__version__ = "10.0.0"
__author__ = "Ricardo Chavez (rc681@cam.ac.uk)"
__copyright__ = "Copyright 2018 Ricardo Chavez"
__contributors__ = [
    # Alphabetical by first name.
    'Ricardo Chavez'
]


import sys 
import os
import time 
# import matplotlib
# matplotlib.use('Agg')

# from mch2gh0 import mch2gh0
# from test import test 
# from simslpv4 import simslpv4
# from wtests import wtests
# from zh2gv0 import zh2gv0
from mch2gv69 import mch2gv69
# from mch2gv77 import mch2gv77
# from mch2gv79 import mch2gv79
# from mch2gv80 import mch2gv80
# from mch2gv81 import mch2gv81
# from mch2gv82 import mch2gv82
# from mch2gv75 import mch2gv75
# from mch2gv73 import mch2gv73
# from mch2gv74 import mch2gv74
# from mcsnev15 import mcsnev15  
# from mcbaov6 import mcbaov6
# from mccmbv8 import mccmbv8
# from mccmbv10 import mccmbv10
# from mch2gcmbbaov8 import mch2gcmbbaov8
# from mcsnecmbbaov7 import mcsnecmbbaov7
# from mcHllBAOCMBcosmo import mcHIIBAOCMBcosmo
# from mcjointplotsv3 import mcjointplotsv3


def main():
    start_time = time.time()
    mcd = os.path.dirname(os.path.abspath(__file__))

    dpath = mcd+'/dat/'
    cpath = dpath + 'resultsMN/'
    # cpath = '/export/data/Chavez/hiigs/resultsMN/'
    # cpath = '/Users/rchavez/h2dat/results/'

    ve = '197' # [196], 2, 193

    if len(sys.argv) > 1:
        nsps = int(sys.argv[1])
    else:
        nsps = 1000

    print('+++++++++++++++++++++++++++++++++++++++++++')
    print('hiigs: '+ve)
    print('+++++++++++++++++++++++++++++++++++++++++++')

    # mcHIIBAOCMBcosmo(ve, dpath, cpath, 0, 0, sps=1000, prs=0, vbs=0)

    # mch2gh0(ve, dpath, cpath)

    # test(dpath)

    # simslpv4(ve, dpath, cpath, vbs=0, prs=0, sps=1000, clc=1, fr=2)

    # wtests(ve, dpath, cpath)

    # zh2gv0(ve, dpath, cpath)

    # mch2simv1(ve, dpath, cpath, spl = 0, zps = 2
            # , opt = 5, clc = 1, drd = 2, obs = 1, prs = 0, vbs = 1
            # , sps = 1000, fr0 = 1, fs = 1, ns = 500
            # , sdp = 1
            # )

    mch2gv69(ve, dpath, cpath, spl=1, zps=0
            , opt=3, clc=1, drd=2, obs=1, prs=0, vbs=1
            # # , a=33.11, aErr=0.145, b=5.05, bErr=0.097 #drd = 1
            # , a=33.268, aErr=0.083, b=5.022, bErr=0.058 #drd = 2
            , a=0.0, aErr=0.0, b=0.0, bErr=0.0 #Dflt
            , sps=1000, fr0=1
            )

    # mch2gv79(ve, dpath, cpath, spl=4, zps=4
    #         , opt=3, clc=0, drd=2, obs=1, prs=0, vbs=1
    #     #     , a=33.11, aErr=0.145, b=5.05, bErr=0.097 #drd = 1
    #     #     , a=33.268, aErr=0.083, b=5.022, bErr=0.058 #drd = 2
    #     #     , a=33.255, aErr=0.072, b=5.022, bErr=0.047 #Dflt    
    #         , a=0.0, aErr=0.0, b=0.0, bErr=0.0 #Dflt
    #         , sps=1000, fr0=1, fra=1
    #         )
    
    # mch2gv80(ve, dpath, cpath, spl=5, zps=1
    #         , opt=103, clc=0, drd=1, obs=1, prs=0, vbs=1
    #     #     , a=33.11, aErr=0.145, b=5.05, bErr=0.097 #drd = 1
    #         # , a=33.268, aErr=0.083, b=5.022, bErr=0.058 #drd = 2
    #         , a=0.0, aErr=0.0, b=0.0, bErr=0.0 #Dflt
    #         , sps=10000, fr0=2, fra=0
    #         )

    # mch2gv81(ve, dpath, cpath, spl=1, zps=41
    #         , opt=5, clc=0, drd=2, obs=1, prs=0, vbs=1
    #     #     , a=33.11, aErr=0.145, b=5.05, bErr=0.097 #drd = 1
    #         # , a=33.268, aErr=0.083, b=5.022, bErr=0.058 #drd = 2
    #         # , a=33.255, aErr=0.072, b=5.022, bErr=0.047 #Dflt    
    #         # , a=0.0, aErr=0.0, b=0.0, bErr=0.0 #Dflt
    #          , sps=1000, fr0=2, fra=0
    #         )

    # Use this for High Z data (stable)
    # mch2gv82(ve, dpath, cpath, spl=1, zps=70
    #         , opt=5, clc=0, drd=2, obs=1, prs=1, vbs=1
    #     #     , a=33.11, aErr=0.145, b=5.05, bErr=0.097 #drd = 1
    #     #     , a=33.268, aErr=0.083, b=5.022, bErr=0.058 #drd = 2
    #     #     , a=33.255, aErr=0.072, b=5.022, bErr=0.047 #Dflt    
    #         , a=0.0, aErr=0.0, b=0.0, bErr=0.0 #Dflt
    #         , sps=1000, fr0=1, fra=0
    #         )
 
    # mch2gv83(ve, dpath, cpath, spl=1, zps=40
    #         , opt=55, clc=0, drd=2, obs=1, prs=0, vbs=1
    #     #     , a=33.11, aErr=0.145, b=5.05, bErr=0.097 #drd = 1
    #         # , a=33.268, aErr=0.083, b=5.022, bErr=0.058 #drd = 2
    #         # , a=33.255, aErr=0.072, b=5.022, bErr=0.047 #Dflt    
    #         , a=0.0, aErr=0.0, b=0.0, bErr=0.0 #Dflt
    #         , sps=1000, fr0=2, fra=0
    #         )

    # mch2gv75(ve, dpath, cpath, spl=1, zps=1
            # , opt=51, clc=1, drd=2, obs=1, prs=0, vbs=0
            # # , a=33.11, aErr=0.145, b=5.05, bErr=0.097 #drd = 1
            # , a=33.268, aErr=0.083, b=5.022, bErr=0.058 #drd = 2
            # # , a=0.0, aErr=0.0, b=0.0, bErr=0.0 #Dflt
            # , sps=nsps, fr0=2
            # )


    # mch2gv71(ve, dpath, cpath, spl=1, zps=12
            # , opt=3, clc=1, drd=2, obs=1, prs=0, vbs=0
            # # , a=33.268, aErr=0.083, b=5.022, bErr=0.058 #drd = 2
            # , a=0.0, aErr=0.0, b=0.0, bErr=0.0 #Dflt
            # , sps=1000, fr0=1
            # )

    # mch2gv73(ve, dpath, cpath, prs=0, sps=1000, clc=0)

    # mcsnev15(ve, dpath, cpath, clc=0, opt=0, sps=1000, prs=0, vbs=1)
    
#     mccmbv8(ve, dpath, cpath, opt = 1, prs = 0, sps = 1000, vbs = 1)

    # mccmbv10(ve, dpath, cpath, opt = 2, prs = 0, sps = 10000, vbs = 1)

    # mcbaov6(ve, dpath, cpath, opt = 2, prs = 0, sps = 10000)

    # mch2gcmbbaov8(ve, dpath, cpath, spl = 2, zps = 1
            # , opt = 2, clc = 1, drd = 2, obs = 1, prs = 1, vbs = 0
            # #, a = 33.11, aErr = 0.145, b = 5.05, bErr = 0.097 #drd = 1
            # , a = 33.268, aErr = 0.083, b = 5.022, bErr = 0.058 #drd = 2
            # #, a = 0.0, aErr = 0.0, b = 0.0, bErr = 0.0 #Dflt
            # , sps = 1000, fr0 = 1
            # )

    # mcsnecmbbaov7(ve, dpath, cpath, clc=0, opt=12, sps=1000, prs=0, vbs=1)

    # mcjtplotsv2(ve, dpath, cpath, spl=2, zps=1
            # , opt=21, clc=1, drd=2, obs=1, prs=1, vbs = 0
            # #, a = 33.11, aErr = 0.145, b = 5.05, bErr = 0.097 #drd = 1
            # , a=33.268, aErr=0.083, b=5.022, bErr=0.0                                                                                                  58 #drd = 2
            # #, a = 0.0, aErr = 0.0, b = 0.0, bErr = 0.0 #Dflt
            # , sps=1000, fr0=1
            # )

    # mcjointplotsv3(ve, dpath, cpath, opt=0)

    print('The End')
    print("ETime:--- %s seconds ---" % (time.time() - start_time))
    return


if __name__ == "__main__":
    main()
