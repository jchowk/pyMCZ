##############################################################################
##Calculate metalicity, code originally in IDL written by Lisa
##new calculation based on the most recent version of the .pro file.
##
##I ignored the section she had of not calculating the lines that didn't
##meet some mass condition.
##inputs:
## measured - flux data, must be the format returned by readfile()
## num - also returned by readfile()
## outfilename - the name of the file the results will be appended to
## red_corr - True by default
## disp - if True prints the results, default False
## saveres - if True appends the results onto outfilename, True by default
##############################################################################
import sys,os
import numpy as np
from pylab import hist,show
##all the lines that go like
##if S_mass[i] > low_lim and S_mass[i] < 14.0 and all_lines[i] != 0.0:
##are ignored, replaced by (if not G) where G is set to False

IGNOREDUST=False

##list of metallicity methods, in order calculated
Zs=["E(B-V)", #Halpha, Hbeta
    "logR23", #Hbeta,  [OII]3727, [OIII]5007, [OIII]4959

    "D02",    #Halpha, [NII]6584
    "Z94",    #Hbeta,  [OII]3727, [OIII]5007, ([OIII]4959 )
    "M91",    #Hbeta,  [OII]3727, [OIII]5007, ([OIII]4959 )
    "C01",    #[OII]3727, [OIII]5007, [NII]6584, [SII]6717
    "C01_R23",#Hbeta,  [OII]3727, [OIII]5007, ([OIII]4959 )

    "Pi05",   #Hbeta,  [OII]3727, [OIII]5007, ([OIII]4959 )
    "PP04_N2",#Halpha, [NII]6584
    "PP04_O3N2",   #Halpha, Hbeta,[OIII]5007, [NII]6584
    
    "pyqzN2S2_O3S2",    "pyqzN2S2_O3Hb",    "pyqzN2S2_O3O2",    "pyqzN2O2_O3S2",    "pyqzN2O2_O3Hb",    "pyqzN2O2_O3O2",    "pyqzN2Ha_O3Hb",    "pyqzN2Ha_O3O2",


    "KD02_N2O2",   #Halpha, Hbeta,  [OII]3727, [NII]6584
    "KD02_N2Ha",   #Halpha, Hbeta,  [OII]3727, [NII]6584
    "KD02_R23", #Hbeta,  [OII]3727, [OIII]5007, ([OIII]4959 )

    "KD02comb","KD02comb_R23","KD02comb_updated"] #'KD02_N2O2', 'KD03new_R23', 'M91', 'KD03_N2Ha'

    
def get_keys():
    return Zs
 

##############################################################################
##fz_roots function as used in the IDL code  FED:reference the code here!
##############################################################################

#@profile
def calculation(diags,measured,num,(bsmeas,bserr),Smass,mds,outfilename='blah.txt',dust_corr=True,disp=False,saveres=False, verbose=False): 

    global IGNOREDUST
    diags.setdustcorrect()
    raw_lines={}
    raw_lines['[OIII]5007']=np.array([float('NaN')])
    raw_lines['Hb']=np.array([float('NaN')])
    for k in measured.iterkeys():
        #kills all non-finite terms 
        measured[k][~(np.isfinite(measured[k][:]))]=0.0 
        raw_lines[k]=measured[k]



    ######FED why this????????????
    raw_lines['[OIII]4959']=raw_lines['[OIII]5007']/3.
    raw_lines['[OIII]49595007']=raw_lines['[OIII]4959']+raw_lines['[OIII]5007']
    
    diags.setHab(raw_lines['Ha'],raw_lines['Hb'])
    

    #if Ha or Hb is zero, cannot do red correction
    if dust_corr and diags.hasHa and diags.hasHb:
        with np.errstate(invalid='ignore'):
            #print 'extinction correction ',i
            diags.calcEB_V()
    elif dust_corr and not IGNOREDUST:
        response=raw_input("WARNING: reddening correction cannot be done without both H_alpha and H_beta measurement!! Continuing without reddening correction? [Y/n]\n").lower()
        assert(not (response.startswith('n'))),"please fix the input file to include Halpha and Hbeta measurements"
        IGNOREDUST=True
        dust_corr=False
        diags.mds['E(B-V)']=np.ones(len(raw_lines['Ha']))*1e-5
    else:
        diags.unsetdustcorrect()
        diags.mds['E(B-V)']=np.ones(len(raw_lines['Ha']))*1e-5

    for k in ['[OII]3727','[OIII]5007','[OI]6300','[OIII]4959',
              '[SII]6717','[SII]6731','[SIII]9069','[SIII]9532'
              '[OII]3727','[OIII]5007','[OI]6300','[OIII]4959',
              '[NII]6584','[SIII]9532']:
        if k not in raw_lines or len(raw_lines[k])==1:
            raw_lines[k]=np.array([0.]*num)

    diags.setOlines(raw_lines['[OII]3727'], raw_lines['[OIII]5007'], raw_lines['[OI]6300'], raw_lines['[OIII]4959'])

    diags.setNII(raw_lines['[NII]6584'])
    diags.setSII(raw_lines['[SII]6717'],raw_lines['[SII]6731'],raw_lines['[SIII]9069'],raw_lines['[SIII]9532'])

    if diags.checkminimumreq(dust_corr,IGNOREDUST,Smass) == -1:
        return -1
        
    diags.calcNIIOII()
    diags.calcNIISII()


    diags.calcR23()
    #diags.calcS23()

    diags.initialguess()
    mds=mds.split(',')
    #needs N2 and Ha
    if verbose: print "calculating diagnostics: ",mds
    if 'all' in mds:
         diags.calcD02()
         if   os.getenv("PYQZ_DIR"):
              cmd_folder = os.getenv("PYQZ_DIR")+'/'
              if cmd_folder not in sys.path:
                   sys.path.insert(0, cmd_folder)
              import pyqz
              diags.calcpyqz()
         else:
              print '''set path to pyqz as environmental variable :
export PYQZ_DIR="your/path/where/pyqz/resides/ in bash, for example, if you want this diagnostic. '''


         diags.calcPP04()
         diags.calcZ94()
         diags.calcM91()
         #all these above were checked

         diags.Pi05()

         diags.calcKD02_N2O2()
         diags.calcKD02_N2Ha()
         diags.calcC01_ZR23()
         
         diags.calcKD02R23()

         diags.calcKDcombined()
    if 'D02' in mds:
         diags.calcD02()
    if 'pyqz' in mds:
         if   os.getenv("PYQZ_DIR"):
              cmd_folder = os.getenv("PYQZ_DIR")
              if cmd_folder not in sys.path:
                   sys.path.insert(0, cmd_folder)
              import pyqz
              diags.calcpyqz()
         else:
              print '''set path to pyqz as environmental variable 
PYQZ_DIR if you want this diagnostic. '''

    if 'PP04' in mds:
       diags.calcPP04()
    if 'Z94' in mds:
       diags.calcZ94()
    if 'M91' in mds:
       diags.calcM91()
       #all these above were checked
    if 'Pi05' in mds:
       diags.Pi05()
    if 'C01' in mds:
       diags.calcC01_ZR23()
    if 'KD02' in mds :
       diags.calcKD02_N2O2()
       diags.calcKD02_N2Ha()
       
       diags.calcKD02R23()
    if 'KD02comb' in mds:
         diags.calcKDcombined()
