import numpy as np
import scipy.special as sp
from scipy import constants
import math
from linetools.spectra.xspectrum1d import XSpectrum1D
from linetools.spectra import convolve as lsc
from linetools.lists.linelist import LineList
from astropy import units as u
from astropy.units import Quantity

# Set up constants for NaI or CaII
def transitions(name_line):

    if(name_line == 'NaI'):
        name_blue = 'NaI  5891'
        name_red =  'NaI  5897'
        
    elif(name_line == 'CaII'):
        name_blue = 'CaII 3934'
        name_red = 'CaII 3969'

    else:
        print("""This transition is not a good option!  Pick NaI or CaII.""")
        
    strong = LineList('Strong')

    # Get line info in dictionary form
    bline_info = strong[name_blue]
    rline_info = strong[name_red]

    lamblu0 = float(bline_info['wrest'] / u.AA)
    lamred0 = float(rline_info['wrest'] / u.AA)

    # 2/3/19: Lines which are commented out no longer work - seems that linetools was updated and 'log(w*f)' is no longer a valid key, so they are replaced with lines which retrieve it using 'wrest'*'f'
    #lamfblu0 = 10.0**bline_info['log(w*f)']
    #lamfred0 = 10.0**rline_info['log(w*f)']
    lamfblu0 = bline_info['wrest'] * bline_info['f']
    lamfred0 = rline_info['wrest'] * rline_info['f']
    
    #lamblu0 = 5891.5833
    #lamred0 = 5897.5581

    #lamfblu0 = 3718.17822063
    #lamfred0 = 1875.4234758

    return {'lamblu0':lamblu0, 'lamred0':lamred0, 'lamfblu0':lamfblu0, 'lamfred0':lamfred0}





# Set up model line profile
# theta contains lamred, logN, bD (in that order)
def model_lineprofile(theta,transinfo,velres,newwvb,newwvr):

    # First, get info on transitions
    sol = constants.c/1000.0    # km/s
    velratio = 1.0 + (transinfo['lamblu0'] - transinfo['lamred0'])/transinfo['lamred0']
    dmwv = 0.001     # in Angstroms
    modwin = 500.0   # km/s
    
    lamred, logN, bD = theta

    N = 10.0**logN
    lamblu = lamred * velratio
    #print '***DEBUGGING***  transinfo[lamfred0] = ', transinfo['lamfred0']
    #print '***DEBUGGING***  bD = ', bD#['value']
    #print '***DEBUGGING***  N = ', N

    # MW: These are now unitless quantities (was AA * km/s), added .value to transinfo
    taured0 = N * 1.497e-15 * transinfo['lamfred0'].value / bD#['value']
    taublu0 = N * 1.497e-15 * transinfo['lamfblu0'].value / bD#['value']
    #taured0 = N * 1.497e-15 * transinfo['lamfred0'] / bD#['value'] # line from the old code that couldn't read pyigmguesses JSON
    #taublu0 = N * 1.497e-15 * transinfo['lamfblu0'] / bD#['value'] # line from the old code that couldn't read pyigmguesses JSON

    modwv_red_rng = (np.array([-1.0,1.0])*(modwin/sol)*transinfo['lamred0']) + transinfo['lamred0']
    modwv_blu_rng = (np.array([-1.0,1.0])*(modwin/sol)*transinfo['lamblu0']) + transinfo['lamblu0']
    ## modwv_red = (np.array([-500.0,500.0]) * transinfo['lamred0'] / sol) + transinfo['lamred0']
    ## modwv_blu = (np.array([-500.0,500.0]) * transinfo['lamblu0'] / sol) + transinfo['lamblu0']
    modwv_red = np.arange(modwv_red_rng[0], modwv_red_rng[1], dmwv)
    modwv_blu = np.arange(modwv_blu_rng[0], modwv_blu_rng[1], dmwv)
    #print(modwv_red)

    
    #print '***DEBUGGING***  modwv_red = ', modwv_red
    #print '***DEBUGGING***  lamred = ', lamred
    #print '***DEBUGGING***  bD = ', bD#['value']
    #print '***DEBUGGING***  sol = ', sol
    exp_red = -1.0 * (modwv_red - lamred)**2 / (lamred * bD / sol)**2
    exp_blu = -1.0 * (modwv_blu - lamblu)**2 / (lamblu * bD / sol)**2

    taured = taured0 * np.exp(exp_red)
    taublu = taublu0 * np.exp(exp_blu)

    ## Unsmoothed model profile
    #print '***DEBUGGING***  taublu = ', taublu.value
    #print '***DEBUGGING***  f = ', f
    #print '***DEBUGGING***  xk = ', xk
    #print '***DEBUGGING***  d = ', d
    #print '***DEBUGGING***  d[k] = ', d[k]
    #print '***DEBUGGING***  f0 = ', f0
    model_bline = np.exp(-1.0*(taublu)) # old: model_bline = np.exp(-1.0*(taublu.value))
    model_rline = np.exp(-1.0*(taured)) # old: model_rline = np.exp(-1.0*(taured.value))
    xspec_b = XSpectrum1D.from_tuple((modwv_blu,model_bline))
    xspec_r = XSpectrum1D.from_tuple((modwv_red,model_rline))
    
    ## Now smooth with a Gaussian resolution element
    ## Can try XSpectrum1D.gauss_smooth (couldn't get this to work)

    # FWHM resolution in pix
    wvresb = transinfo['lamblu0'] * velres / sol
    wvresr = transinfo['lamred0'] * velres / sol
    pxresb = wvresb / dmwv
    pxresr = wvresr / dmwv

    smxspec_b = xspec_b.gauss_smooth(pxresb)
    smxspec_r = xspec_r.gauss_smooth(pxresr)
    
    
    ## Now rebin to match pixel size of observations
    ## Can try XSpectrum1D.rebin, need to input observed wavelength array
    wv_unit = u.AA
    uwave_b = u.Quantity(newwvb,unit=wv_unit)
    uwave_r = u.Quantity(newwvr,unit=wv_unit)
    
    # Rebinned blue spectrum
    rbsmxspec_b = smxspec_b.rebin(uwave_b)
    modwv_b = rbsmxspec_b.wavelength.value
    modflx_b = rbsmxspec_b.flux.value

    # Rebinned red spectrum
    rbsmxspec_r = smxspec_r.rebin(uwave_r)
    modwv_r = rbsmxspec_r.wavelength.value
    modflx_r = rbsmxspec_r.flux.value

    
    return {'modwv_b':modwv_b, 'modflx_b':modflx_b, 'modwv_r':modwv_r, 'modflx_r':modflx_r, 'spec_b':rbsmxspec_b, 'spec_r':rbsmxspec_r}
