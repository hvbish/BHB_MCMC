# Define likelihood and priors
import numpy as np
import scipy.special as sp
from scipy import constants
import math
import model_lineprofile

# Define the probability function as likelihood * prior.
# This is the prior
# lamlim1 and lamlim2 come from the json file
def lnprior(theta,transinfo,lamlim1,lamlim2):

    lamred, logN, bD = theta

    sol = constants.c/1000.0   # km/s
    #transinfo = model_lineprofile.transitions(trans_name)
    #vlim = 700.0     # km/s
    #lamlim1 = -1.0 * (vlim * transinfo['lamred0'] / sol) + transinfo['lamred0']
    #lamlim2 = (vlim * transinfo['lamred0'] / sol) + transinfo['lamred0']

    logNlim1 = 8.0
    logNlim2 = 13.5

    bDlim1 = 0.5
    bDlim2 = 75.0 #Changed from original value of 20.0
    
    #if -5.0 < m < 5.0 and -10.0 < b < 10.0 and -10.0 < lnf < 10.0:
    if lamlim1 < lamred < lamlim2 and logNlim1 < logN < logNlim2 and bDlim1 < bD < bDlim2:
        return 0.0
    return -np.inf

# This is the likelihood
def lnlike(theta, transinfo, wave_b, flux_b, err_b, wave_r, flux_r, err_r, velres):
    
    lamred, logN, bD = theta

    model = model_lineprofile.model_lineprofile(theta,transinfo,velres,wave_b,wave_r)
    flx_model_b = model['modflx_b']
    flx_model_r = model['modflx_r']

    err = np.concatenate([err_b,err_r])
    flx_model = np.concatenate([flx_model_b,flx_model_r])
    flux = np.concatenate([flux_b,flux_r])
    
    inv_sigma2 = 1.0/(err**2)
    return -0.5*(np.sum((flux-flx_model)**2*inv_sigma2 - np.log(2.0*math.pi*inv_sigma2)))



def lnprob(theta, transinfo, lamlim1, lamlim2, wave_b, flux_b, err_b, wave_r, flux_r, err_r, velres):
    lp = lnprior(theta, transinfo, lamlim1, lamlim2)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, transinfo, wave_b, flux_b, err_b, wave_r, flux_r, err_r, velres)


# Debugging
#def isigsq(theta, x, y, yerr):
#    m, b, lnf = theta
#    model = m * x + b
#    inv_sigma2 = 1.0/(yerr**2 + model**2*np.exp(2*lnf))
#    term1 = (y-model)**2*inv_sigma2
#    term2 = np.log(2.0*math.pi*inv_sigma2)
#    return term1-term2

# More debugging
#def term(theta, x, y, yerr):
#    m, b, lnf = theta
#    model = m * x + b
#    inv_sigma2 = 1.0/(yerr**2 + model**2*np.exp(2*lnf))
#    term1 = (y-model)**2*inv_sigma2
#    term2 = np.log(2.0*math.pi*inv_sigma2)
#    return (y-model)**2
