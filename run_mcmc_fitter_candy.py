
from __future__ import print_function

import math
import numpy as np
import scipy.optimize as op
from scipy import constants
import os
import fnmatch
import matplotlib.pyplot as pl
pl.rcParams['font.family'] = 'stixgeneral'
from matplotlib.ticker import MaxNLocator
from matplotlib.backends.backend_pdf import PdfPages
import corner
from astropy.io import fits
from astropy.table import Table
from linetools.spectra.xspectrum1d import XSpectrum1D
from linetools.spectra import convolve as lsc
import model_lineprofile
import model_fitter
import interp_json

def run_all(inputfits,inputjson,outtxt,z_component,species,BHB):

    ###########################################################

    sol = constants.c/1000.0    # km/s
    #outdir = '/Users/rubin/Research/BHBs/Analysis/MCMC/' #Kate's original line
    outdir = './' #Hannah's test line

    ## Input spectrum and igm_guesses results
    fil = inputfits
    jsonfil = inputjson
    fit_cmp = 'z'+z_component+'_'+species
    trans_name = species  ## can be either NaI or CaII
    outfits = BHB+trans_name+'.fits'
    BHB_name = BHB
    


    ## Kate's original inputs:
    #fil = '/Users/rubin/Research/BHBs/data/BHB2_J1534+5015a_ltnorm2.fits' #Kate's original file
    #jsonfil = '../../data/J1534+5015_model.json' #Kate's original file
    #fit_cmp = 'z-0.00005_NaI'
    #fit_cmp = 'z-0.00046_CaII'
    #trans_name ='NaI'
    #outfits = 'J1534+5015-'+trans_name+'.fits'

    ## Hannah's test inputs:
    #BHB_name = 'J1231+3719a'
    #fil = '~/Dropbox/BHB_abs/HIRES-March2016/Rawdata/2016mar30/NormSpec/'+BHB_name+'_ltnorm_cor.fits' #Hannah's test file
    #jsonfil = '../../BHB_abs/HIRES-March2016/Rawdata/2016mar30/NormSpec/'+BHB_name+'.json' #Hannah's test file
    #BHB_name = 'J1534+5015'
    #fil = './BHB2_J1534+5015a_ltnorm2.fits' #Hannah's testing John's FITS
    #jsonfil = './J1534+5015_model.json' #Hannah's testing John's JSON
    #fit_cmp = 'z-0.00011_NaI'
    #trans_name = 'NaI'
    #outfits = BHB_name+'-'+fit_cmp+'.fits'

    sp = XSpectrum1D.from_file(fil)
    sp.normalize(sp.co)
    flux = sp.flux.value
    wave = sp.wavelength.value
    error = sp.sig.value
    #cont = sp.co.value
    velres = 6.6   # km/s (FWHM) - from X

    ## For plotting
    if(trans_name=='NaI'):
        xrng = [5888,5900]
    elif(trans_name=='CaII'):
        xrng = [3920,3980]


    ## Use vlim from .json file
    json = interp_json.interp_json(jsonfil,fit_cmp) # This returns {'zfit':zfit, 'vlim':vlim, 'Nfit':Nfit, 'bfit':bfit}

    ## Read in info on absorption transition
    transinfo = model_lineprofile.transitions(trans_name)

    if(json['vlim']==0.0):
        print("You need to adjust fit_cmp!")

    else:

        ## Find line centers
        lamcen_b = (1.0+json['zfit']) * transinfo['lamblu0']
        lamcen_r = (1.0+json['zfit']) * transinfo['lamred0']

        ## Find line ranges
        fitlim_b = (np.array(json['vlim']) * lamcen_b / sol) + lamcen_b
        fitlim_r = (np.array(json['vlim']) * lamcen_r / sol) + lamcen_r


        ## Cut out wave_b, wave_r for NaI
        ind_r = np.where((wave > fitlim_r[0]) & (wave < fitlim_r[1]))
        wv_red = wave[ind_r]
        flx_red = flux[ind_r]
        err_red = error[ind_r]

        ind_b = np.where((wave > fitlim_b[0]) & (wave < fitlim_b[1]))
        wv_blu = wave[ind_b]
        flx_blu = flux[ind_b]
        err_blu = error[ind_b]

        fig, ax = pl.subplots(1,1,sharex=True,sharey=True,figsize=(6.0,5.0))
        ax.plot(wave,flux,color='black')
        ax.plot(wv_red,flx_red,color='red')
        ax.plot(wv_blu,flx_blu,color='blue')
        ax.set_xlim(xrng[0],xrng[1])
        ax.set_ylim(-0.1,1.1)


        data = {'wave_b':wv_blu, 'flux_b':flx_blu, 'err_b':err_blu,
                'wave_r':wv_red, 'flux_r':flx_red, 'err_r':err_red,
                'velres':velres, 'lamlim1':fitlim_r[0], 'lamlim2':fitlim_r[1]}

        ## Guess good model parameters
        #logN = 11.5 # Kate's original line
        logN = json['Nfit']
        #bD = 20.0 # Kate's original line - changed from 6.0 as a test
        bD = json['bfit']
        #lamred = transinfo['lamred0']-0.25 #Kate's original line. BHB lines vary too much in z, need new lamred each fit
        lamred = (1+json['zfit']) * transinfo['lamred0']
        theta_guess = lamred, logN, bD

        ## Make an example model profile
        guess_mod = model_lineprofile.model_lineprofile(theta_guess, transinfo, data['velres'],
                                                        data['wave_b'], data['wave_r'])
        ax.plot(guess_mod['modwv_r'], guess_mod['modflx_r'], color='cyan')
        ax.plot(guess_mod['modwv_b'], guess_mod['modflx_b'], color='cyan')
        #fig.savefig("J1534+5015-"+trans_name+"-debug.png") # Kate's original line
        fig.savefig(BHB_name+'-'+fit_cmp+"-debug.png") # Hannah's test line

        ## Set up fitter
        datfit = model_fitter.model_fitter(data, trans_name, theta_guess)


        # Run maximum likelihood fit
        datfit.maxlikelihood()
        lamred_ml, logN_ml, bD_ml = datfit.theta_ml
        print(fit_cmp)
        #print("""Maximum likelihood result:
        print("""Maximum likelihood fit from JSON input:
        lamred = {0} (guess: {1})
        logN = {2} (guess: {3})
        bD = {4} (guess: {5})                 
        """.format(lamred_ml, lamred, logN_ml, logN, bD_ml, bD))


        # Run the MCMC
        datfit.mcmc()
        # Read in MCMC percentiles
        lamred_mcmc, logN_mcmc, bD_mcmc = datfit.theta_percentiles
        theta_mcmc = [lamred_mcmc[0], logN_mcmc[0], bD_mcmc[0]]
        
        # Output MCMC values to a text file
        with open(outtxt,'a') as file:
            file.write(BHB_name+"\t"+trans_name+"\t{0:.4f}\t{1:.4f}\t{2:.4f}\t{3:.4f}\t{4:.4f}\t{5:.4f}\t{6:.4f}\t{7:.4f}\t{8:.4f}\n".format(lamred_mcmc[0],lamred_mcmc[1],lamred_mcmc[2],logN_mcmc[0],logN_mcmc[1],logN_mcmc[2],bD_mcmc[0],bD_mcmc[1],bD_mcmc[2]))
        

        # Make triangle plot
        fig = corner.corner(datfit.samples, labels=["$lamred$", "$logN$", "$bD$"],truths=[lamred_mcmc[0], logN_mcmc[0], bD_mcmc[0]])

        ax = fig.add_subplot(444)
        ax.plot(wv_red,flx_red,drawstyle='steps-mid',color='black')
        modlineprof = model_lineprofile.model_lineprofile(theta_mcmc,transinfo,velres,wv_blu,wv_red)
        ax.plot(modlineprof['modwv_r'],modlineprof['modflx_r'],color='cyan',lw=2)
        ax.ticklabel_format(useOffset=False)
        ax.xaxis.set_major_locator(MaxNLocator(3))
        ax.set_xlabel(r'Wavelength (Ang)')
        ax.set_ylim(0.5,1.2)
        ax.plot([transinfo['lamred0'],transinfo['lamred0']], [0.0,2.0])

        ax = fig.add_subplot(443)
        ax.plot(wv_blu,flx_blu,drawstyle='steps-mid',color='black')
        ax.plot(modlineprof['modwv_b'],modlineprof['modflx_b'],color='cyan',lw=2)
        ax.ticklabel_format(useOffset=False)
        ax.xaxis.set_major_locator(MaxNLocator(3))
        ax.set_xlabel(r'Wavelength (Ang)')
        ax.set_ylim(0.5,1.2)
        ax.set_ylabel(r'Normalized Flux')
        ax.plot([transinfo['lamblu0'],transinfo['lamblu0']], [0.0,2.0])


        #sv_samples.append(datfit.samples)
        #sv_percentiles.append(datfit.theta_percentiles)


        #fig.savefig("J1534+5015-"+trans_name+"-line-triangle-lowblim.png") # Kate's original line
        fig.savefig(BHB+"_"+trans_name+"_"+z_component+"_line_triangle.png") # Hannah's line

        print(datfit.samples.shape)
        lamred_samples = datfit.samples[:,0]
        logN_samples = datfit.samples[:,1]
        bD_samples = datfit.samples[:,2]

        #t = Table([datfit.samples, datfit.theta_percentiles], names=('samples', 'percentiles'))
        t = Table([lamred_samples, logN_samples, bD_samples],
                  names=('lamred_samples', 'logN_samples', 'bD_samples'))
        fits.writeto(outdir+outfits, np.array(t),clobber=True)
