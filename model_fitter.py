#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copied from DFM's script to fit a line

from __future__ import print_function

import emcee
import corner
import lnlikelihood
import model_lineprofile
import math
import numpy as np
import scipy.optimize as op
from scipy import constants
import matplotlib.pyplot as pl
from matplotlib.ticker import MaxNLocator

class model_fitter:

    def __init__(self,data,trans_name,guesses,BHB):

        
        lamred_guess, logN_guess, bD_guess = guesses
        #print('***DEBUGGING***  bD_guess = ', bD_guess)
        self.wave_b = data['wave_b']
        self.flux_b = data['flux_b']
        self.err_b = data['err_b']
        self.wave_r = data['wave_r']
        self.flux_r = data['flux_r']
        self.err_r = data['err_r']
        self.lamlim1 = data['lamlim1']
        self.lamlim2 = data['lamlim2']
        
        self.velres = data['velres']

        transinfo = model_lineprofile.transitions(trans_name)
        self.transinfo = transinfo
        self.transname = trans_name # Added for manual runs

        # MCMC setup
        self.sampndim = 3
        #self.sampnwalk = 100
        self.sampnwalk = 50
        #self.nsteps = 500
        #self.burnin = 150

        # REAL VALUES
        #self.burnin = 100##########
        #self.nsteps = 200##########
        self.theta_guess = [lamred_guess, logN_guess, bD_guess]

        self.burnin = 150
        self.nsteps = 500

        """ Shorter chains for quick test runs
        self.sampnwalk = 50
        self.burnin = 150
        self.nsteps = 500
        """

        
    def maxlikelihood(self):

        """

        Calculate the maximum likelihood model

        """

        chi2 = lambda *args: -2 * lnlikelihood.lnlike(*args)       
        # print('***DEBUGGING***  chi2 = ', chi2)
        # print('***DEBUGGING***  self.theta_guess = ', self.theta_guess)
        # print('***DEBUGGING***  self.transinfo = ', self.transinfo)
        # print('***DEBUGGING***  self.wave_b = ', self.wave_b)
        # print('***DEBUGGING***  self.flux_b = ', self.flux_b)
        # print('***DEBUGGING***  self.err_b = ', self.err_b)
        # print('***DEBUGGING***  self.wave_r = ', self.wave_r)
        # print('***DEBUGGING***  self.flux_r = ', self.flux_r)
        # print('***DEBUGGING***  self.err_r = ', self.err_r)
        # print('***DEBUGGING***  self.velres = ', self.velres)
        result = op.minimize(chi2, self.theta_guess,
                             args=(self.transinfo, self.wave_b, self.flux_b, self.err_b,
                                   self.wave_r, self.flux_r, self.err_r, self.velres))

        self.theta_ml = result["x"]



    def mcmc(self,BHB):

        """
        
        Set up the sampler.
        Then run the chain and make time plots for inspection

        """

        ndim = self.sampndim
        nwalkers = self.sampnwalk
        #startpoint = self.theta_ml
        startpoint = self.theta_guess
        pos = [startpoint + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]

        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnlikelihood.lnprob,
                                        args=(self.transinfo, self.lamlim1, self.lamlim2,
                                              self.wave_b, self.flux_b, self.err_b,
                                              self.wave_r, self.flux_r, self.err_r,
                                              self.velres))

        # Clear and run the production chain.
        print("Running MCMC...")
        sampler.run_mcmc(pos, self.nsteps, rstate0=np.random.get_state())
        print("Done.")

        pl.clf()
        fig, axes = pl.subplots(ndim, 1, sharex=True, figsize=(8, 9))

        for ind in range(0,ndim):
            
            axes[ind].plot(sampler.chain[:, :, ind].T, color="k", alpha=0.4)
            axes[ind].yaxis.set_major_locator(MaxNLocator(5))
            axes[ind].axhline(startpoint[ind], color="#888888", lw=2)
            # axes[ind].set_ylabel("$m$")


        fig.tight_layout(h_pad=0.0)
        fig.savefig("line-time_"+BHB+"_"+self.transname+"_"+str(self.lamlim1)+".png") # Edited for manual runs


        burnin = self.burnin
        samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))
        self.samples = samples

        # Compute the quantiles.
        theta_mcmc = list(map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                        zip(*np.percentile(samples, [16, 50, 84],
                                            axis=0))))
        
        
        self.theta_percentiles = theta_mcmc
        print("""MCMC result:""")                   
        for ind in range(0,ndim):
            print(""" par {0} = {1[0]} +{1[1]} -{1[2]}""".format(ind, theta_mcmc[ind]))
