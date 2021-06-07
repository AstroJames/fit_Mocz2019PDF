""""
    Title:          Fitting script for the Mocz & Burkhart (2019) PDF.
    Notes:          Utilising log liklehood, MCMC and multi core processing
                    Emcee class - describing some of the fitting and plotting routines.
    Author:         James Beattie
    First Created:  2 / June / 2021

"""

##################################################################################################################################

import numpy as np
import pandas as pd
import emcee
import corner
from multiprocessing import Pool, cpu_count
from MatplotlibStyle import *
from MoczPDF import *

##################################################################################################################################


class Emcee:

    def __init__(self,pdf_data,initial_guess,n_cores):
        # Initialised variables
        bins,pdf,pdf_std    = pdf_data
        self.bins           = bins
        self.pdf            = pdf
        self.pdf_std        = pdf_std
        self.initial        = initial_guess
        self.n_cores        = n_cores

        # MCMC parameters
        self.pdf_threshold  = 1e-8
        self.num_of_walkers = 50
        self.num_of_params  = len(self.initial)
        self.num_of_steps   = 1000
        self.pos_of_walkers = self.initial + 1e-4 * np.random.randn(self.num_of_walkers,self.num_of_params)

        # Derived variables
        self.samples            = []
        self.best_fit_params    = []


    def log_likelihood(self, theta, bins, pdf, pdf_std):
        """
        The log likelihood function for the Mocz PDF,
        including mass and probability conservation.

        INPUTS:
        #########################################################################
        theta           - the set of parameters (s_0, s_star, f) in that order
        bins            - bins of the histogram for different values of s
        pdf             - the value of the PDF for each bin
        pdf_err         - the uncertainty in each bin
        pdf_threshold   - a lower pdf threshold to avoid weird affects when
                        scaling with log

        OUTPUTS:
        #########################################################################
        log_like    - the log likelihood

        """

        # read out parameters
        s0, s_star, f = theta

        # create a PDF for a given parameter set
        fit     = MoczPDF(s0, s_star, f)
        model   = fit.call_mocz_pdf(bins)

        # Avoid catastrophic cancellation of local variables.
        model[(model < self.pdf_threshold) & (model > 0)] = 0
        model = model[pdf_std > 0]
        pdf = pdf[pdf_std > 0]
        pdf_std = pdf_std[pdf_std > 0]

        # construct the likelihood function
        sigma_square = pdf_std ** 2. + model ** 2.
        log_like = -0.5 * np.sum((pdf - model) ** 2 / sigma_square + np.log(sigma_square))
        return(log_like)

    def log_prior(self, theta):
        """
        the log prior for the Mocz PDF

        INPUTS:
        #########################################################################
        theta       - the set of parameters (s0, s_star, f) in that order

        OUTPUTS:
        #########################################################################
        0.0         - returns 1.0 for the probability in log space
        -np.inf     - returns 0.0 for the probability in log space

        """

        # read out parameters
        s0, s_star, f = theta

        if 0 < s_star < 10.0 and 0.0 < f < 1.0 and -10 < s0 < 5:
            return(0.0)
        return(-np.inf)

    def log_probability(self, theta, bins, pdf, pdf_std):
        """
        The posterior function for the Mocz PDF

        INPUTS:
        #########################################################################
        theta       - the set of parameters (s0, s_star, f) in that order
        bins        - bins of the histogram for different values of s
        pdf         - the value of the PDF for each bin
        pdf_std     - the uncertainty in each bin

        OUTPUTS:
        #########################################################################
        -np.inf     - returns 0.0 for the probability in log space
        log_prob    - log probability for a given set of parameters, theta

        """

        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return(-np.inf)
        log_prob = lp + self.log_likelihood(theta, bins, pdf, pdf_std)
        return(log_prob)

    def run_chain(self):
        """
        Run the emcee chains, sampling the posterior distribution of
        the Mocz PDF.

        INPUTS:
        #########################################################################
        num_of_walkers  - the number of MCMC walkers
        num_of_params   - the number of model parameters (3)
        bins            - bins of the histogram for different values of s
        pdf             - the value of the PDF for each bin
        pdf_std         - the uncertainty in each bin
        pos_of_walkers  - the initial position of the walkers
                        (derived from the initial guess)
        n_cores         - the number of cores to use for the walkers

        OUTPUTS:
        #########################################################################
        samples         - the entire probability sample of all MCMC walkers
        best_fit_params - the 16th, 50th, 84th quantiles for the walkers,
                        defining the best best fit parameters, in dictionary
                        data type {50:...,16:...,84:...}.

        """

        print(f"run_chain: Running walkers with {self.n_cores} CPUs")
        with Pool(self.n_cores) as pool:
            sampler = emcee.EnsembleSampler(self.num_of_walkers,
                                            self.num_of_params,
                                            self.log_probability,
                                            args=(self.bins, self.pdf, self.pdf_std),
                                            pool=pool)
            sampler.run_mcmc(self.pos_of_walkers,
                             self.num_of_steps,
                             progress=True)

        # flatten sample data and compute the appropriate quantiles
        self.samples = sampler.get_chain(discard=100,thin=15,flat=True)
        best_fit_median = np.median(self.samples,axis=0)
        best_fit_16th   = np.quantile(self.samples,0.16,axis=0)
        best_fit_84th   = np.quantile(self.samples,0.84,axis=0)
        self.best_fit_params = {"50":best_fit_median,
                                "16":best_fit_16th,
                                "84":best_fit_84th}

        return(self.best_fit_params)

    def corner_plot(self,out_label):
        """
        Create a corner plot of the posterior for the fit.

        INPUTS:
        #########################################################################
        out_label           - a label for the plot, in this case, the simulation
                            label, but can be anything.
        samples             - the entire probability sample of all MCMC walkers

        OUTPUTS:
        #########################################################################
        corner_{out_label}.pdf    - a pdf output of the corner plot

        """

        # labels for the three parameters
        labels          = [r"$s_0$", r"$s^*$", r"$f$"]
        # initialise figure
        fig             = corner.corner(self.samples,
                            labels=labels,
                            quantiles=[0.16, 0.5, 0.84],
                            show_titles=True,
                            title_kwargs={"fontsize": 12})
        plt.savefig(f"./corner_{out_label}.pdf")
        plt.close()

    def pdf_plot(self,out_label):
        """
        A plot for of the pdf model and the pdf data, overlayed.

        INPUTS:
        #########################################################################
        best_fit_params     - the best fit parameters for thbe Mocz PDF, derived
                            from the MCMC algorithm
        out_label           - a label for the plot, in this case, the simulation
                            label, but can be anything.
        bins                - bins of the histogram for different values of s
        pdf                 - the value of the PDF for each bin
        pdf_std             - the uncertainty in each bin
        pdf_threshold       - a lower pdf threshold to avoid weird affects when
                            scaling with log

        OUTPUTS:
        #########################################################################
        mocz_fit_{sim}.pdf   - a pdf output of the pdf plot


        """

        def contruct_log_std_bars():
            upper = 10** ( np.log10(self.pdf) + (1./np.log(10)) * ( self.pdf_std / self.pdf ) )
            lower = 10** ( np.log10(self.pdf) - (1./np.log(10)) * ( self.pdf_std / self.pdf ) )
            return(upper,lower)

        f, ax = plt.subplots()
        s0, s_star,f   = self.best_fit_params["50"]
        fit            = MoczPDF(s0, s_star, f)
        pdf            = fit.call_mocz_pdf(self.bins)
        print(f"best fit: total probability {fit.norm_test}")
        print(f"best fit: total mass {fit.mass_test}")
        bins_ = self.bins[pdf>self.pdf_threshold]
        pdf_ = pdf[pdf>self.pdf_threshold]
        ax.plot(self.bins[self.pdf>self.pdf_threshold],
                self.pdf[self.pdf>self.pdf_threshold],"k",label="data")
        ax.plot(bins_,pdf_,"r--",label="best fit")
        upper, lower = contruct_log_std_bars()
        ax.fill_between(self.bins,lower,upper,color="k",alpha=0.1)
        ax.set_yscale("log")
        ax.set_ylabel(r"$p(s)$")
        ax.set_xlabel(r"$s\equiv\ln(\rho/\rho_0)$")
        plt.legend()
        ax.set_ylim(self.pdf_threshold,10)
        ax.set_xlim(-10,10)
        plt.savefig(f"./mocz_fit_{out_label}.pdf")
        plt.close()
