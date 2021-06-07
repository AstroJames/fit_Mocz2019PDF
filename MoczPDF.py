""""
    Title:          Fitting script for the Mocz & Burkhart (2019) PDF.
    Notes:          Utilising log liklehood, MCMC and multi core processing
                    MoczPDF class - defining the Mocz PDF, moment generation
                    for the PDF, and mass-weighted quantities.
    Author:         James Beattie
    First Created:  2 / June / 2021

"""

##################################################################################################################################

import scipy.integrate as integrate
import numpy as np

##################################################################################################################################

class MoczPDF:

    def __init__(self,s0,s_star,f):
        # Initialised variables
        self.s0     = s0
        self.s_star = s_star
        self.f      = f

        # Derived variables
        self.bins       = []
        self.pdf        = []
        self.norm_test  = []
        self.mass_test  = []

    def log_mocz_pdf(self, s):
        """
        logarthimic Mocz PDF from Mocz & Burkhart (2018).

        INPUTS:
        #########################################################################
        s       - log density / mean density
        s0      - mean log density / mean density
        s_star  - variance of the Mocz & Burkhart (2018) PDF
        f       - the time-scale fraction between over-densities and
                under-densities

        OUTPUTS:
        #########################################################################
        ln_pdf  - the natural log probability density function (not normalised)

        """

        ln_pdf = -( (s - self.s0)**2 * ( 1+self.f*(s - self.s0)*np.heaviside(s, self.s0) ) ) / ( 2*self.s_star )
        return(ln_pdf)

    def mass_weighted_mocz_pdf(self, s):
        """
        mass-weighted Mocz PDF from Mocz & Burkhart (2018)

        INPUTS:
        #########################################################################
        s       - log density / mean density
        s0      - mean log density / mean density
        s_star  - variance of the Mocz & Burkhart (2018) PDF
        f       - the time-scale fraction between over-densities and
                under-densities

        OUPUTS:
        #########################################################################
        pdf     - the mass-weighted (exp(s)) probability density function
                (unnormalised)

        """

        pdf = np.exp(s)*np.exp(self.log_mocz_pdf(s))
        return(pdf)

    def mass_weighted_mocz_pdf_mass_corr(self, s, mass_corr):
        """
        mass-weighted Mocz PDF from Mocz & Burkhart (2018)
        shifted by mass_corr so as to conserve mass
        (fitting niavely violates mass conservation)

        INPUTS:
        #########################################################################
        s           - log density / mean density
        s0          - mean log density / mean density
        s_star      - variance of the Mocz & Burkhart (2018) PDF
        f           - the time-scale fraction between over-densities and
                    under-densities
        mass_cor    - the mass correction that shifts the mean to preserve mass conservation

        OUPUTS:
        #########################################################################
        pdf         - the mass-weighted (exp(s)) probability density function
                    (unnormalised)

        """

        pdf = np.exp(s)*np.exp(self.log_mocz_pdf(s+np.log(mass_corr)))
        return(pdf)

    def call_mocz_pdf(self, s):
        """
        Normalise and adjust the mean and skewness of the PDF
        to conserve mass.

        INPUTS:
        #########################################################################
        s           - the logarthmic density / mean density
        s_star      - variance of the Mocz & Burkhart (2018) PDF
        f           - the time-scale fraction between over-densities
                    and under-densities
        s0          - a guess of the initial logarithmic mean density

        OUPUTS:
        #########################################################################
        pdf         - the mass-weighted (exp(s)) probability density
                    function (unnormalised)
        norm_test   - confirm that the integral over the PDF = 1
        mass_test   - confirm that the total mass under the PDF = M
        mu          - adjusted mean
        var         - adjusted variance

        """

        self.bins = s
        # normalise the unadjusted PDF
        norm = integrate.simps(np.exp(self.log_mocz_pdf(s)),s)
        # compute the mass correction based on the integral over the mass_weighted PDF
        mass_corr = integrate.simps(self.mass_weighted_mocz_pdf(s)/norm,s)
        # adjust the PDF
        self.pdf = np.exp(self.log_mocz_pdf(s+np.log(mass_corr)))/norm
        # compute the mean of the adjusted PDF
        mu = integrate.simps(s * np.exp(self.log_mocz_pdf(s+np.log(mass_corr)))/norm,s)
        # compute the variance of the adjusted PDF
        var = integrate.simps((s - mu)**2 * np.exp(self.log_mocz_pdf(s+np.log(mass_corr)))/norm,s)
        # double check that the new PDF conserves probability and mass
        self.norm_test = integrate.simps(np.exp(self.log_mocz_pdf(s+np.log(mass_corr)))/norm,s)
        self.mass_test = integrate.simps(self.mass_weighted_mocz_pdf_mass_corr(s,mass_corr=mass_corr)/norm,s)
        return(self.pdf)
