""""
    Title:          Fitting script for the Mocz & Burkhart (2019) PDF.
    Notes:          Utilising log liklehood, MCMC and multi core processing.
                    Main file.
    Author:         James Beattie
    First Created:  2 / June / 2021

"""

##################################################################################################################################

import numpy as np
import pandas as pd
from Emcee import *
from MoczPDF import *

##################################################################################################################################

def main():
    # Fix the random seed for the MCMC process
    np.random.seed(42)

    # Define the pdf data to be fit to
    fit_label       = "M2MA2"
    pdf_data        = pd.read_csv(f"../../Data/averaged_data/volume_weighted/{sim_label}_ln_dens.csv")
    bins            = np.array(pdf_data["bins"])    # the bins from the PDF
    pdf             = np.array(pdf_data["PDF"])     # the values of the PDF
    pdf_std         = np.array(pdf_data["PDF_std"]) # the uncertainty (1\sigma) in each bin
    pdf_data        = np.array([bins,pdf,pdf_std])

    # Initial parameters guesses
    s_star          = 2             # initial guess for the variance s - distribution
    f               = 0.5           # initial guess for the intermittency param (0.5 is a good guess)
    s0              = -1            # initial guess for the mean of the s - distribution
    initial         = np.array([s0,s_star,f])

    # emcee parameters
    n_cores         = 8 # the number of cores to use for the fit

    # Initialise the PDF and fitting routines
    FitMoczPDF      = Emcee(pdf_data,initial,n_cores)

    # Compute the best fit parameters for s_0, s_star, f
    best_fit_params =  FitMoczPDF.run_chain()

    # Plot projections of the posterior distribution for the
    # fitting parameters
    FitMoczPDF.corner_plot(fit_label)

    # Plot the fit overlayed with the data
    FitMoczPDF.pdf_plot(fit_label)

if __name__ == '__main__':
    main()
