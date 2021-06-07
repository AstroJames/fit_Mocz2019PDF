""""
    Title:          Fitting script for the Mocz & Burkhart (2019) PDF.
    Notes:          Utilising log liklehood, MCMC and multi core processing
                    MatplotlibStyle - changing the defualt matplotlib parameters
                    for plots.
    Author:         James Beattie
    First Created:  2 / June / 2021

"""

##################################################################################################################################

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rc, ticker, colors
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{mathrsfs} \usepackage{amssymb}'
rc('font', **{'family': 'DejaVu Sans','size':12})
rc('text', usetex=True)
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.top'] = True
mpl.rcParams['xtick.bottom'] = True
mpl.rcParams['ytick.left'] = True
mpl.rcParams['ytick.right'] = True
mpl.rcParams['figure.constrained_layout.use'] = True
mpl.rcParams['figure.dpi'] = 100

##################################################################################################################################
