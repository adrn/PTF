#!/usr/bin/env python

""" Specify an objid as a command line argument, and this script will
    plot and show the light curve!
"""

from argparse import ArgumentParser
import matplotlib.pyplot as plt
from ptf.db.DatabaseConnection import *

parser = ArgumentParser(description="")
parser.add_argument("-o", "--objid", dest="objid", type=int,
                help="The objid of the light curve you want to plot.")
parser.add_argument("-e", "--error-cut", dest="error_cut", type=float, default=0.0,
                help="An optional error cut to apply to points in the plot.")

args = parser.parse_args()

if args.error_cut == 0:
    error_cut = None
else:
    error_cut = args.error_cut

lc = session.query(LightCurve).filter(LightCurve.objid == args.objid).one()
lc.plot(error_cut=error_cut)
