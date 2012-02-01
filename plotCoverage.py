"""
    Plot field coverage using the ptf_microlensing database
"""

import os, sys
import logging

import numpy as np
from sqlalchemy import func
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import apwlib.geometry as g

from db.DatabaseConnection import *
from db.NumpyAdaptors import *

def plotCoverage(minimum_number_of_observations=1, coordinate_system="equitorial", ax=None):
    """ Given the above parameters (coordinate_system is 
        either equitorial or galactic), plot the PTF fields
        and shade them by how many observations they have
    """
    
    allFields = session.query(Field).all()
    
    if ax == None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="aitoff")
    
    for field in allFields:
        if field.numberOfExposures < minimum_number_of_observations: continue
        
        logging.debug("field: {0}".format(field.id))
        if coordinate_system.lower() == "equitorial":
            x = session.query(func.avg(Exposure.ra)).filter(Exposure.field_id == field.id).one()[0]
            y = session.query(func.avg(Exposure.dec)).filter(Exposure.field_id == field.id).one()[0]
            x = -(x - 180.)
        elif coordinate_system.lower() == "galactic":
            x = session.query(func.avg(Exposure.l)).filter(Exposure.field_id == field.id).one()[0]
            y = session.query(func.avg(Exposure.b)).filter(Exposure.field_id == field.id).one()[0]
            if x > 180: x = -(360. - x)
        
        circ = Ellipse((np.radians(x),np.radians(y)), np.radians(1.5)/np.cos(np.radians(np.fabs(y))), np.radians(1.5), color='k', alpha=0.3)
        ax.add_patch(circ)
    
    if ax != None:
        return ax
    else:
        plt.savefig("allFields_{0}.pdf".format(minimum_number_of_observations))
    
def plotGalacticCenter(minimum_number_of_observations=25, radius=12.):
    """ Plot only fields within `radius` degrees of the 
        galactic center with more than `minimum_number_of_observations`
        exposures
    """
    galacticCenterFields = session.query(Field).join(Exposure).\
                                filter(func.q3c_radial_query(Exposure.l, Exposure.b, 0., 0., radius)).all()
    
    fig = plt.figure()
    ax = fig.add_subplot(111)#, projection="aitoff")
    
    for field in galacticCenterFields:
        if field.numberOfExposures < minimum_number_of_observations: continue
    
        l = session.query(func.avg(Exposure.l)).filter(Exposure.field_id == field.id).one()[0]
        b = session.query(func.avg(Exposure.b)).filter(Exposure.field_id == field.id).one()[0]
        
        if l > 180:
            l = -(360. - ccdExposure.l)
        
        circ = Ellipse((l, b), 1.5, 1.5, color='k', alpha=0.5)
        ax.add_patch(circ)
        ax.plot(l, b, 'r.')
    
    ax.set_xlim(-20., 20.)
    ax.set_ylim(-20., 20.)
    plt.savefig("galacticCenterFields.pdf")

def plotGlobularClusters():
    data = np.genfromtxt("globularClusters.txt", delimiter=",", dtype=str)
    globularData = np.array([(g.RA.fromHours(row[1]).degrees, g.Dec.fromDegrees(row[2]).degrees, float(row[3]), float(row[4])) for row in data], dtype=[("ra",float), ("dec",float), ("l",float), ("b",float)]).view(np.recarray)
    
    x,y = globularData.l, globularData.b
    for ii in range(len(x)):
        if x[ii] > 180: x[ii] = -(360. - x[ii])
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="aitoff")
    ax.plot(x, y, "ro", ms=10, alpha=0.5)
    ax = plotCoverage(25, "galactic", ax)
    plt.savefig("globularClusters.pdf")
    

if __name__ == "__main__":
    #plotCoverage(25, "galactic")
    #plotGalacticCenter(50, 20.0)
    plotGlobularClusters()