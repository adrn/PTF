""" Contains the PTFLightCurve class """

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard Library
import sys, os
import copy
import logging

# Third-party dependencies
import numpy as np

# Package dependences
import ptf.analyze.analyze as analyze

# ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=

class PTFLightCurve:
    """ Represents a PTF Light Curve """
    
    def __init__(self, mjd, mag, error, metadata=None):
        """ Create a PTFLightCurve by passing equal-length arrays of mjd, magnitude, and
            magnitude errors. This object also accepts an optional metadata parameter, 
            which is any numpy recarray that contains extra information about the light
            curve data.
        """
            
        idx = np.argsort(mjd)
        self.amjd = self.mjd = np.array(mjd)[idx].astype(np.float64)
        self.amag = self.mag = np.array(mag)[idx].astype(np.float64)
        self.error = np.array(error)[idx].astype(np.float64)
        self.metadata = np.array(metadata)
    
    def plot(self, ax=None, **kwargs):
        """ Either plots the light curve and show()'s it to the display, or plots it on 
            the given matplotlib Axes object
        """
        
        if not kwargs.has_key("ls") and not kwargs.has_key("linestyle"):
            kwargs["ls"] = 'none'
        
        if not kwargs.has_key("marker"):
            kwargs["marker"] = 'o'
        
        if not kwargs.has_key("c") and not kwargs.has_key("color"):
            kwargs["c"] = 'k'
        
        if ax == None:
            import matplotlib.pyplot as plt
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.errorbar(self.mjd, self.mag, self.error, ecolor='0.7', capsize=0, **kwargs)
            ax.set_xlim(min(self.mjd), max(self.mjd))
            ax.set_ylim(ax.get_ylim()[::-1])
            plt.show()
            return
        
        ax.errorbar(self.mjd, self.mag, self.error, ecolor='0.7', capsize=0, **kwargs)
        ax.set_ylim(ax.get_ylim()[::-1])
        return ax
    
    def save(self, filename, overwrite=False):
        """ Save the light curve to a file """

        if os.path.exists(filename) and not overwrite:
            raise IOError("File {} already exists! You must specify overwrite=True or be more careful!".format(filename))
        elif os.path.exists(filename) and overwrite:
            os.remove(filename)
        
        xx, ext = os.path.split(filename)
        
        if ext == ".txt":
            np.savetxt(filename, np.transpose((self.mjd, self.mag, self.error)), fmt="%.5f\t%.5f\t%.5f")
        elif ext == ".pickle":
            import cPickle as pickle
            
            f = open(filename)
            pickle.dump(self, f)
            f.close()
        else:
            raise ValueError("I don't know how to handle {} files!".format(ext))
            
