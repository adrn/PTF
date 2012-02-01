""" 
    Given a field RA/Dec, print the commands to download PTF 
    images from IPAC.
"""

import os
import apwlib.geometry as g
import numpy as np

def getImageList(ra, dec, outputFile=["test.txt"]):
    """ Assume RA and Dec are in decimal degrees! """
    
    ra = list(ra)
    dec = list(dec)
    if not isinstance(outputFile, list) and not isinstance(outputFile, np.array):
        outputFile = [outputFile]
    
    if len(list(ra)) != len(list(dec)) and len(list(ra)) != len(list(outputFile)):
        raise ValueError("All inputs must have the same length!")
        
    wgets = []
    for ii in range(len(ra)):
        wgets.append('wget --http-user=PTF --http-password=palomar "http://kanaloa.ipac.caltech.edu/ibe/search/ptf/dev/process?POS={0},{1}" -O {2}'.format(ra[ii],dec[ii],outputFile[ii]))
    
    return wgets

def getImageDownloadCommand(pimage):
    """ 
        pimage is something like: proc/2011/10/12/f11/c1/p5/v1/PTF_201110123982_i_p_scie_t093322_u009413166_f11_p013320_c01.fits
    """
    print 'wget --http-user=PTF --http-password=palomar "{0}"'.format(os.path.join("http://kanaloa.ipac.caltech.edu/ibe/data/ptf/dev/process/", pimage))

if __name__ == "__main__":
    data = np.genfromtxt("globularClusters.txt", delimiter=",", dtype=str)
    globularData = np.array([(g.RA.fromHours(row[1]).degrees, g.Dec.fromDegrees(row[2]).degrees, float(row[3]), float(row[4])) for row in data], dtype=[("ra",float), ("dec",float), ("l",float), ("b",float)]).view(np.recarray)
    
    commands = getImageList(globularData.ra, globularData.dec, [name + ".txt" for name in data[:,0]])
    
    getImageDownloadCommand("proc/2011/10/12/f11/c1/p5/v1/PTF_201110123982_i_p_scie_t093322_u009413166_f11_p013320_c01.fits")
    
    
    #http://kanaloa.ipac.caltech.edu/ibe/data/ptf/dev/process/proc/2011/10/12/f12/c1/p5/v1/PTF_201110124005_i_p_scie_t093640_u009414639_f12_p013320_c01.jpg 