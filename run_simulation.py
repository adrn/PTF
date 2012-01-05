# Run the entire simulation, e.g. loop this:
#   - (simulated_event.py) generate a fake data set given the time samples
#   - (step1.py) See if >1 points are outside of 2-sigma (??), and if so, if they are clustered
#   - (step2.py) Attempt to fit the point-lens, point-mass microlensing curve to the data set
# britt lundgren

# Standard Library
import sys, os
import cPickle as pickle

# External Packages
import numpy as np
import matplotlib.pyplot as plt

# Project Scripts
import simulated_event as SE
import step1
import step2

np.random.seed(115)
NUM_SIMULATIONS = 10000
SAVEFIGS = False
SHOW = False

def run_one_simulation(timesteps):
    id = np.random.randint(sys.maxint)

    true_params = (np.exp(-10.*np.random.uniform()), \
                   np.random.uniform(min(timesteps), max(timesteps)), \
                   np.random.uniform(5, 50), \
                   SE.RMagToFlux(15.))
    """
    true_params = (np.exp(-10.*np.random.uniform()), \
                   55180, \
                   np.random.uniform(5, 50), \
                   SE.RMagToFlux(15.))
    """

    data = SE.simulate_event(timesteps, true_params=true_params)
    #plot_model(id, data, true_params)
    
    contMag, contStd = step1.get_continuum(data)
    #plot_continuum(id, data, contMag, contStd)
    
    clusterIndices = step1.find_clusters(data, contMag, contStd)
    #plot_clusters(id, data, clusterIndices)
    
    if not clusterIndices.all():
        print "Cluster not found!"
        return False
    
    initial_u0 = np.exp(-10.*np.random.uniform())
    initial_t0 = np.median(data.t[clusterIndices])
    initial_tE = (max(data.t[clusterIndices]) - min(data.t[clusterIndices])) / 2.
    initial_F0 = SE.RMagToFlux(15.)
    
    p0 = (initial_u0,\
          initial_t0, \
          initial_tE, \
          initial_F0)
    
    try:
        popt = step2.fit_lightcurve(data, p0)
    except RuntimeError:
        print "maxfev reached!"
        return False
    #plot_fit(id, data, popt, true_params)
    
    if not SAVEFIGS and SHOW:
        plt.show()
    
    return True

# Plotting
def plot_model(id, data, true_params):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    modelT = np.arange(min(data.t), max(data.t), 0.1)
    ax.errorbar(data.t, data.mag, data.sigma, ls='None')
    ax.plot(modelT, SE.FluxToRMag(SE.FLUXMODEL(modelT, *true_params)), 'r-')
    ax.set_ylim(ax.get_ylim()[::-1])
    ax.set_xlabel("MJD")
    ax.set_ylabel(r"$M_R$")
    if SAVEFIGS:
        fig.savefig("sample_lightcurves/{0}_model.png".format(id))
    del fig

def plot_continuum(id, data, contMag, contStd):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(data.t, data.mag, 'b.')
    ax.axhline(contMag)
    ax.axhline(contMag+3*contStd, c='r', ls='--')
    ax.axhline(contMag-3*contStd, c='r', ls='--')
    ax.set_ylim(ax.get_ylim()[::-1])
    ax.set_xlabel("MJD")
    ax.set_ylabel(r"$M_R$")
    if SAVEFIGS:
        fig.savefig("sample_lightcurves/{0}_continuum_search.png".format(id))
    del fig

def plot_clusters(id, data, clusterIndices):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(data.t, data.mag, 'b.')
    ax.plot(data.t[clusterIndices], data.mag[clusterIndices], 'g+', ms=10)
    ax.axvline(np.median(data.t[clusterIndices]), c='r')
    ax.set_ylim(ax.get_ylim()[::-1])
    ax.set_xlabel("MJD")
    ax.set_ylabel(r"$M_R$")
    if SAVEFIGS:
        fig.savefig("sample_lightcurves/{0}_clusters.png".format(id))
    del fig

def plot_fit(id, data, popt, true_params):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    modelT = np.arange(min(data.t), max(data.t), 0.1)
    ax.errorbar(data.t, data.mag, data.sigma, ls='None')
    ax.plot(modelT, SE.FluxToRMag(SE.FLUXMODEL(modelT, *true_params)), 'r-', label='Model')
    ax.plot(modelT, SE.FluxToRMag(SE.FLUXMODEL(modelT, *popt)), 'g-', label='Fit')
    ax.set_ylim(ax.get_ylim()[::-1])
    ax.set_xlabel("MJD")
    ax.set_ylabel(r"$M_R$")
    ax.legend()
    if SAVEFIGS:
        fig.savefig("sample_lightcurves/{0}_fit.png".format(id))
    del fig

if __name__ == "__main__":
    # Get all .pickle filenames
    pickles = [os.path.join("sample_lightcurves", name) \
                for name in os.listdir("sample_lightcurves") \
                if os.path.splitext(name.lower())[1] == ".pickle"]
    
    counter = 0.
    for pickleIndex in np.random.randint(len(pickles), size=NUM_SIMULATIONS):
        f = open(pickles[pickleIndex])
        timesteps = pickle.load(f)
        f.close()
        
        success = run_one_simulation(timesteps)
        if success:
            counter += 1.
    
    print "Fraction:", counter/NUM_SIMULATIONS