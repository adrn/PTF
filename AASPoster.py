# coding: utf-8
"""
    Generate figures that will go on my AAS poster
"""
from __future__ import division

# Standard library
import sys
import os
import cPickle as pickle

# Third-party
#import apwlib.convert as c
import apwlib.geometry as g
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pyfits as pf
from sqlalchemy import func

# PTF
from ptf.parameters import *
from ptf.db.DatabaseConnection import *
import ptf.simulation.util as simu
from ptf import PTFLightCurve
import coverageplots
import detectionefficiency as de

title_font_size = 38
label_font_size = 34
tick_font_size = 24

parameter_to_label = {"j" : "J", "k" : "K", "sigma_mu" : r"$\sigma/\mu$", "eta" : r"$\eta$", "delta_chi_squared" : r"$\Delta \chi^2$"}

def survey_coverage():

    # PTF:
    raw_field_data = pf.open("data/exposureData.fits")[1].data
    unq_field_ids = np.unique(raw_field_data.field_id)
    
    ptf_fields = []
    for field_id in unq_field_ids:
        one_field_data = raw_field_data[raw_field_data.field_id == field_id]
        mean_ra = np.mean(one_field_data.ra) / 15.
        mean_dec = np.mean(one_field_data.dec)
        observations = len(one_field_data) / len(np.unique(one_field_data.ccd_id))
        
        ptf_fields.append(coverageplots.PTFField(mean_ra, mean_dec, id=field_id, number_of_observations=observations))
    
    # OGLE:
    high_cadence = np.genfromtxt("data/ogle4_common.txt", names=["ra","dec","l","b"], usecols=[6,7,8,9]).view(np.recarray)
    low_cadence = np.genfromtxt("data/ogle4_less_frequent.txt", names=["ra","dec","l","b"], usecols=[6,7,8,9]).view(np.recarray)
    
    ogle_high_cadence_fields = []
    for row in high_cadence: ogle_high_cadence_fields.append(coverageplots.OGLEField(row["ra"], row["dec"]))
    
    ogle_low_cadence_fields = []
    for row in low_cadence: ogle_low_cadence_fields.append(coverageplots.OGLEField(row["ra"], row["dec"]))
    
    coverage_plot = coverageplots.PTFCoveragePlot(figsize=(30,15), projection="aitoff")
    coverage_plot.addFields(ptf_fields, label="PTF", color_by_observations=True)
    coverage_plot.addFields(ogle_low_cadence_fields + ogle_high_cadence_fields, label="OGLE-IV", color="c", alpha=0.15)
    #coverage_plot.addFields(ogle_high_cadence_fields, label="OGLE-IV - high cadence", color="r", alpha=0.15)
    
    # Now I need to add globular and open clusters to the plot!
    open_clusters = np.genfromtxt("data/open_clusters.csv", usecols=[0,1,2,11], dtype=[("name","|S20"),("ra","|S8"), ("dec","|S9"), ("diameter", float)], delimiter=",").view(np.recarray)
    for ii,cluster in enumerate(open_clusters):
        ra_deg = g.RA(cluster["ra"]).degrees
        dec_rad = g.Dec(cluster["dec"]).radians
        if ii == 0:
            circle = matplotlib.patches.Ellipse((np.radians(-ra_deg+180), dec_rad), width=np.radians(cluster["diameter"]/60./np.cos(dec_rad)), height=np.radians(cluster["diameter"]/60.), alpha=.4, edgecolor="r", facecolor="r", label="Open Clusters")
        else:
            circle = matplotlib.patches.Ellipse((np.radians(-ra_deg+180), dec_rad), width=np.radians(cluster["diameter"]/60./np.cos(dec_rad)), height=np.radians(cluster["diameter"]/60.), alpha=.4, edgecolor="r", facecolor="r")
        coverage_plot.axis.add_patch(circle)
    
    """
    globular_clusters = np.genfromtxt("data/allGlobularClusters.txt", dtype=[("r_h",float),("ra","|S11"), ("dec","|S12")], delimiter=",").view(np.recarray)
    for ii,cluster in enumerate(globular_clusters):
        ra_deg = g.RA(cluster["ra"]).degrees
        dec_rad = g.Dec(cluster["dec"]).radians
        diameter = 2.*cluster["r_h"]*10
        
        if ii == 0:
            circle = matplotlib.patches.Ellipse((np.radians(-ra_deg+180), dec_rad), width=np.radians(diameter/60./np.cos(dec_rad)), height=np.radians(diameter/60.), alpha=.4, edgecolor="g", facecolor="g", label="Globular Clusters")
        else:
            circle = matplotlib.patches.Ellipse((np.radians(-ra_deg+180), dec_rad), width=np.radians(diameter/60./np.cos(dec_rad)), height=np.radians(diameter/60.), alpha=.4, edgecolor="g", facecolor="g")
        coverage_plot.axis.add_patch(circle)
    """
    
    coverage_plot.addLegend()
    coverage_plot.title.set_fontsize(title_font_size)
    legendtext = coverage_plot.legend.get_texts()
    plt.setp(legendtext, fontsize=label_font_size)    # the legend text fontsize
    
    #plt.show()
    coverage_plot.figure.savefig("plots/aas_ptf_coverage.png")

# To be used by the Praesepe timescale distribution plot and the 
#   detection efficiency
timescale_bins = np.logspace(np.log10(1), np.log10(1000), 100) # from 1 day to 1000 days

def praesepe_timescale_distribution():
    
    # TODO: get file from Amanda, put in data/ and put the correct filename below
    filename = "data/praesepeTimeScales.npy"
    timescales = np.load(filename)
    
    plt.hist(timescales, bins=timescale_bins)
    plt.xscale("log")
    plt.show()

def praesepe_event_rate():

    # TODO: get file from Amanda, put in data/ and put the correct filename below
    filename = "data/praesepeTimeScales.npy"
    timescales = np.load(filename)
    
    # TODO: get the global Praesepe event rate from Amanda
    global_event_rate = 0.01 #??
    
    # To get the event rate distribution, I have to normalize the timescale dist. so
    #   the integral from 0 to infinity = global event rate
    timescale_pdf, bin_edges = np.histogram(timescales, bins=timescale_bins, density=True)
    event_rate_distribution = timescale_pdf*global_event_rate
    
    # Get the Praesepe detection efficiency
    filename = "data/praesepe_detection_efficiency.npy"
    
    # Load the simulation results
    sim_results = np.load(filename)
        
    # Get the RMS scatter of delta chi-squared for the vanilla light curves
    dcs = [x[0] for x in session.query(VariabilityIndices.delta_chi_squared).join(LightCurve).filter(LightCurve.objid < 100000).all()]
    sigma = np.std(dcs)
    # 2*sigma ~ 300
    
    sim_results[np.isnan(sim_results["tE"])] = 0.
    
    detections = sim_results[sim_results["delta_chi_squared"] > 2.*sigma]
    detections = detections[detections["event_added"] == True]

    #detections = sim_results[(sim_results["delta_chi_squared"] > 2.*sigma)]
    #detections = detections[detections["event_added"] == True]
    
    tE_counts, tE_bin_edges = np.histogram(detections["tE"], bins=timescale_bins)
    total_counts, bin_edges = np.histogram(sim_results[sim_results["event_added"] == True]["tE"], bins=timescale_bins)
    detection_efficiency_distribution = tE_counts / total_counts
    bin_widths = bin_edges[1:] - bin_edges[:-1]
    
    # Compute number of events!
    #   - detection_efficiency_distribution is dE/dt_E
    #   - event_rate_distribution is dN/dt_E
    #   - bin_widths give us dt_E
    N_exp = np.sum(detection_efficiency_distribution * event_rate_distribution / 365. * bin_widths) * 102. # days of Praesepe obs.
    
    # Number of events if we had observed it consistently for 3 years
    N_exp_all_survey = np.sum(detection_efficiency_distribution * event_rate_distribution / 365. * bin_widths) * 1095. # days of Praesepe obs.
    
    print "Number of events in our Praesepe sample (102 days):", N_exp
    print "Number of events if we had observed it consistently for 3 years:", N_exp_all_survey

def praesepe_detection_efficiency():
    filename = "data/praesepe_detection_efficiency.npy"
    if not os.path.exists(filename):
        # Select out just the Praesepe light curves (objid < 100000)
        light_curve_generator = de.AllPraesepeLightCurves(limit=10000, random=True)

        sim_results = de.run_simulation(light_curve_generator, N=100)
        
        np.save(filename, sim_results)
    
    # Load the simulation results
    sim_results = np.load(filename)
        
    # Get the RMS scatter of delta chi-squared for the vanilla light curves
    dcs = [x[0] for x in session.query(VariabilityIndices.delta_chi_squared).join(LightCurve).filter(LightCurve.objid < 100000).all()]
    sigma = np.std(dcs)
    # 2*sigma ~ 300
    
    sim_results[np.isnan(sim_results["tE"])] = 0.
    
    detections = sim_results[sim_results["delta_chi_squared"] > 2.*sigma]
    detections = detections[detections["event_added"] == True]

    #detections = sim_results[(sim_results["delta_chi_squared"] > 2.*sigma)]
    #detections = detections[detections["event_added"] == True]
    
    tE_counts, tE_bin_edges = np.histogram(detections["tE"], bins=timescale_bins)
    total_counts, total_bin_edges = np.histogram(sim_results[sim_results["event_added"] == True]["tE"], bins=timescale_bins)
    detection_efficiency = tE_counts / total_counts
    bin_widths = total_bin_edges[1:] - total_bin_edges[:-1]
    #print np.sum(bin_widths*detection_efficiency)
    #return
    
    plt.figure(figsize=(15,15))
    # Multiply by 2 because we only put events in 50% of the cases
    plt.semilogx((total_bin_edges[1:]+total_bin_edges[:-1])/2, tE_counts / total_counts, 'k-', lw=3)
    plt.xlabel(r"$t_E$ [days]", size=label_font_size)
    plt.ylabel(r"Detection Efficiency $\mathcal{E}(t_E)$", size=label_font_size)
    plt.ylim(0., 0.75)
    t = plt.title("PTF Detection Efficiency for Praesepe Light Curves", size=title_font_size)
    t.set_y(1.04)
    
    # Change tick label size
    ax = plt.gca()
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(tick_font_size)
    
    plt.tight_layout()
    plt.savefig("plots/aas_praesepe_detection_efficiency.png")

def random_praesepe_light_curve():
    objid = np.random.randint(82415)
    try:
        lc = session.query(LightCurve).filter(LightCurve.objid == objid).one()
    except:
        lc = session.query(LightCurve).filter(LightCurve.objid == 101).one()
    
    return lc
    
def survey_detection_effieciency():
    """ Here I want a figure that shows how the detection efficiency
        changes for uniform, random, and clumpy observations
    """
    baseline = 365 #days
    # TODO: Rerun with 1024, add praesepe line
    max_num_observations = 1024
    min_num_observations = 16
    num_clumps = 4
    num_iterations = 10000
    
    if not os.path.exists("data/aas_survey_detection_efficiency.pickle"):
        data_dict = {"clumpy" : {1. : [], 10. : [], 100 : []}, "uniform" : {1. : [], 10. : [], 100 : []}}
        
        for timescale in [1., 10., 100.]:
            for sampling in ["clumpy", "uniform"]: #, "random"]:
                if sampling == "random":
                    mjd = np.random.random(max_num_observations)*baseline
                elif sampling == "clumpy":
                    sparse_samples = np.random.random(max_num_observations/2)*baseline
                    
                    clumps = []
                    days = []
                    sum = 0.
                    pts_per_clump = max_num_observations / 2 / num_clumps
                    for ii in range(num_clumps):
                        day = np.random.randint(365)
                        if day in days: continue
                        
                        days.append(day)
                        clumpy_samples = np.linspace(day+0.1, day+0.6, pts_per_clump)
                        clumps.append(clumpy_samples)
        
                    clumps.append(sparse_samples)
                    mjd = np.concatenate(tuple(clumps))
                    
                    plt.plot(mjd, [1.]*len(mjd), 'ro', alpha=0.4)
                    plt.show()
                
                elif sampling == "uniform":
                    mjd = np.linspace(0., baseline, max_num_observations)
                
                for jj in range(num_iterations):
                    lc = random_praesepe_light_curve()
                    if len(lc.mag) < 100: continue
                    
                    dupe_mags = np.array(lc.mag*15)
                    dupe_err = np.array(list(lc.error)*15)
                    
                    shuffled_idx = np.arange(0, len(dupe_mags))
                    np.random.shuffle(shuffled_idx)
        
                    mags = dupe_mags[shuffled_idx]
                    err = dupe_err[shuffled_idx]
                    
                    sim_light_curve = simu.SimulatedLightCurve(mjd=mjd, mag=mags[:len(mjd)], error=err[:len(mjd)])
                    sim_light_curve.addMicrolensingEvent(tE=timescale)
                    #sim_light_curve.plot()
                    
                    delta_chi_squareds = []
                    sim_mjd = sim_light_curve.mjd
                    sim_mag = sim_light_curve.mag
                    sim_err = sim_light_curve.error
                    while True:
                        if len(sim_mjd) < min_num_observations: break
                        
                        dcs = simu.compute_delta_chi_squared((sim_mjd, sim_mag, sim_err), force_fit=True)
                        delta_chi_squareds.append(dcs)
                        
                        prune = np.arange(len(sim_mjd))
                        np.random.shuffle(prune)
                        prune = prune[::2]
                        sim_mjd = sim_mjd[prune]
                        sim_mag = sim_mag[prune]
                        sim_err = sim_err[prune]
                        
                    data_dict[sampling][timescale].append(delta_chi_squareds)
        
        f = open("data/aas_survey_detection_efficiency.pickle", "w")
        pickle.dump(data_dict, f)
        f.close()
    
    f = open("data/aas_survey_detection_efficiency.pickle", "r")
    data_dict = pickle.load(f)
    
    # Plotting stuff
    plt.figure(figsize=(15,15))
    dcs_cutoff = 300.
    num_observations = [2**x for x in range(int(np.log2(max_num_observations)), int(np.log2(min_num_observations))-1, -1)]
    linestyles = {"uniform" : "--", "clumpy" : "-"}
    linecolors = {1. : "k", 10. : "r", 100. : "c"}
    for sampling in data_dict.keys():
        for timescale in data_dict[sampling].keys():
            data = np.array(data_dict[sampling][timescale])
            
            efficiencies = []
            for col,num_obs in enumerate(num_observations):
                efficiencies.append(np.sum(data[:,col] > dcs_cutoff) / len(data[:,col]))
            
            plt.plot(np.log2(num_observations), efficiencies, ls=linestyles[sampling], color=linecolors[timescale], label=r"$t_E={}$ day, {} sampling".format(int(timescale), sampling), lw=3)
    
    #plt.axvline(np.log2(625.), c="g", ls="--", lw=2, label="PTF Praesepe fields")
    
    plt.xlabel("Number of Observations / 1 year", size=label_font_size)
    plt.ylabel(r"Detection Efficiency $\mathcal{E}(t_E)$", size=label_font_size)
    plt.title("Simulated Detection Efficiency for\nDifferent Sampling Patterns", size=title_font_size)
    
    # Change tick label size
    ax = plt.gca()
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(tick_font_size)
    
    ax.set_xticklabels(num_observations[::-1])
    
    legend = plt.legend(loc="upper left", shadow=True, fancybox=True)
    legendtext  = legend.get_texts()
    plt.setp(legendtext, fontsize=tick_font_size)    # the legend text fontsize
    plt.tight_layout()
    #plt.show()
    plt.savefig("plots/aas_survey_detection_efficiency.png")

def variability_indices():
    import PraesepeLightCurves as plc
    # TODO: Sample timescale from the distribution that Amanda will send me
    # TODO: Fix legend in post-editing
    plc.aas_figure()

def variability_indices_detection_efficiency():
    """ This figure should show the detection efficiency curve for the Praesepe 
        data for each variability index (cut at 2-sigma)
    """
    filename = "data/praesepe_detection_efficiency.npy"
    # Load the simulation results
    sim_results = np.load(filename)
    
    var_indices = session.query(VariabilityIndices).join(LightCurve).filter(LightCurve.objid < 100000).all()
    
    styles = [(3,"-."), (3,":"), (3,"--"), (1.5,"--"), (2,"-")]
    colors = ["c", "m", "g", "y", "k"]
    plt.figure(figsize=(15,15))
    for ii,idx in enumerate(["j", "k", "eta", "sigma_mu", "delta_chi_squared"]):
        values = [getattr(x, idx) for x in var_indices]
        sigma = np.std(values)
        mu = np.mean(values)
        
        sim_results[np.isnan(sim_results["tE"])] = 0.
        
        detections = sim_results[(np.fabs(sim_results[idx]) > (mu + 2.*sigma)) | (np.fabs(sim_results[idx]) < (mu - 2.*sigma))]
        detections = detections[detections["event_added"] == True]
        
        tE_counts, tE_bin_edges = np.histogram(detections["tE"], bins=timescale_bins)
        total_counts, total_bin_edges = np.histogram(sim_results[sim_results["event_added"] == True]["tE"], bins=timescale_bins)
        
        lw,ls = styles[ii]
        plt.semilogx((total_bin_edges[1:]+total_bin_edges[:-1])/2, tE_counts / total_counts, c=colors[ii], lw=lw, label=r"{}".format(parameter_to_label[idx]), ls=ls)
        
    plt.xlabel(r"$t_E$ [days]", size=label_font_size)
    plt.ylabel(r"Detection Efficiency $\mathcal{E}(t_E)$", size=label_font_size)
    plt.ylim(0., 1.0)
    t = plt.title("PTF Detection Efficiency for Praesepe Light Curves", size=title_font_size)
    t.set_y(1.04)
    
    # Change tick label size
    ax = plt.gca()
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(tick_font_size)
    
    leg = plt.legend(shadow=True, fancybox=True)
    legendtext = leg.get_texts()
    plt.setp(legendtext, fontsize=label_font_size)
    plt.tight_layout()
    plt.savefig("plots/aas_var_indices_detection_efficiency.png")
    #plt.show()    

def systematics_552():
    # Bad1 http://kanaloa.ipac.caltech.edu/ibe/data/ptf/dev/process/proc/2010/04/17/f2/c6/p13/v1/PTF_201004172696_i_p_scie_t062817_u011575385_f02_p110002_c06.fits?center=127.403,19.6476deg&size=100px
    # Bad2 http://kanaloa.ipac.caltech.edu/ibe/data/ptf/dev/process/proc/2010/04/17/f2/c6/p13/v1/PTF_201004173180_i_p_scie_t073759_u011575280_f02_p110002_c06.fits?center=127.403,19.6476deg&size=100px
    # Good http://kanaloa.ipac.caltech.edu/ibe/data/ptf/dev/process/proc/2010/04/07/f2/c6/p13/v1/PTF_201004071439_i_p_scie_t032714_u011539562_f02_p110002_c06.fits?center=127.403,19.6476deg&size=100px
    
    mjd_offset = 54832
    lc1 = session.query(LightCurve).filter(LightCurve.objid == 552).one()
    bad_light_curve = PTFLightCurve.fromDBLightCurve(lc1)
    light_curves = session.query(LightCurve).filter(func.q3c_radial_query(LightCurve.ra, LightCurve.dec, lc1.ra, lc1.dec,30/3600.)).all()
    
    bad_obs1 = 55303.26964
    bad_obs2 = 55303.31804
    good_obs = 55293.14391
    
    # seeing, airmass, filename, mjd
    imlist = np.genfromtxt("data/aas_552_imagelist.txt", skiprows=4, usecols=[11,12,20,25], dtype=[("seeing", float), ("airmass", float), ("filename", "|S100"), ("mjd", float)])
    idx_sort = np.argsort(imlist["mjd"])
    imlist = imlist[idx_sort]
    
    print "Bad1:", imlist["filename"][imlist["mjd"] == 55303.26964]
    print "Bad2:", imlist["filename"][imlist["mjd"] == 55303.31804]
    print "Good:", imlist["filename"][imlist["mjd"] == 55293.14391]
    return
    
    plt.plot(imlist["mjd"], imlist["seeing"], "r.")
    plt.show()

def systematics_9347():
    # http://kanaloa.ipac.caltech.edu/ibe/search/ptf/dev/process?POS=129.568,19.6232
    # one http://kanaloa.ipac.caltech.edu/ibe/data/ptf/dev/process/proc/2010/05/15/f2/c6/p13/v1/PTF_201005152355_i_p_scie_t053906_u011486277_f02_p110004_c06.fits?center=129.568,19.6232deg&size=150px
    # two http://kanaloa.ipac.caltech.edu/ibe/data/ptf/dev/process/proc/2010/04/25/f2/c6/p13/v1/PTF_201004251929_i_p_scie_t043750_u011578017_f02_p110004_c06.fits?center=129.568,19.6232deg&size=150px
    
    mjd_offset = 54832
    lc1 = session.query(LightCurve).filter(LightCurve.objid == 9347).one()
    bad_light_curve = PTFLightCurve.fromDBLightCurve(lc1)
    light_curves = session.query(LightCurve).filter(func.q3c_radial_query(LightCurve.ra, LightCurve.dec, lc1.ra, lc1.dec,30/3600.)).all()
    
    bad_light_curve.plot()
    
    print [x.ra for x in light_curves]
    print [x.dec for x in light_curves]
    
    print lc1.ra, lc1.dec
    return
    
if __name__ == "__main__":
    #survey_coverage()
    #praesepe_detection_efficiency()
    survey_detection_effieciency()
    #variability_indices()
    #variability_indices_detection_efficiency()
    #praesepe_timescale_distribution()
    #praesepe_event_rate()