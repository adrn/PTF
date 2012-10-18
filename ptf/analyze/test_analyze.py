
import numpy as np
import matplotlib.pyplot as plt
from ptf.simulation.simulatedlightcurve import SimulatedLightCurve

import analyze

def test_fit_microlensing_event():
    true_params = {"u0" : 0.3, "t0" : 100., "tE" : 20., "m0" : 15.}
    
    light_curve = SimulatedLightCurve(mjd=np.linspace(0., 200., 300.), mag=15., error=[0.1])
    light_curve.addMicrolensingEvent(**true_params)
    
    params = analyze.fit_microlensing_event(light_curve)
    plt.errorbar(light_curve.mjd, light_curve.mag, light_curve.error, color="k", marker=".")
    plt.plot(light_curve.mjd, analyze.microlensing_model(params, light_curve.mjd), "r-")
    plt.save()

test_fit_microlensing_event()