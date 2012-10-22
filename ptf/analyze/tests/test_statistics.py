def test_estimate_continuum():
    import matplotlib.pyplot as plt
    
    np.random.seed(1)
    # Generate a flat light curve and test continuum fit on flat
    #   light curve
    for truth in np.random.random(100)*100:
        mjds = np.linspace(0., 100., 100)
        sigmas = np.zeros(len(mjds)) + 0.1
        mags = truth*np.ones(len(mjds)) + np.random.normal(0., sigmas)
        mags += np.exp(-(mjds-50)**2/(2.*5**2))
     
        from ptf.ptflightcurve import PTFLightCurve
        lc = PTFLightCurve(mjds, mags, sigmas)
        
        popt, sigma = estimate_continuum(lc, sigma_clip=False)
        popt_clipped, sigma_clipped = estimate_continuum(lc, sigma_clip=True)
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(np.linspace(0., 100., 100.), [popt[0]]*100, 'r--', label="not clipped")
        ax.plot(np.linspace(0., 100., 100.), [popt_clipped[0]]*100, 'b--', label="clipped")
        ax.plot(np.linspace(0., 100., 100.), [truth]*100, 'g-')
        ax = lc.plot(ax)
        ax.legend()
        plt.show()

def test_gaussian_constant_delta_chi_squared():
    import matplotlib.pyplot as plt
    u0s = np.logspace(-3, 0.127, 20)
    dcs = []
    for u0 in u0s:
        lc = slc.SimulatedLightCurve(mjd=np.linspace(0., 100., 60), mag=np.array([15.]*60), error=np.array([1.0]*60))
        lc.addMicrolensingEvent(u0=u0, t0=50., tE=10.)
        dcs.append(gaussian_constant_delta_chi_squared(lc))
    
    plt.clf()
    plt.semilogx(u0s, dcs, "ko")
    plt.savefig("plots/derp.png")

def test_stetson_j():
    import copy
    from ptf.simulation.simulatedlightcurve import SimulatedLightCurve
    mjd = np.linspace(0., 500., 150)
    sigmas = 10.**np.random.uniform(-2, -3, size=150)
    light_curve = SimulatedLightCurve(mjd, error=sigmas)
    lc2 = copy.copy(light_curve)
    
    # Make a flat light curve + microlensing event, compute J
    light_curve.addMicrolensingEvent(t0=250.)
    
    assert stetson_j(lc2) < stetson_j(light_curve)

def test_stetson_k():
    import copy
    from ptf.simulation.simulatedlightcurve import SimulatedLightCurve
    mjd = np.linspace(0., 500., 150)
    sigmas = 10.**np.random.uniform(-2, -3, size=150)
    light_curve = SimulatedLightCurve(mjd, error=sigmas)
    
    # Make a flat light curve + microlensing event, compute J
    light_curve.addMicrolensingEvent(t0=250.)
    
    assert 0.7 < stetson_k(light_curve) < 0.9

def test_eta():
    from ptf.simulation.simulatedlightcurve import SimulatedLightCurve
    mjd = np.linspace(0., 500., 150)
    sigmas = 10.**np.random.uniform(-2, -3, size=150)
    light_curve = SimulatedLightCurve(mjd, error=sigmas)
    
    print eta(light_curve)
    
    # Make a flat light curve + microlensing event, compute J
    light_curve.addMicrolensingEvent(t0=250.)
    
    print eta(light_curve)

def test_compute_variability_indices():
    # Here we're really just seeing how long it takes to run...
    
    from ptf.simulation.simulatedlightcurve import SimulatedLightCurve
    mjd = np.linspace(0., 500., 200)
    sigmas = 10.**np.random.uniform(-2, -3, size=200)
    light_curve = SimulatedLightCurve(mjd, error=sigmas)
    light_curve.addMicrolensingEvent()
    
    import time
    
    a = time.time()
    for ii in range(100):
        idx = compute_variability_indices(light_curve)
        
    print time.time() - a 
    