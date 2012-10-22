def test_ptffield():
    # Test PTFField
    
    field_110001 = PTFField(110001)
    ccd0 = field_110001.ccds[0]
    print "CCDS: {}".format(",".join([str(x) for x in field_110001.ccds.keys()]))
    chip = ccd0.read()
    print "Number of observations on CCD 0:", chip.sources.col("nobs")
    print "Number of GOOD observations on CCD 0:", chip.sources.col("ngoodobs")
    ccd0.close()
    
    print "Number of observations per ccd:"
    print field_110001.number_of_exposures
    print "Baseline per ccd:"
    print field_110001.baseline

def time_compute_var_indices():
    # Time computing variability indices for all light curves on a chip
    
    field = PTFField(110002)
    
    # Try to get light curve generator for a CCD
    import time
    a = time.time()
    N = 0
    for light_curve in field.ccds[0].light_curves("(ngoodobs > 15) & (matchedSourceID < 1000)"):
        indices = compute_variability_indices(light_curve, indices=["j","k","eta","sigma_mu","delta_chi_squared"])
        N += 1
    
    print "Took {} seconds to compute for {} light curves".format(time.time() - a, N)
    
def test_field():
    filter_R = Filter(filter_id="R")
    filter_g = Filter(filter_id="g")
    
    # Test all the various ways to initialize the object
    field = Field(field_id=110002, filter=filter_R)
    field = Field(field_id="110002", filter=filter_g)
    
    # Test number of exposures and baseline
    field = Field(field_id=110002, filter=filter_R)
    print field.number_of_exposures
    print field.baseline

def test_filter():
    filter = Filter(filter_id=2)
    filter = Filter(filter_id=1)
    
    filter = Filter(filter_id="R")
    filter = Filter(filter_id="g")