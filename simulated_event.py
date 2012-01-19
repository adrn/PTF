# m = -2.5*log10(F/F0)
#   R-band F0 = 2875 Jy or 2.25E-9
#   http://www.sr.bham.ac.uk/~somak/constants.html

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

G = 6.673E-11 # m^3 / kg / s^2
M = 1.4 * 1.989E30 #kg
c = 3.0E8 # m/s

Ds = 10. * 3.0856802E19 # kpc -> m
Dl = 4. * 3.0856802E19 # kpc -> m
mu_rel = 1. / 1000. / 60. / 60. # mas/yr -> deg/day
theta_E = np.degrees(np.sqrt((Ds-Dl) / (Ds*Dl) * 4.*G*M/c**2)) # deg
theta_E_mas = theta_E*60*60*1000.

#u0 = 0.5 # theta_E
t0 = 0.
t_E = 5. #theta_E / mu_rel

def RMagToFlux(R):
    # Returns a flux in Janskys
    return 2875.*10**(R/-2.5)

def FluxToRMag(f):
    # Accepts a flux in Janskys
    return -2.5*np.log10(f/2875.)

def u_t(t, u_0, t_0 , t_E):
    return np.sqrt(u_0**2 + ((t - t_0)/t_E)**2)

def A_u(u):
    return (u**2 + 2) / (u*np.sqrt(u**2 + 4))

def FLUXMODEL(t, u0, t0, tE, F0):
    return F0*A_u(u_t(t, u0, t0, tE))

def simulate_flux():
    t = np.arange(-5., 5, 0.01) # days
    
    fig = plt.figure(figsize=(7,14))
    plt.subplots_adjust(hspace=0.2, left=0.15)
    
    ax1 = fig.add_subplot(211)
    cir = Circle((0., 0.), radius=theta_E_mas, fill=False, ec='r', ls="dashed")
    ax1.add_patch(cir)
    ax1.set_xlim(-theta_E_mas*2., theta_E_mas*2.)
    ax1.set_ylim(-theta_E_mas*2., theta_E_mas*2.)
    ax1.set_xlabel(r"$\theta/\theta_E$", size=28)
    ax1.set_ylabel(r"$\theta/\theta_E$", size=28)
    ax1.plot([0], [0], 'ko')
    
    ax2 = fig.add_subplot(212)
    for u0,ls,lw in [(0.01, 'solid', 1.5), (0.1, 'dashed', 1.5), (0.2, 'dashdot', 1.5), (0.3, 'dotted', 1.5), \
                     (0.4, 'solid', 0.5), (0.5, 'dashed', 0.5), (0.6, 'dashdot', 0.5), (0.7, 'dotted', 0.5)]:
        ax1.plot([-theta_E_mas*2., theta_E_mas*2.], [u0*theta_E_mas, u0*theta_E_mas], ls=ls, lw=lw, c='k')
        ax2.semilogy(t, A_u(u_t(t, u0)), c='k', ls=ls, lw=lw, label=r"$u_0=%.2f$" % u0)
    ax2.set_xlabel(r"$(t-t_0)/t_E$", size=28)
    ax2.set_ylabel(r"Amplification", size=28)
    
    ax2.legend()
    #fig.suptitle("Magnification curves for a point-mass microlensing event for various impact parameters.")
    
    plt.show()

def simulate_event(t, mag, magErr, true_params):
    u0, t0, tE, F0 = true_params
    
    if mag == None:
        t = np.sort(t)
        
        #Ft = F0*A_u(u_t(t, u0, t0, tE)) + np.random.normal(0., 0.03*F0, size=len(t))
        Ft = FLUXMODEL(t, u0, t0, tE, F0) + np.random.normal(0., 0.03*F0, size=len(t))
        mag = FluxToRMag(Ft)
        
        results = []
        for ii in range(len(Ft)):
            results.append((t[ii], mag[ii], np.random.normal(0.005, 0.01)))
        
        return np.array(results, dtype=[('t', float), ('mag', float), ('sigma', float)]).view(np.recarray)
    else:
        newMag = FluxToRMag(FLUXMODEL(t, u0, t0, tE, RMagToFlux(mag)))
        results = []
        for ii in range(len(newMag)):
            results.append((t[ii], newMag[ii], magErr[ii]))
        return np.array(results, dtype=[('t', float), ('mag', float), ('sigma', float)]).view(np.recarray)
    
def plot_simulated_event():
    
    model_t = np.arange(55000., 55800, 0.2) # days
    real_t = np.array([55531.1495, 55531.105230000001, 55478.201739999997, 55478.246010000003, 55527.188240000003, 55527.232660000001, 55500.512499999997, 55500.468760000003, 55395.462339999998, 55433.379979999998, 55433.3361, 55151.258459999997, 55151.30212, 55562.112959999999, 55380.475879999998, 55473.258370000003, 55473.211969999997, 55734.460830000004, 55734.461960000001, 55120.443149999999, 55120.48792, 55537.196150000003, 55537.240059999996, 55415.387329999998, 55415.343070000003, 55181.127220000002, 55378.451500000003, 55578.165910000003, 55578.122309999999, 55481.329239999999, 55481.285580000003, 55506.460729999999, 55506.503530000002, 55779.340409999997, 55779.397360000003, 55468.490100000003, 55419.405010000002, 55419.361700000001, 55590.195, 55590.23878, 55582.249730000003, 55582.206310000001, 55516.275289999998, 55516.31856, 55399.445030000003, 55399.488740000001, 55423.385799999996, 55423.44427, 55745.453659999999, 55741.445930000002, 55392.482530000001, 55392.43909, 55586.244509999997, 55209.216160000004, 55209.260739999998, 55497.445460000003, 55497.401599999997, 55462.401830000003, 55462.357909999999, 55396.390780000002, 55396.43434, 55514.262519999997, 55514.307139999997, 55776.411119999997, 55776.351589999998, 55451.348100000003, 55451.392119999997, 55400.440589999998, 55400.48416, 55458.422359999997, 55458.344250000002, 55563.105239999997, 55594.179900000003, 55594.136489999997, 55753.456259999999, 55753.411590000003, 55470.359830000001, 55470.316769999998, 55183.09102, 55183.134749999997, 55391.410969999997, 55391.475830000003, 55752.424529999997, 55752.474560000002, 55168.147579999997, 55168.191149999999, 55443.374649999998, 55443.331760000001, 55381.471160000001, 55381.426379999997, 55736.460650000001, 55736.45478, 55447.462330000002, 55447.417909999996, 55416.38233, 55416.338680000001, 55175.133500000004, 55175.177530000001, 55769.369839999999, 55769.413979999998, 55389.484129999997, 55482.303599999999, 55482.352250000004, 55579.11709, 55579.16027, 55420.402770000001, 55420.359830000001, 55780.503579999997, 55742.444190000002, 55469.23792, 55469.194029999999, 55749.461900000002, 55455.22767, 55455.305249999998, 55424.44227, 55424.486440000001, 55746.479039999998, 55597.167269999998, 55431.387029999998, 55431.343000000001, 55777.358310000003, 55777.418940000003, 55203.289019999997, 55203.245369999997, 55428.347529999999, 55428.391839999997, 55750.458180000001, 55397.405500000001, 55397.449059999999, 55530.076979999998, 55763.408080000001, 55763.452929999999, 55432.381820000002, 55432.338309999999, 55529.221720000001, 55434.439980000003, 55434.396439999997, 55387.409180000002, 55387.478629999998, 55533.217129999997, 55533.260730000002, 55535.376880000003, 55535.420299999998, 55436.503360000002, 55436.459730000002, 55459.37444, 55459.418460000001, 55504.412799999998, 55504.311979999999, 55770.383589999998, 55770.426899999999, 55409.442289999999, 55409.390639999998, 55461.393400000001, 55461.436930000003, 55250.151010000001, 55444.407729999999, 55444.355669999997, 55517.363599999997, 55517.317669999997, 55435.41762, 55435.462039999999, 55518.331230000003, 55518.375970000001, 55541.140209999998, 55541.096689999998, 55760.417479999996, 55479.257120000002, 55479.208769999997, 55544.117769999997, 55544.162349999999, 55386.457090000004, 55386.413430000001, 55483.307209999999, 55483.358829999997, 55207.307050000003, 55207.262260000003, 55421.391839999997, 55421.43707, 55452.408920000002, 55452.36591, 55774.39718, 55774.35413, 55743.48171, 55456.37328, 55456.329709999998, 55584.179759999999, 55584.136250000003, 55463.355470000002, 55463.311710000002, 55778.352070000001, 55778.396419999997, 55445.362370000003, 55445.405709999999, 55425.497360000001, 55425.453540000002, 55580.235780000003, 55580.191919999997, 55508.241889999998, 55508.162369999998, 55460.372799999997, 55460.4159, 55430.387479999998, 55430.344140000001, 55429.390959999997, 55429.343679999998, 55394.395340000003, 55114.412409999997, 55114.456879999998, 55486.214059999998, 55486.25866, 55526.163650000002, 55526.207970000003, 55427.344389999998, 55427.300669999997, 55401.456619999997, 55477.214390000001, 55477.170789999996, 55607.11937, 55532.178, 55532.220379999999, 55561.269379999998, 55561.225720000002, 55605.167009999997, 55605.211909999998, 55454.276570000002, 55454.233699999997, 55567.109089999998, 55759.481540000001, 55513.257310000001, 55513.301189999998, 55251.123630000002, 55251.167659999999, 55243.135799999996, 55243.186309999997, 55377.465210000002, 55498.458070000001, 55498.404860000002, 55572.115519999999, 55441.263800000001, 55441.32228, 55538.235139999997, 55538.19167, 55379.451849999998, 55587.199059999999, 55587.242859999998, 55543.161379999998, 55484.427470000002, 55484.471019999997, 55588.281320000002, 55588.237869999997, 55414.386500000001, 55414.343509999999, 55189.12657, 55189.082880000002, 55480.280910000001, 55480.216979999997, 55127.388659999997, 55127.344810000002, 55418.366040000001, 55418.409460000003, 55449.343289999997, 55449.387750000002, 55398.451959999999, 55398.40814, 55453.406969999996, 55515.318370000001, 55515.266340000002, 55390.414929999999, 55390.458500000001, 55142.21615, 55142.260249999999, 55446.41087, 55446.366869999998, 55422.388830000004, 55422.432500000003, 55417.347659999999, 55417.391770000002, 55135.387860000003, 55135.343999999997, 55457.324999999997, 55457.368040000001, 55450.340640000002, 55450.383560000002, 55585.139560000003, 55585.183299999997, 55426.493340000001, 55748.439200000001, 55144.288670000002, 55144.332670000003, 55388.463170000003, 55744.455900000001, 55775.461320000002, 55558.252589999996, 55756.411780000002, 55756.463470000002, 55771.417379999999, 55771.373460000003, 55442.28701, 55442.330950000003, 55472.251380000002, 55472.208379999996])
    t = np.sort(real_t)
    #real_mag = np.array([11.157999999999999, 11.028, 11.202, 11.034000000000001, 10.715, 10.141, 10.109999999999999, 10.23, 11.003, 11.116, 10.831, 10.228, 10.512, 11.003, 9.8179999999999996, 10.035, 10.018000000000001, 11.48, 11.301, 11.573, 11.411, 11.313000000000001, 10.401, 10.273, 10.548, 10.523999999999999, 9.0269999999999992, 10.628, 10.917, 9.9280000000000008, 9.8209999999999997, 11.445, 11.595000000000001, 11.481999999999999, 10.935, 10.848000000000001, 10.254, 10.474, 12.093999999999999, 9.1769999999999996, 9.1799999999999997, 11.17, 11.041, 12.170999999999999, 9.6120000000000001, 9.6150000000000002, 9.6880000000000006, 9.6400000000000006, 10.631, 10.634, 10.124000000000001, 10.109999999999999, 9.6110000000000007, 11.369999999999999, 11.765000000000001, 12.074999999999999, 11.465, 11.456, 11.273, 10.138999999999999, 10.132999999999999, 11.6, 11.519, 11.071999999999999, 11.064, 10.859, 10.975, 11.143000000000001, 11.52, 11.279999999999999, 10.098000000000001, 10.106999999999999, 9.5380000000000003, 9.4459999999999997, 11.457000000000001, 11.345000000000001, 11.898, 11.173999999999999, 12.398, 11.564, 12.401, 12.653, 9.5129999999999999, 9.6470000000000002, 11.412000000000001, 11.180999999999999, 11.461, 11.503, 10.237, 10.093999999999999, 12.195, 12.298, 10.214, 9.8529999999999998, 11.651, 11.489000000000001, 11.803000000000001, 11.519, 11.198, 11.167, 9.6669999999999998, 9.6600000000000001, 11.590999999999999, 11.577, 10.224, 10.122999999999999, 9.4000000000000004, 13.010999999999999, 10.968, 11.476000000000001, 12.212999999999999, 11.234999999999999, 11.403, 11.278, 11.407, 11.846, 10.667999999999999, 10.680999999999999, 11.071999999999999, 11.079000000000001, 12.287000000000001, 11.619, 11.547000000000001, 9.8040000000000003, 9.9870000000000001, 11.212999999999999, 11.349, 11.592000000000001, 9.9510000000000005, 10.005000000000001, 11.26, 11.342000000000001, 10.506, 11.106999999999999, 11.010999999999999, 11.321, 11.06, 9.6549999999999994, 10.907999999999999, 9.4030000000000005, 9.2759999999999998, 10.382999999999999, 10.164999999999999, 11.105, 11.064, 11.228, 11.194000000000001, 10.971, 11.465999999999999, 11.324999999999999, 11.301, 10.621, 10.516999999999999, 11.177, 11.16, 10.631, 11.013, 11.114000000000001, 11.381, 11.390000000000001, 10.943, 11.009, 11.429, 10.829000000000001, 10.353, 11.035, 11.02, 11.178000000000001, 11.098000000000001, 10.426, 10.416, 9.7699999999999996, 9.7889999999999997, 11.388999999999999, 11.335000000000001, 11.592000000000001, 11.611000000000001, 10.555999999999999, 10.545, 11.48, 11.577, 11.108000000000001, 11.114000000000001, 12.682, 11.663, 11.518000000000001, 9.6069999999999993, 9.6150000000000002, 11.65, 11.422000000000001, 11.505000000000001, 11.771000000000001, 10.715, 10.608000000000001, 9.4689999999999994, 9.6010000000000009, 11.351000000000001, 11.428000000000001, 11.853999999999999, 10.805999999999999, 11.276, 11.173999999999999, 11.475, 11.427, 11.364000000000001, 11.295999999999999, 10.456, 10.035, 10.99, 10.316000000000001, 10.119999999999999, 11.515000000000001, 11.625, 11.111000000000001, 10.772, 10.068, 32.0, 11.547000000000001, 11.419, 9.9960000000000004, 9.9030000000000005, 10.542, 10.65, 9.7210000000000001, 9.7149999999999999, 9.8290000000000006, 10.692, 11.372999999999999, 11.125999999999999, 11.224, 10.811999999999999, 10.98, 11.464, 10.968999999999999, 10.945, 9.2520000000000007, 11.327, 11.409000000000001, 8.9860000000000007, 10.116, 11.132, 11.462999999999999, 10.531000000000001, 10.465, 9.3889999999999993, 9.3789999999999996, 11.398, 11.529999999999999, 10.449999999999999, 11.49, 11.6, 9.0239999999999991, 9.3819999999999997, 10.346, 10.191000000000001, 10.175000000000001, 10.368, 11.638999999999999, 11.452, 10.253, 10.364000000000001, 11.398999999999999, 11.507, 10.055, 10.003, 9.5760000000000005, 9.5020000000000007, 10.859999999999999, 11.016, 10.738, 10.577, 11.417999999999999, 11.404, 11.385999999999999, 9.8499999999999996, 10.177, 10.573, 10.648, 10.342000000000001, 10.675000000000001, 11.506, 11.525, 9.5869999999999997, 9.484, 10.727, 11.375999999999999, 11.324999999999999, 10.677, 10.821, 11.082000000000001, 12.109999999999999, 9.6259999999999994, 10.1, 11.387, 11.359999999999999, 12.544, 11.214, 11.076000000000001, 10.891, 11.343999999999999, 11.647, 11.401])
    
    # Uncertainties
    sigma_mu = 0.001 #np.random.uniform(0., 0.2, len(t))
    sigma_sigma = 0.001 # np.random.uniform(0., 0.01, len(t))
    sigma = np.fabs(np.random.normal(sigma_mu, sigma_sigma, len(t)))
    
    true_mag = M(t, *params)
    mag = true_mag + np.random.normal(0., sigma)
    
    #w = np.random.randint(0, len(t)-1, 25)
    w = np.arange(0, len(t)-1, 1, dtype=int)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.errorbar(t[w], mag[w], sigma[w], ls='none', marker='.')
    ax.plot(model_t, M(model_t, *params), 'r-', alpha=0.3)
    #ax.plot(real_t, real_mag, 'b.')
    ax.set_ylim(ax.get_ylim()[::-1])
    ax.set_xlabel("MJD")
    ax.set_ylabel(r"$M_R$")
    plt.show()

if __name__ == "__main__":
    np.random.seed(111)
    #plot_simulated_event()
    a = simulate_event(np.array([55531.1495, 55531.105230000001, 55478.201739999997, 55478.246010000003, 55527.188240000003, 55527.232660000001, 55500.512499999997, 55500.468760000003, 55395.462339999998, 55433.379979999998, 55433.3361, 55151.258459999997, 55151.30212, 55562.112959999999, 55380.475879999998, 55473.258370000003, 55473.211969999997, 55734.460830000004, 55734.461960000001, 55120.443149999999, 55120.48792, 55537.196150000003, 55537.240059999996, 55415.387329999998, 55415.343070000003, 55181.127220000002, 55378.451500000003, 55578.165910000003, 55578.122309999999, 55481.329239999999, 55481.285580000003, 55506.460729999999, 55506.503530000002, 55779.340409999997, 55779.397360000003, 55468.490100000003, 55419.405010000002, 55419.361700000001, 55590.195, 55590.23878, 55582.249730000003, 55582.206310000001, 55516.275289999998, 55516.31856, 55399.445030000003, 55399.488740000001, 55423.385799999996, 55423.44427, 55745.453659999999, 55741.445930000002, 55392.482530000001, 55392.43909, 55586.244509999997, 55209.216160000004, 55209.260739999998, 55497.445460000003, 55497.401599999997, 55462.401830000003, 55462.357909999999, 55396.390780000002, 55396.43434, 55514.262519999997, 55514.307139999997, 55776.411119999997, 55776.351589999998, 55451.348100000003, 55451.392119999997, 55400.440589999998, 55400.48416, 55458.422359999997, 55458.344250000002, 55563.105239999997, 55594.179900000003, 55594.136489999997, 55753.456259999999, 55753.411590000003, 55470.359830000001, 55470.316769999998, 55183.09102, 55183.134749999997, 55391.410969999997, 55391.475830000003, 55752.424529999997, 55752.474560000002, 55168.147579999997, 55168.191149999999, 55443.374649999998, 55443.331760000001, 55381.471160000001, 55381.426379999997, 55736.460650000001, 55736.45478, 55447.462330000002, 55447.417909999996, 55416.38233, 55416.338680000001, 55175.133500000004, 55175.177530000001, 55769.369839999999, 55769.413979999998, 55389.484129999997, 55482.303599999999, 55482.352250000004, 55579.11709, 55579.16027, 55420.402770000001, 55420.359830000001, 55780.503579999997, 55742.444190000002, 55469.23792, 55469.194029999999, 55749.461900000002, 55455.22767, 55455.305249999998, 55424.44227, 55424.486440000001, 55746.479039999998, 55597.167269999998, 55431.387029999998, 55431.343000000001, 55777.358310000003, 55777.418940000003, 55203.289019999997, 55203.245369999997, 55428.347529999999, 55428.391839999997, 55750.458180000001, 55397.405500000001, 55397.449059999999, 55530.076979999998, 55763.408080000001, 55763.452929999999, 55432.381820000002, 55432.338309999999, 55529.221720000001, 55434.439980000003, 55434.396439999997, 55387.409180000002, 55387.478629999998, 55533.217129999997, 55533.260730000002, 55535.376880000003, 55535.420299999998, 55436.503360000002, 55436.459730000002, 55459.37444, 55459.418460000001, 55504.412799999998, 55504.311979999999, 55770.383589999998, 55770.426899999999, 55409.442289999999, 55409.390639999998, 55461.393400000001, 55461.436930000003, 55250.151010000001, 55444.407729999999, 55444.355669999997, 55517.363599999997, 55517.317669999997, 55435.41762, 55435.462039999999, 55518.331230000003, 55518.375970000001, 55541.140209999998, 55541.096689999998, 55760.417479999996, 55479.257120000002, 55479.208769999997, 55544.117769999997, 55544.162349999999, 55386.457090000004, 55386.413430000001, 55483.307209999999, 55483.358829999997, 55207.307050000003, 55207.262260000003, 55421.391839999997, 55421.43707, 55452.408920000002, 55452.36591, 55774.39718, 55774.35413, 55743.48171, 55456.37328, 55456.329709999998, 55584.179759999999, 55584.136250000003, 55463.355470000002, 55463.311710000002, 55778.352070000001, 55778.396419999997, 55445.362370000003, 55445.405709999999, 55425.497360000001, 55425.453540000002, 55580.235780000003, 55580.191919999997, 55508.241889999998, 55508.162369999998, 55460.372799999997, 55460.4159, 55430.387479999998, 55430.344140000001, 55429.390959999997, 55429.343679999998, 55394.395340000003, 55114.412409999997, 55114.456879999998, 55486.214059999998, 55486.25866, 55526.163650000002, 55526.207970000003, 55427.344389999998, 55427.300669999997, 55401.456619999997, 55477.214390000001, 55477.170789999996, 55607.11937, 55532.178, 55532.220379999999, 55561.269379999998, 55561.225720000002, 55605.167009999997, 55605.211909999998, 55454.276570000002, 55454.233699999997, 55567.109089999998, 55759.481540000001, 55513.257310000001, 55513.301189999998, 55251.123630000002, 55251.167659999999, 55243.135799999996, 55243.186309999997, 55377.465210000002, 55498.458070000001, 55498.404860000002, 55572.115519999999, 55441.263800000001, 55441.32228, 55538.235139999997, 55538.19167, 55379.451849999998, 55587.199059999999, 55587.242859999998, 55543.161379999998, 55484.427470000002, 55484.471019999997, 55588.281320000002, 55588.237869999997, 55414.386500000001, 55414.343509999999, 55189.12657, 55189.082880000002, 55480.280910000001, 55480.216979999997, 55127.388659999997, 55127.344810000002, 55418.366040000001, 55418.409460000003, 55449.343289999997, 55449.387750000002, 55398.451959999999, 55398.40814, 55453.406969999996, 55515.318370000001, 55515.266340000002, 55390.414929999999, 55390.458500000001, 55142.21615, 55142.260249999999, 55446.41087, 55446.366869999998, 55422.388830000004, 55422.432500000003, 55417.347659999999, 55417.391770000002, 55135.387860000003, 55135.343999999997, 55457.324999999997, 55457.368040000001, 55450.340640000002, 55450.383560000002, 55585.139560000003, 55585.183299999997, 55426.493340000001, 55748.439200000001, 55144.288670000002, 55144.332670000003, 55388.463170000003, 55744.455900000001, 55775.461320000002, 55558.252589999996, 55756.411780000002, 55756.463470000002, 55771.417379999999, 55771.373460000003, 55442.28701, 55442.330950000003, 55472.251380000002, 55472.208379999996]))
    
    print a.t
    print a.mag
    
    