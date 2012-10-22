""" Note: Never finished writing this! This is a graveyard script... """

import numpy as np    
   
c =  299792458.0 # m s^-1
G = 6.67300E-11 # m^3 kg^-1 s^-2
   
def relative_proper_motion(star1, star2):
    # Compute relative proper motion between the given stars
    
    # TODO: This is wrong
    return star1.proper_motion - star2.proper_motion    

def relative_distance(star1, star2):
    # Compute the relative distance D_rel in parsecs
    
    return (star1.distance**-1 - star2.distance**-1)**-1

class Star:
    
    def __init__(self, mass, distance, proper_motion):
        """ Represents a star at a given distance with a given proper motion
            
            Parameters
            ----------
            mass : float, int
                A mass in solar masses
            distance : float, int
                A distance to the star in parsecs
            proper_motion : numpy.array, list, float, int
                A proper motion vector (PM_ra, PM_dec) in mas / yr
        
        """
        self.distance = distance*3.08568025E16 # meters
        self.mass = mass*1.98892E30 # kg
        
        if isinstance(proper_motion, list):
            self.proper_motion = np.array(proper_motion)
        else:
            self.proper_motion = proper_motion
    
    def relative_proper_motion(self, other_star):
        return relative_proper_motion(self, other_star)
    
    def relative_distance(self, other_star):
        return relative_distance(self, other_star)
    
class LensStar(Star):
    
    @property
    def schwarzschild_radius(self):
        return 2.*G*M / c**2
        
class SourceStar(Star):
    pass

class MicrolensingEvent:
    
    def __init__(self, lens_star, source_star):
        self.lens = lens_star
        self.source = source_star
    
    @property
    def crossing_time(self):
        mu_rel = relative_proper_motion(self.lens, self.source) # milliarcseconds / yr
        mu_rel_mag = np.sqrt(np.dot(mu_rel, mu_rel.T))
        mu_rel_mag *= 1.5373341E-16 # rad / s
        einstein_angle = np.sqrt(4.*G*self.lens.mass / c**2 * (self.source.distance - self.lens.distance) / (self.lens.distance * self.source.distance))
        
        return einstein_angle / mu_rel_mag # seconds

if __name__ == "__main__":
    lens = LensStar(mass=0.3, distance=250, proper_motion=[-4.62, -5.60])
    
    # Bulge PM: https://openaccess.leidenuniv.nl/bitstream/handle/1887/15120/05.pdf;jsessionid=BE631BF1B0522730D3C617C7F2E85230?sequence=6
    source = SourceStar(mass=1.0, distance=8000., proper_motion=[0., 0.])
    
    ml = MicrolensingEvent(lens, source)
    print ml.crossing_time / 60. / 60. / 24.
    
    source = SourceStar(mass=1.0, distance=500., proper_motion=[0, 0])
    
    ml = MicrolensingEvent(lens, source)
    print ml.crossing_time / 60. / 60. / 24.




""" API wants:
    
    print(G * M / c**2)
    
    G.as_units(M.units) ???

"""