"""
    Various classes and functions for interfacing with the PTF photometric
    database (David Levitan's db), as described here:
        http://www.astro.caltech.edu/~dlevitan/ptf/photomdb.html
"""

# Standard library
import sys, os

# Third-party
import numpy as np
import apwlib.geometry as g

# PTF
from ..globals import ccd_size, all_fields, ccd_edge_cutoff
from ..lightcurve import PTFLightCurve, PDBLightCurve
from ..analyze import compute_variability_indices
from ..util import get_logger
logger = get_logger(__name__)

try:
    import tables
except ImportError:
    logger.warning("PyTables not found! Some functionality won't work."
                   "\n (on navtara, try running with: /scr4/dlevitan/sw/"
                    "epd-7.1-1-rh5-x86_64/bin/python instead.)")

try:
    import galacticutils
except ImportError:
    logger.warning("galacticutils not found! SDSS search functionality "
                   "won't work.")

# Path to npy file containing all field IDs
#_all_fields = os.path.join(ptf_params.config["PROJECTPATH"], 
#                           "data", "all_fields.npy")
_all_fields = os.path.join(os.path.split(globals._base_path)[0], 
                           "data", "all_fields.npy")
all_fields = np.load(_all_fields)

match_path = "/scr4/dlevitan/matches"
pytable_base_string = os.path.join(match_path, "match_{filter.id:02d}"
                                   "_{field.id:06d}_{ccd.id:02d}.pytable")
filter_map = {"R" : 2, "g" : 1}
inv_filter_map = {2 : "R", 1 : "g"}

def quality_cut(sourcedata, source_id=None):
    """ This function accepts a Pytables table object (from the 'sourcedata' table)
        and returns only the rows that pass the given quality cuts.

        Updated 2012-10-16:
            David Levitan suggested not using sextractor bit 2, or IPAC bits 6 / 11.

        Parameters
        ----------
        sourcedata : table
            A pytables Table object -> 'chip.sourcedata'
        source_id : int
            A matchedSourceID

        IPAC Flags:
            # 0 	aircraft/satellite track
            # 1 	detection by SExtractor (before flattening)
            # 2 	high dark current
            # 3 	reserved
            # 4 	noisy/hot pixel
            # 5 	ghost
            # 6 	CCD-bleed
            # 7 	rad hit
            # 8 	saturated
            # 9 	dead/bad pixel
            # 10 	not-a-number pixel
            # 11 	dirt on optics (pixel Nsigma below coarse local median)
            # 12 	halo

        relPhotFlags:
            # bit 0 = calibration detection
            # bit 1 = no relative photometry generated
            # bit 2 = exposure with high systematic error
            # bit 3 = saturated

        SExtractor flags:
            0     The object has neighbors, bright and close enough to
                  significantly bias the photometry, or bad pixels
                  (more than 10% of the integrated area affected).
            1     The object was originally blended with another one.
            2     At least one pixel of the object is saturated
                  (or very close to).
            3     The object is truncates (to close to an image boundary).
            4     Object's aperture data are incomplete or corrupted.
            5     Object's isophotal data are incomplete or corrupted.
            6     A memory overflow occurred during deblending.
            7     A memory overflow occurred during extraction.
    """
    x_cut1, x_cut2 = ccd_edge_cutoff, ccd_size[0] - ccd_edge_cutoff
    y_cut1, y_cut2 = ccd_edge_cutoff, ccd_size[1] - ccd_edge_cutoff

    # TODO: Do a robust test to figure out what is faster: many constraints in 'where', or doing it with numpy

    if source_id == None:
        where_string = ""
    else:
        where_string = "(matchedSourceID == {})".format(source_id)

    if where_string.strip() == "":
        data = sourcedata.read()
        data = np.array(data, dtype=sourcedata.dtype)
    else:
        #data = sourcedata.readWhere(where_string)
        data = [x.fetch_all_fields() for x in sourcedata.where(where_string)]
        data = np.array(data, dtype=sourcedata.dtype)

    # Saturation limit, 14.3, based on email conversation with David Levitan
    # UPDATE: I NO LONGER CUT ON SATURATION LIMIT (data["mag"] > 14.3) &
    cut_data = data[(data["x_image"] > 15) & (data["x_image"] < 2033) & \
                    (data["y_image"] > 15) & (data["y_image"] < 4081) & \
                    (data["relPhotFlags"] < 4) & \
                    (data["mag"] < 21) & \
                    ((data["sextractorFlags"] & 251) == 0) & \
                    ((data["ipacFlags"] & 6077) == 0) & \
                    np.isfinite(data["mag"]) & np.isfinite(data["mjd"]) & np.isfinite(data["magErr"])]

    if len(cut_data) == 0:
        return np.array([])

    return cut_data

# ==================================================================================================
#   Classes
#

class Filter(object):

    def __init__(self, filter_id):
        """ Create a new filter object given a string or number id """

        if isinstance(filter_id, str):
            assert filter_id in filter_map.keys(), "Filter ID not valid: {}".format(filter_id)
            self.id = filter_map[filter_id]
            self.name = filter_id
        elif isinstance(filter_id, int):
            assert filter_id in inv_filter_map.keys(), "Filter ID not valid: {}".format(filter_id)
            self.id = filter_id
            self.name = inv_filter_map[filter_id]
        else:
            raise ValueError("filter_id must be a number (e.g. 1, 2) or a string (e.g. 'g', 'R')")

    def __repr__(self):
        return "<Filter: id={}, name={}>".format(self.id, self.name)

    def __str__(self):
        return self.name

class Field(object):
    """ Represents PTF Field """

    def __repr__(self):
        return "<Field: id={}, filter={}>".format(self.id, self.filter.id)

    def __init__(self, field_id, filter=None, number_of_exposures=None):
        """ Create a field object given a PTF Field ID

            Parameters
            ----------
            field_id : int, Field
                The PTF Field ID for a field.
            filter : Filter
                A PTF Filter object (e.g. R = 2, g = 1)
            number_of_exposures : int (optional)
                The number of exposures this field has in the specified filter.
        """

        if isinstance(field_id, Field):
            filter = field_id.filter
            field_id = field_id.id

        # Validate Field ID
        try:
            self.id = int(field_id)
        except ValueError:
            print "field_id must be an integer or parseable string, e.g. 110002 or '110002'"
            raise

        # Validate filter_id
        if isinstance(filter, Filter):
            self.filter = filter
        elif isinstance(filter, str):
            self.filter = Filter(filter)
        elif filter == None:
            raise ValueError("You must specify the filter parameter!")
        else:
            raise ValueError("filter parameter must be Filter object or a string (e.g. R, g)")

        # Validate number_of_exposures
        if number_of_exposures != None:
            self._num_exp = int(number_of_exposures)
        else:
            self._num_exp = None

        self.ccds = dict()

        # Create new CCD objects for this field
        for ccd_id in range(12):
            try:
                self.ccds[ccd_id] = CCD(ccd_id, field=self, filter=self.filter)
            except ValueError:
                #logger.debug("CCD {} not found for Field {}.".format(ccd_id, self))
                pass

        if len(self.ccds) == 0:
            logger.debug("No CCD data found for: {}".format(self))

        this_field = all_fields[all_fields["id"] == self.id]
        self.ra = self.dec = None
        try:
            self.ra = g.RA.fromDegrees(this_field["ra"][0])
            self.dec = g.Dec.fromDegrees(this_field["dec"][0])

            if self.dec.degrees < -40: raise ValueError()
        except IndexError:
            logger.warning("Field {} not found in field list, with {} observations!".format(self, self._num_exp))
            self.ra = self.dec = None
        except ValueError:
            logger.warning("Field {} has wonky coordinates: {}, {}".format(self, self.ra, self.dec))
            self.ra = self.dec = None

    @property
    def number_of_exposures(self):
        """ If the number is specified when the field is instantiated, it simply returns that
            number. Otherwise, it returns the median number of exposures for all CCDs in this field.
        """
        if self._num_exp == None:
            return int(np.median([len(exp) for exp in self.exposures]))
        else:
            return self._num_exp

    @property
    def exposures(self):
        """ To get the number of observations, this must be run on navtara so the
            script has access to the PTF Photometric Database.
        """
        exposures_per_ccd = dict()
        for ccdid,ccd in self.ccds.items():
            chip = ccd.read()
            exposures_per_ccd[ccdid] = chip.exposures[:]
            #ccd.close()

        return exposures_per_ccd

    @property
    def baseline(self):
        """ To get the baseline, this must be run on navtara so the
            script has access to the PTF Photometric Database.
        """
        baseline_per_ccd = dict()
        for ccdid,ccd in self.ccds.items():
            chip = ccd.read()
            mjds = np.sort(chip.exposures.col("obsMJD"))
            baseline_per_ccd[ccdid] = mjds[-1]-mjds[0]
            #ccd.close()

        return baseline_per_ccd

    def close(self):
        for ccd in self.ccds.values():
            ccd.close()

class CCD(object):
    _chip = None
    _file = None

    def __repr__(self):
        return "<CCD: id={}, field={}>".format(self.id, self.field)

    def __init__(self, ccd_id, field, filter):
        # Validate CCD ID

        if isinstance(ccd_id, CCD):
            ccd_id = ccd_id.id

        try:
            self.id = int(ccd_id)
        except ValueError:
            print "ccd_id must be an integer or parseable string, e.g. 1 or '03'"
            raise

        if isinstance(field, Field):
            self.field = field
        else:
            raise ValueError("field parameter must be a Field object.")

        if isinstance(filter, Filter):
            self.filter = filter
        else:
            raise ValueError("filter parameter must be a Filter object.")

        self.filename = pytable_base_string.format(filter=self.filter, field=self.field, ccd=self)

        if not os.path.exists(self.filename):
            raise ValueError("CCD data file does not exist!")

    def read(self):
        if self._file == None:
            self._file = tables.openFile(self.filename)

        if self._chip == None:
            self._chip = getattr(getattr(getattr(self._file.root, "filter{:02d}".format(self.filter.id)), "field{:06d}".format(self.field.id)), "chip{:02d}".format(self.id))

        return self._chip

    def close(self):
        self._file.close()
        self._file = None

    def maximum_outlier_eta(self):
        chip = self.read()

        sources = chip.sources.readWhere("(vonNeumannRatio > 0.)")
        while True:
            outlier = sources[np.array(sources["vonNeumannRatio"]).argmin()]
            lc = self.light_curve(outlier["matchedSourceID"], clean=True)
            if len(lc) < 10:
                sources = np.delete(sources, np.array(sources["vonNeumannRatio"]).argmin())
            else:
                break

        return lc


    def light_curve(self, source_id, mag_type="relative", clean=False, barebones=True):
        """ Get a light curve for a given source ID from this chip """

        chip = self.read()

        if clean:
            data = quality_cut(chip.sourcedata, source_id=source_id)
        else:
            data = chip.sourcedata.readWhere('(matchedSourceID == {})'.format(source_id))

        if len(data) == 0:
            return PTFLightCurve(mjd=[], mag=[], error=[])

        mjd = data["mjd"]

        if mag_type == 'relative':
            mag = data["mag"]
            mag_err = data["magErr"]
        elif mag_type == 'absolute':
            mag = data["mag_auto"]/1000.0 + data["absphotzp"]
            mag_err = data["magerr_auto"]/10000.0

        try:
            ra, dec = data[0]["alpha_j2000"], data[0]["delta_j2000"]
        except IndexError: # no sourcedata for this source
            try:
                ra, dec = data[0]["ra"], data[0]["dec"]
            except IndexError:
                ra, dec = None, None

        if barebones:
            return PDBLightCurve(mjd=mjd, mag=mag, error=mag_err, field_id=self.field.id, ccd_id=self.id, source_id=source_id, ra=ra, dec=dec)
        else:
            lc = PDBLightCurve(mjd=mjd, mag=mag, error=mag_err, field_id=self.field.id, ccd_id=self.id, source_id=source_id, metadata=data, ra=ra, dec=dec)
            return lc

    def light_curves(self, source_ids, clean=True):
        """ """

        chip = self.read()

        if clean:
            sourcedata = quality_cut(chip.sourcedata)

            for source_id in source_ids:
                this_sourcedata = sourcedata[sourcedata["matchedSourceID"] == source_id]
                if len(this_sourcedata) == 0: continue

                ra, dec = this_sourcedata[0]["ra"], this_sourcedata[0]["dec"]
                yield PDBLightCurve(mjd=this_sourcedata["mjd"], mag=this_sourcedata["mag"], error=this_sourcedata["magErr"], field_id=self.field.id, ccd_id=self.id, source_id=source_id, ra=ra, dec=dec)

        else:
            raise NotImplementedError()

def random_light_curve(field_id=100101, *args, **kwargs):
    field = Field(field_id, "R")
    ccd = field.ccds.values()[np.random.randint(len(field.ccds.values()))]
    chip = ccd.read()
    sources = chip.sources.readWhere("ngoodobs > 10")
    np.random.shuffle(sources)

    for source in sources:
        lc = ccd.light_curve(source["matchedSourceID"], *args, **kwargs)
        if len(lc) > 10:
            return lc

def get_light_curve(field_id, ccd_id, source_id, **kwargs):
    field = Field(field_id, "R")
    ccd = field.ccds[ccd_id]
    lc = ccd.light_curve(source_id, **kwargs)
    ccd.close()
    return lc

"""
sources:
  "astrometricRMS": Float64Col(shape=(), dflt=0.0, pos=0),
  "bestAstrometricRMS": Float64Col(shape=(), dflt=0.0, pos=1),
  "bestChiSQ": Float32Col(shape=(), dflt=0.0, pos=2),
  "bestCon": Float32Col(shape=(), dflt=0.0, pos=3),
  "bestLinearTrend": Float32Col(shape=(), dflt=0.0, pos=4),
  "bestMagRMS": Float32Col(shape=(), dflt=0.0, pos=5),
  "bestMaxMag": Float32Col(shape=(), dflt=0.0, pos=6),
  "bestMaxSlope": Float32Col(shape=(), dflt=0.0, pos=7),
  "bestMeanMag": Float32Col(shape=(), dflt=0.0, pos=8),
  "bestMedianAbsDev": Float32Col(shape=(), dflt=0.0, pos=9),
  "bestMedianMag": Float32Col(shape=(), dflt=0.0, pos=10),
  "bestMinMag": Float32Col(shape=(), dflt=0.0, pos=11),
  "bestNAboveMeanByStd": UInt16Col(shape=(3,), dflt=0, pos=12),
  "bestNBelowMeanByStd": UInt16Col(shape=(3,), dflt=0, pos=13),
  "bestNMedianBufferRange": UInt16Col(shape=(), dflt=0, pos=14),
  "bestNPairPosSlope": UInt16Col(shape=(), dflt=0, pos=15),
  "bestPercentiles": Float32Col(shape=(12,), dflt=0.0, pos=16),
  "bestPeriodSearch": Float32Col(shape=(2,), dflt=0.0, pos=17),
  "bestSkewness": Float32Col(shape=(), dflt=0.0, pos=18),
  "bestSmallKurtosis": Float32Col(shape=(), dflt=0.0, pos=19),
  "bestStetsonJ": Float32Col(shape=(), dflt=0.0, pos=20),
  "bestStetsonK": Float32Col(shape=(), dflt=0.0, pos=21),
  "bestVonNeumannRatio": Float32Col(shape=(), dflt=0.0, pos=22),
  "bestWeightedMagRMS": Float32Col(shape=(), dflt=0.0, pos=23),
  "bestWeightedMeanMag": Float32Col(shape=(), dflt=0.0, pos=24),
  "chiSQ": Float32Col(shape=(), dflt=0.0, pos=25),
  "con": Float32Col(shape=(), dflt=0.0, pos=26),
  "dec": Float64Col(shape=(), dflt=0.0, pos=27),
  "linearTrend": Float32Col(shape=(), dflt=0.0, pos=28),
  "magRMS": Float32Col(shape=(), dflt=0.0, pos=29),
  "matchedSourceID": Int32Col(shape=(), dflt=0, pos=30),
  "maxMag": Float32Col(shape=(), dflt=0.0, pos=31),
  "maxSlope": Float32Col(shape=(), dflt=0.0, pos=32),
  "meanMag": Float32Col(shape=(), dflt=0.0, pos=33),
  "medianAbsDev": Float32Col(shape=(), dflt=0.0, pos=34),
  "medianMag": Float32Col(shape=(), dflt=0.0, pos=35),
  "minMag": Float32Col(shape=(), dflt=0.0, pos=36),
  "nAboveMeanByStd": UInt16Col(shape=(3,), dflt=0, pos=37),
  "nBelowMeanByStd": UInt16Col(shape=(3,), dflt=0, pos=38),
  "nMedianBufferRange": UInt16Col(shape=(), dflt=0, pos=39),
  "nPairPosSlope": UInt16Col(shape=(), dflt=0, pos=40),
  "nbestobs": UInt16Col(shape=(), dflt=0, pos=41),
  "ngoodobs": UInt16Col(shape=(), dflt=0, pos=42),
  "nobs": UInt16Col(shape=(), dflt=0, pos=43),
  "percentiles": Float32Col(shape=(12,), dflt=0.0, pos=44),
  "periodSearch": Float32Col(shape=(2,), dflt=0.0, pos=45),
  "ra": Float64Col(shape=(), dflt=0.0, pos=46),
  "referenceMag": Float32Col(shape=(), dflt=0.0, pos=47),
  "referenceMagErr": Float32Col(shape=(), dflt=0.0, pos=48),
  "skewness": Float32Col(shape=(), dflt=0.0, pos=49),
  "smallKurtosis": Float32Col(shape=(), dflt=0.0, pos=50),
  "stetsonJ": Float32Col(shape=(), dflt=0.0, pos=51),
  "stetsonK": Float32Col(shape=(), dflt=0.0, pos=52),
  "uncalibMeanMag": Float32Col(shape=(), dflt=0.0, pos=53),
  "vonNeumannRatio": Float32Col(shape=(), dflt=0.0, pos=54),
  "weightedMagRMS": Float32Col(shape=(), dflt=0.0, pos=55),
  "weightedMeanMag": Float32Col(shape=(), dflt=0.0, pos=56),
  "x": Float64Col(shape=(), dflt=0.0, pos=57),
  "y": Float64Col(shape=(), dflt=0.0, pos=58),
  "z": Float64Col(shape=(), dflt=0.0, pos=59)}

sourcedata:
  "a_world": Float32Col(shape=(), dflt=0.0, pos=0),
  "absphotzp": Float32Col(shape=(), dflt=0.0, pos=1),
  "alpha_j2000": Float64Col(shape=(), dflt=0.0, pos=2),
  "b_world": Float32Col(shape=(), dflt=0.0, pos=3),
  "background": Float32Col(shape=(), dflt=0.0, pos=4),
  "class_star": Float32Col(shape=(), dflt=0.0, pos=5),
  "delta_j2000": Float64Col(shape=(), dflt=0.0, pos=6),
  "errx2_image": Float32Col(shape=(), dflt=0.0, pos=7),
  "errxy_image": Float32Col(shape=(), dflt=0.0, pos=8),
  "erry2_image": Float32Col(shape=(), dflt=0.0, pos=9),
  "flux_auto": Float32Col(shape=(), dflt=0.0, pos=10),
  "flux_radius_1": Float32Col(shape=(), dflt=0.0, pos=11),
  "flux_radius_2": Float32Col(shape=(), dflt=0.0, pos=12),
  "flux_radius_3": Float32Col(shape=(), dflt=0.0, pos=13),
  "flux_radius_4": Float32Col(shape=(), dflt=0.0, pos=14),
  "flux_radius_5": Float32Col(shape=(), dflt=0.0, pos=15),
  "fluxerr_auto": Float32Col(shape=(), dflt=0.0, pos=16),
  "fwhm_image": Float32Col(shape=(), dflt=0.0, pos=17),
  "ipacFlags": Int16Col(shape=(), dflt=0, pos=18),
  "isoarea_world": Float32Col(shape=(), dflt=0.0, pos=19),
  "kron_radius": Float32Col(shape=(), dflt=0.0, pos=20),
  "mag": Float32Col(shape=(), dflt=0.0, pos=21),
  "magErr": Float32Col(shape=(), dflt=0.0, pos=22),
  "mag_aper_1": Int16Col(shape=(), dflt=0, pos=23),
  "mag_aper_2": Int16Col(shape=(), dflt=0, pos=24),
  "mag_aper_3": Int16Col(shape=(), dflt=0, pos=25),
  "mag_aper_4": Int16Col(shape=(), dflt=0, pos=26),
  "mag_aper_5": Int16Col(shape=(), dflt=0, pos=27),
  "mag_auto": Int16Col(shape=(), dflt=0, pos=28),
  "mag_iso": Int16Col(shape=(), dflt=0, pos=29),
  "mag_petro": Int16Col(shape=(), dflt=0, pos=30),
  "magerr_aper_1": Int16Col(shape=(), dflt=0, pos=31),
  "magerr_aper_2": Int16Col(shape=(), dflt=0, pos=32),
  "magerr_aper_3": Int16Col(shape=(), dflt=0, pos=33),
  "magerr_aper_4": Int16Col(shape=(), dflt=0, pos=34),
  "magerr_aper_5": Int16Col(shape=(), dflt=0, pos=35),
  "magerr_auto": Int16Col(shape=(), dflt=0, pos=36),
  "magerr_iso": Int16Col(shape=(), dflt=0, pos=37),
  "magerr_petro": Int16Col(shape=(), dflt=0, pos=38),
  "matchedSourceID": Int32Col(shape=(), dflt=0, pos=39),
  "mjd": Float64Col(shape=(), dflt=0.0, pos=40),
  "mu_max": Float32Col(shape=(), dflt=0.0, pos=41),
  "mu_threshold": Float32Col(shape=(), dflt=0.0, pos=42),
  "petro_radius": Float32Col(shape=(), dflt=0.0, pos=43),
  "pid": Int32Col(shape=(), dflt=0, pos=44),
  "relPhotFlags": Int16Col(shape=(), dflt=0, pos=45),
  "sextractorFlags": Int16Col(shape=(), dflt=0, pos=46),
  "sid": Int64Col(shape=(), dflt=0, pos=47),
  "theta_j2000": Float32Col(shape=(), dflt=0.0, pos=48),
  "threshold": Float32Col(shape=(), dflt=0.0, pos=49),
  "x": Float64Col(shape=(), dflt=0.0, pos=50),
  "x2_image": Float32Col(shape=(), dflt=0.0, pos=51),
  "x_image": Float32Col(shape=(), dflt=0.0, pos=52),
  "xpeak_image": Float32Col(shape=(), dflt=0.0, pos=53),
  "y": Float64Col(shape=(), dflt=0.0, pos=54),
  "y2_image": Float32Col(shape=(), dflt=0.0, pos=55),
  "y_image": Float32Col(shape=(), dflt=0.0, pos=56),
  "ypeak_image": Float32Col(shape=(), dflt=0.0, pos=57),
  "z": Float64Col(shape=(), dflt=0.0, pos=58)}

exposures:
  "absPhotZP": Float32Col(shape=(), dflt=0.0, pos=0),
  "absPhotZPRMS": Float32Col(shape=(), dflt=0.0, pos=1),
  "airmass": Float32Col(shape=(), dflt=0.0, pos=2),
  "background": Float32Col(shape=(), dflt=0.0, pos=3),
  "ccdID": UInt8Col(shape=(), dflt=0, pos=4),
  "decRMS": Float32Col(shape=(), dflt=0.0, pos=5),
  "expTime": Float32Col(shape=(), dflt=0.0, pos=6),
  "expid": Int32Col(shape=(), dflt=0, pos=7),
  "fieldID": UInt32Col(shape=(), dflt=0, pos=8),
  "filename": StringCol(itemsize=1024, shape=(), dflt='', pos=9),
  "filterID": UInt16Col(shape=(), dflt=0, pos=10),
  "infobits": Int32Col(shape=(), dflt=0, pos=11),
  "limitMag": Float32Col(shape=(), dflt=0.0, pos=12),
  "moonIllumFrac": Float32Col(shape=(), dflt=0.0, pos=13),
  "nSources": Int32Col(shape=(), dflt=0, pos=14),
  "nStars": Int32Col(shape=(), dflt=0, pos=15),
  "nTransients": Int32Col(shape=(), dflt=0, pos=16),
  "nightID": Int32Col(shape=(), dflt=0, pos=17),
  "obsDate": StringCol(itemsize=255, shape=(), dflt='', pos=18),
  "obsHJD": Float64Col(shape=(), dflt=0.0, pos=19),
  "obsMJD": Float64Col(shape=(), dflt=0.0, pos=20),
  "pid": Int32Col(shape=(), dflt=0, pos=21),
  "raRMS": Float32Col(shape=(), dflt=0.0, pos=22),
  "relPhotSatMag": Float32Col(shape=(), dflt=0.0, pos=23),
  "relPhotSysErr": Float32Col(shape=(), dflt=0.0, pos=24),
  "relPhotZP": Float32Col(shape=(), dflt=0.0, pos=25),
  "relPhotZPErr": Float32Col(shape=(), dflt=0.0, pos=26),
  "seeing": Float32Col(shape=(), dflt=0.0, pos=27)
"""
