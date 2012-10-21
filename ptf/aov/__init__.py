import logging
try:
    from aov import aov_periodogram_asczerny, testperiod_asczerny, findPeaks_aov
except ImportError:
    logging.warn("AOV failed to import.")
