import sys
import numpy as np
from astropy.io import ascii

candidates = ascii.read("data/candidate_list_ipac.tbl.txt")

ptf_name = sys.argv[1]

row = np.array(candidates[candidates["object"] == ptf_name])

latex_row = ""

try:
    print "2MASS"
    print "-----"
    print "J = {0} +/- {1}, Quality: {2}".format(row["j_m_2mass"][0], row["j_msig_2mass"][0], row["ph_qual_2mass"][0][0])
    print "H = {0} +/- {1}, Quality: {2}".format(row["h_m_2mass"][0], row["h_msig_2mass"][0], row["ph_qual_2mass"][0][1])
    print "K = {0} +/- {1}, Quality: {2}".format(row["k_m_2mass"][0], row["k_msig_2mass"][0], row["ph_qual_2mass"][0][2])
    print "J-K = {0} +/- {1}".format(row["j_m_2mass"][0]-row["k_m_2mass"][0], np.sqrt(row["j_msig_2mass"][0]**2+row["k_msig_2mass"][0]**2))
    
    latex_row += "{0:.3f}$\pm${1:.3f} & {2:.3f}$\pm${3:.3f} & {4:.3f}$\pm${5:.3f}".format(row["j_m_2mass"][0], row["j_msig_2mass"][0],
                                                                                    row["h_m_2mass"][0], row["h_msig_2mass"][0],
                                                                                    row["k_m_2mass"][0], row["k_msig_2mass"][0])
except IndexError:
    pass

try:
    print "-----"
    print "WISE"
    print "-----"
    print "W1 = {0} +/- {1}, Quality: {2[0]}, Detection? {3[0]}".format(row["w1mpro"][0], row["w1sigmpro"][0], row["ph_qual"][0], "{0:0<4d}".format(int(str(bin(row["det_bit"][0]))[2:])))
    print "W2 = {0} +/- {1}, Quality: {2[1]}, Detection? {3[1]}".format(row["w2mpro"][0], row["w2sigmpro"][0], row["ph_qual"][0], "{0:0<4d}".format(int(str(bin(row["det_bit"][0]))[2:])))
    print "W3 = {0} +/- {1}, Quality: {2[2]}, Detection? {3[2]}".format(row["w3mpro"][0], row["w3sigmpro"][0], row["ph_qual"][0], "{0:0<4d}".format(int(str(bin(row["det_bit"][0]))[2:])))
    print "W4 = {0} +/- {1}, Quality: {2[3]}, Detection? {3[3]}".format(row["w4mpro"][0], row["w4sigmpro"][0], row["ph_qual"][0], "{0:0<4d}".format(int(str(bin(row["det_bit"][0]))[2:])))
    print "W1-W2 = {0} +/- {1}".format(row["w1mpro"][0]-row["w2mpro"][0], np.sqrt(row["w1sigmpro"][0]**2+row["w2sigmpro"][0]**2))
    print "W2-W3 = {0} +/- {1}".format(row["w2mpro"][0]-row["w3mpro"][0], np.sqrt(row["w2sigmpro"][0]**2+row["w3sigmpro"][0]**2))
    print "Number of blend fits to source: {0}".format(row["nb"][0])
    print "Probability that source morphology is not consistent with single PSF: {0}".format(row["ext_flg"][0])
    
    latex_row += " & {0:.3f}$\pm${1:.3f} & {2:.3f}$\pm${3:.3f} & {4:.3f}$\pm${5:.3f} & {6:.3f}$\pm${7:.3f}".format(row["w1mpro"][0], row["w1sigmpro"][0],
                                                                                                           row["w2mpro"][0], row["w2sigmpro"][0],
                                                                                                           row["w3mpro"][0], row["w3sigmpro"][0],
                                                                                                           row["w4mpro"][0], row["w4sigmpro"][0])
except IndexError:
    pass
    
print latex_row