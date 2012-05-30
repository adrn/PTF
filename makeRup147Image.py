import os,glob
import Image
import numpy as np
import pyfits as pf
import ptf.util as pu

def get_rup147_images():
    iq = pu.PTFImageQuery()
    iq.at_position("19 16 40", "-16 17 59")
    iq.field([1947,1948])
    iq.on_date((2012,4,30))
    iq.size(8.0)
    
    il = pu.PTFImageList(iq)
    
    to_download = {}
    for fieldid in [1947, 1948]:
        to_download[fieldid] = []
        for ccdid in range(12):
            sub_table = il.table[(il.table.fieldid == fieldid) & (il.table.ccdid == ccdid)]
            if len(sub_table) == 0: continue
            
            #to_download[fieldid].append(sub_table.data_filename[0])
            to_download[fieldid].append(sub_table.mask_filename[0])
    
    for fieldid in [1947, 1948]:
        pu.getAllImages(to_download[fieldid], prefix="/home/adrian/tmp/rup147")

def write_masked_images():
    scie = sorted(glob.glob("/home/adrian/tmp/rup147/*_scie*.fits"))
    mask = sorted(glob.glob("/home/adrian/tmp/rup147/*_mask*.fits"))
    
    for ii in range(len(scie)):
        fn_split = scie[ii].split("_")
        field, ccd = fn_split[8], fn_split[9].split(".")[0]
        
        im_file = scie[ii]
        mask_file = mask[ii]
    
        mask_hdulist = pf.open(mask_file)
        im_hdulist = pf.open(im_file)
    
        bitmask = mask_hdulist[0].data
        im_data = im_hdulist[0].data
    
        bm = int(0b111101110101)
        im_data[np.bitwise_and(bitmask, bm) > 0] = 0
        
        hdu = pf.PrimaryHDU(im_data)
        
        for key,val in im_hdulist[0].header.items():
            hdu.header.update(key,val)
            
        hdu.writeto("/home/adrian/tmp/rup147/masked/{}_{}.fits".format(field, ccd))
        
    return
    
def write_coadd_image():
    im_hdulist = pf.open("/home/adrian/tmp/rup147/masked/coadd.fits")
    im_data = im_hdulist[0].data
    
    scaled_im_data = np.arcsinh(5.*(im_data - im_data.min()) / (im_data.max() - im_data.min()))
    scaled_im_data = (scaled_im_data - scaled_im_data.min()) / (scaled_im_data.max() - scaled_im_data.min())
    scaled_im_data *= 255.
    
    im = Image.fromarray(scaled_im_data.astype(np.uint8))
    im.save("test.png")

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.DEBUG)
    
    #get_rup147_images()
    write_masked_images()
    #write_coadd_image()