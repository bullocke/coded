#!/usr/bin/env python
# -*- coding: UTF-8 -*-
""" Postprocess Continuous Degradation Detection (CDD) results.

Usage: postprocess.py [options] <input> <output> <param>

"""


import cv2, sys
import pymeanshift as pms
import numpy as np
import gdal
import scipy.stats
from docopt import docopt
from skimage.color import rgb2gray
#from skimage.filters import sobel
from scipy import ndimage
from skimage.segmentation import felzenszwalb, slic, quickshift#, watershed
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
import time, sys
import yaml

# CDD utilities
from postprocess_utils import *
from classify import *

def main(config, input, output):
    """ Main function for handling postprocessing procedure """

    ## GATHER INPUTS ##
 
    # Classification
    print "doing classification..."
    before_array, after_array, ftf_array = do_classify(config, input)    

    # Sieve
    print "doing sieve..."
    sieved_array = sieve(config, input)


    print "cleaning it up a bit..."
    #Convert date from years since 1970 to year
    sieved_array = convert_date(config, sieved_array)

    full_array = np.copy(sieved_array)


    #Geometry features
    if config['postprocessing']['window']['do_window']:
        window_array = get_geom_feats(config, array)

    # Get change difference
    change_dif_array = convert_change_dif(config, sieved_array)

    # Buffer non-forest
    if config['postprocessing']['buffer']['do_buffer']:
	print "buffering non-forest pixels..."
        sieved_array = buffer_nonforest(config, input, sieved_array, before_array, change_dif_array)

        if config['postprocessing']['do_connect']:
    	    print "returning areas that connect to previous change or non forest..."
            sieved_array = extend_nonforest(config, sieved_array, full_array)

    # Use forest to forest classification to get sieved deg date, magnitude, and change in nfdi
    print "almost there..."
    deg_mag = get_deg_magnitude(config, ftf_array, sieved_array, change_dif_array)

    deg_mag = min_max_years(config, deg_mag)
    # Segmentation
    # TODO 

    # save
    save_raster(deg_mag, input, output) 
    print "done!"


if __name__ == '__main__':
    args = docopt(__doc__, version='0.6.2')

image = args['<input>']
output = args['<output>']
params = args['<param>']
try:
    with open(params, 'r') as f:
	config = yaml.safe_load(f)
except:
    print "Invalid config file"
    sys.exit()
main(config, image, output)
