""" Get map strata for entire study period. 

Parameters:
  param: postprocessing yaml parameter
  input: list of time-sequential cdd output rasters
  output: location to save output raster

Usage: make_strata_map.py [options] <param> <input> <output>

  --allclasses=<AC>      output all class strata instead of f/nf

"""

from docopt import docopt
import gdal
from osgeo import ogr, osr
import sys
import os
import numpy as np
import yaml
import pandas as pd
from postprocess_utils import save_raster_simple

def main(inputs, opts, output):

    """ 
    Main function for getting map strata
    
    Output is map strata for study time period:
	1: stable forest
	2: stable non-forest
	3: deforestation
	4: degradation
	5: degradation -> deforestation
	6: deforestation -> other -> regrowth

    Input stratas represent:
	1-5: stable class during that time period
	7: degradation
	8: deforestation
    """ 

    # Parse param file
    strata_band = opts['general']['strata_band']
    forestlabel = opts['classification']['forestlabel']

    # Make strata input raster
    im_op = gdal.Open(inputs[0])
    im_ar = im_op.GetRasterBand(1).ReadAsArray()
    dim1, dim2 = np.shape(im_ar) 
    im_ar = None

    stratas = np.zeros((dim1, dim2, len(inputs)))
    outstrata = np.zeros((dim1, dim2))

    for i in range(len(inputs)):
	im_op = gdal.Open(inputs[i])
	stratas[:,:,i] = im_op.GetRasterBand(strata_band).ReadAsArray()

    for _y in range(dim1):
        print _y
	for _x in range(dim2):
	    # Get all years data
	    data = stratas[_y,_x,:]

	    # LABEL: 1 -- stable forest
	    if np.all(data == forestlabel):
		outstrata[_y, _x] = 1
	    # LABEL: 2 -- stable non-forest
	    elif np.all(data <= forestlabel) and np.any(data != 0):
		outstrata[_y, _x] = 2

            # Test for degradation
	    degradation = np.where(data == 7)[0]
	    if degradation.shape[0] > 0:

		# LABEL: 4 -- degradation (may be overwritten by deg->def)
		outstrata[_y, _x] = 4
		after_deg = data[degradation[0]:]
		deg_to_def = np.any(np.in1d(after_deg, [1,2,3,4,8]))
		if deg_to_def:
		    outstrata[_y, _x] = 6

	    # Test for deforestation
	    deforestation = np.where(data == 8)[0]
	    if deforestation.shape[0] > 0:

		after_def = data[deforestation[0]:]
		bef_def = data[:deforestation[0]]

		regrowth = np.any(np.in1d(after_def, [forestlabel]))
		deg_def = np.any(np.in1d(bef_def, [7]))

		# LABEL: 5 -- deforestation -> regrowth?
		if regrowth:
		    outstrata[_y, _x] = 7
		# LABEL: 5 -- deg before def? 
		elif deg_def:
		    outstrata[_y,_x] = 6
		# LABEL: 3 -- deforestation alone
		else:
		    outstrata[_y,_x] = 3

    save_raster_simple(outstrata, inputs[0], output)

if __name__ == '__main__':
    args = docopt(__doc__, version='0.6.2')

param=args['<param>']
output=args['<output>']
inputs=args['<input>'].split(' ')

try:
    with open(param, 'r') as f:
	opts = yaml.safe_load(f)
except:
    print "Invalid config file"
    sys.exit()

main(inputs, opts, output)


