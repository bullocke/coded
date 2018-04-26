""" Get map strata for entire study period 

Parameters:
  param: postprocessing yaml parameter
  image[1-3]: postprocessed images (1990-2000, 2000-2010, 2010-2013)
  raw_image[1-3]: raw algorithm outptus (1990-2000, 2000-2010, 2010-2013)
  output: location to save output raster

Usage: strata_map.py [options] <param> <image1> <image2> <image3> <output>

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
from postprocess_utils import save_raster, sieve, raster_buffer

# V10: Pre strata, post strata in both forest and nf
# V9: Pre strata, no post strata
# V8: Pre strata, post strata only in forest

def main(inputs, opts, output):

    """ 
    Main function for getting map strata
    
    Output is map strata for study time period: 
	1: stable forest
	2: stable non-forest
	3: deforestation
	4: degradation
	5: potential disturbance in non-forest
	6: potential degradation in deforestation

    Input stratas represent:
	1-5: stable class during that time period
	7: degradation
	8: deforestation
    """ 

    # Parse param file
    strata_band = opts['general']['strata_band']
    date_band = opts['general']['date_band']
    potential_band = opts['general']['potential_band']
    forestlabel = opts['classification']['forestlabel']
    buffer_distance = opts['postprocessing']['buffer_strata']['distance']

    # Make disturbance_dates[1, _y, _x] input raster
    im_op = gdal.Open(inputs[0])
    im_ar = im_op.GetRasterBand(1).ReadAsArray()
    dim1, dim2 = np.shape(im_ar) 
    im_ar = None

    stratas = np.zeros((dim1, dim2, len(inputs)))
    possible_deg = np.zeros((dim1, dim2, len(inputs)))
    years = np.zeros((dim1, dim2, len(inputs)))
    disturbance_dates = np.zeros((2, dim1, dim2))
    strata = np.zeros((dim1, dim2))
    defor_strata = np.zeros((dim1, dim2))

    for i in range(len(inputs)):
	im_op = gdal.Open(inputs[i])
	stratas[:,:,i] = im_op.GetRasterBand(strata_band).ReadAsArray()
	years[:,:,i] = im_op.GetRasterBand(date_band).ReadAsArray()
	possible_deg[:,:,i] = im_op.GetRasterBand(potential_band).ReadAsArray()

    for _y in range(dim1):
	for _x in range(dim2):
	    # Get all years data
	    data = stratas[_y,_x,:]

	    # LABEL: 1 -- stable forest
	    if np.all(data == forestlabel):
		strata[_y, _x] = 1
		disturbance_dates[0, _y, _x] = 0
		disturbance_dates[1, _y, _x] = 0

	    # LABEL: 2 -- stable non-forest
	    elif np.all(data <= forestlabel) and np.any(data != 0):
		strata[_y, _x] = 2
		disturbance_dates[0, _y, _x] = 0
		disturbance_dates[1, _y, _x] = 0

            # Degradation
	    degradation = np.where(data == 7)[0]
	    if degradation.shape[0] > 0:

		# LABEL: 4 -- degradation (may be overwritten by deg->def)
		strata[_y, _x] = 4
		defor_strata[_y, _x] = 1
		disturbance_dates[0, _y, _x] = years[_y,_x,:][degradation[0]]

		# Multiple degradation events
		if degradation.shape[0] > 1:
		    disturbance_dates[1, _y, _x] = years[_y,_x,:][degradation[1]]

		# Degradation before deforestation?
		after_deg = data[degradation[0]:]
		deg_to_def = np.in1d(after_deg, [1,2,3,4,8])
		if deg_to_def.sum() > 1:
		    strata[_y, _x] = 3
		    defor_strata[_y, _x] = 1
		    if np.all(years[_y,_x,:][deg_to_def] == 0):
  		        disturbance_dates[1, _y, _x] =  disturbance_dates[0, _y, _x]
		    else:
			disturbance_dates[1, _y, _x] = years[_y,_x,:][deg_to_def][0]

	    # Test for deforestation
	    deforestation = np.where(data == 8)[0]
	    if deforestation.shape[0] > 0:

		after_def = data[deforestation[0]:]
		bef_def = data[:deforestation[0]]

		deg_def = np.any(np.in1d(bef_def, [7]))

		# LABEL: 3 -- deg before def? 
		if deg_def:
		    strata[ _y,_x] = 3
		    defor_strata[_y, _x] = 1
		    possible_dates = np.where(years[_y, _x, :] > 0)[0]
		    disturbance_dates[1, _y,_x] = years[_y, _x, :][possible_dates[1]]
		# LABEL: 3 -- deforestation alone
		else:
		    strata[_y,_x] = 3
		    defor_strata[_y, _x] = 1
		    possible_dates = np.where(years[_y, _x, :] > 0)[0]
		    disturbance_dates[0, _y,_x] = years[_y, _x, :][possible_dates[0]]

    # Buffer deforestation
    def_buffer_strata = raster_buffer(inputs[0], defor_strata, dist=buffer_distance)
    def_buffer_indices = np.where((def_buffer_strata > 0) & (strata < 3) & (strata > 0))

    for i in range(len(inputs)):
        # Possible dist in stable
        #pos_deg = np.where((possible_deg[:,:,i] == 1) & (strata < 3) & (strata > 0))

	# Possible dist in forest
        pos_deg = np.where((possible_deg[:,:,i] == 1) & (strata == 1))
	strata[pos_deg] = 5

	# Possible deg in def
        pos_deg = np.where((possible_deg[:,:,i] == 2) & (strata == 3))
	strata[pos_deg] = 6

    strata[def_buffer_indices] = 5

    outstrata = np.stack((strata, disturbance_dates[0,:,:],disturbance_dates[1,:,:]))

    save_raster(outstrata, inputs[0], output)
    sieve(opts, output)

if __name__ == '__main__':
    args = docopt(__doc__, version='0.6.2')

param=args['<param>']
output=args['<output>']
inputs = []
image1=args['<image1>']
image2=args['<image2>']
image3=args['<image3>']
inputs.append(image1)
inputs.append(image2)
inputs.append(image3)

try:
    with open(param, 'r') as f:
	opts = yaml.safe_load(f)
except:
    print "Invalid config file"
    sys.exit()

main(inputs, opts, output)


