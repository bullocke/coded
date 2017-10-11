#!/usr/bin/env python
# -*- coding: UTF-8 -*-
""" Turn post-processed CDD result into disturbance classes

Usage: disurbance_classes.py [options] <class_csv> <input> <output>

  --mag=MAG     Change magnitude band for input file (default 2)
  --bef=BEG     Band containing NFDI magnitude before disturbance (default 5)
  --aft=AFT     Band containing NFDI magnitude after disturbance (default 4)

The disturbance classes: 
  1. Low mag, temporary
  2. Low mag, permanent or increasing in mag
  3. Medium mag, temporary
  4. Medium mag, permament or increasing in mag
  5. High mag, temporary
  6. High mag, permament or increasing in mag

The <class_csv> file is a csv file containing ? columns: 
  1. class label (integer)
  2. magnitude range (beginning)
  3. magnitude range (end)
  4. delta mag (pre-disturbance - post-disturbance) range (beginning)
  5. delta mag range (end) 

"""

import sys
import numpy as np
import gdal
import pandas as pd
from docopt import docopt

def save_raster(array, path, dst_filename):

    example = gdal.Open(path)
    x_pixels = array.shape[1]  # number of pixels in x
    y_pixels = array.shape[0]  # number of pixels in y
    bands = 1
    driver = gdal.GetDriverByName('GTiff')
    dataset = driver.Create(dst_filename,x_pixels, y_pixels, bands ,gdal.GDT_Byte)

    geotrans=example.GetGeoTransform()  #get GeoTranform from existed 'data0'
    proj=example.GetProjection() #you can get from a exsited tif or import 
    dataset.SetGeoTransform(geotrans)
    dataset.SetProjection(proj)

    dataset.GetRasterBand(1).WriteArray(array)

    dataset.FlushCache()
    #dataset=None

def get_inputs(image, mag_band, bef_band, aft_band):
    im_op = gdal.Open(image)

    mag_array = im_op.GetRasterBand(mag_band).ReadAsArray()

    bef_array = im_op.GetRasterBand(bef_band).ReadAsArray()

    aft_array = im_op.GetRasterBand(aft_band).ReadAsArray()

    delta_array = bef_array - aft_array

    return mag_array, delta_array

def classify(mag_ar, delta_ar, class_data):
    class_out = np.zeros_like(mag_ar)

    num_classes = class_data.shape[0]

    for i in range(num_classes):
        class_id = class_data['class'][i]

        class_indices = np.logical_and(
			   np.logical_and
				((mag_ar > class_data['mag0'][i])
				,(mag_ar < class_data['mag1'][i]))
			   ,
			   np.logical_and
				((delta_ar > class_data['delta0'][i])
				,(delta_ar < class_data['delta1'][i])))

        class_out[class_indices] = class_id

    return class_out

if __name__ == '__main__':
    args = docopt(__doc__, version='0.6.2')

image = args['<input>']
output = args['<output>']
class_csv = args['<class_csv>']

if args['--mag']:
    mag_band = int(args['--mag'])
else:
    mag_band = 2 

if args['--bef']:
    bef_band = int(args['--bef'])
else:
    bef_band = 5

if args['--aft']:
    aft_band = int(args['--aft'])
else:
    aft_band = 4

class_data = pd.read_csv(class_csv)

mag_ar, delta_ar = get_inputs(image, mag_band, bef_band, aft_band)

class_out = classify(mag_ar, delta_ar, class_data)

save_raster(class_out, image, output)

