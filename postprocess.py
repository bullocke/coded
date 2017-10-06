#!/usr/bin/env python
# -*- coding: UTF-8 -*-
""" Postprocess Continuous Degradation Detection (CDD) results.

Usage: postprocess_cdd.py [options] (sieve | fz | kmeans) <input> <output>

  --seg=<SEG_SIZE>        Minimum segment size 
  --sigma=<SIGMA>         Sigma value for FZ test
  --scale=<SCALE>         Scale value for FZ test
  --convdate=<CONVDATE>   Convert date to year

"""

import cv2, sys
import pymeanshift as pms
import numpy as np
import gdal
import scipy.stats
from docopt import docopt
from skimage.color import rgb2gray
#from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, slic, quickshift#, watershed
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
import time

def convert_date(array):
    array[0,:,:][array[0,:,:] > 0] += 1970
    array[0,:,:] = array[0,:,:].astype(np.int)
    return array

def save_raster(array, path, dst_filename, convdate):

    #Convert date from years since 1970 to year
    if convdate:
        array = convert_date(array)

    example = gdal.Open(path)
    x_pixels = array.shape[2]  # number of pixels in x
    y_pixels = array.shape[1]  # number of pixels in y
    bands = array.shape[0]
    driver = gdal.GetDriverByName('GTiff')
    dataset = driver.Create(dst_filename,x_pixels, y_pixels, bands ,gdal.GDT_Float64)

    geotrans=example.GetGeoTransform()  #get GeoTranform from existed 'data0'
    proj=example.GetProjection() #you can get from a exsited tif or import 
    dataset.SetGeoTransform(geotrans)
    dataset.SetProjection(proj)

    for b in range(bands):
        dataset.GetRasterBand(b+1).WriteArray(array[b,:,:])

    dataset.FlushCache()
    #dataset=None

def save_raster_memory(array, path):
    example = gdal.Open(path)
    x_pixels = array.shape[1]  # number of pixels in x
    y_pixels = array.shape[0]  # number of pixels in y
    driver = gdal.GetDriverByName('MEM')
    dataset = driver.Create('',x_pixels, y_pixels, 1,gdal.GDT_Int32) #TODO: bands
    dataset.GetRasterBand(1).WriteArray(array[:,:])

    # follow code is adding GeoTranform and Projection
    geotrans=example.GetGeoTransform()  #get GeoTranform from existed 'data0'
    proj=example.GetProjection() #you can get from a exsited tif or import 
    dataset.SetGeoTransform(geotrans)
    dataset.SetProjection(proj)
    return dataset

def segment_fz(image, output, scale, sigma, minseg, convdate):
    original_im = gdal.Open(image)

    #Assign median values based on Felzenzwalb segmentation algorithm
    three_d_image = original_im.ReadAsArray().astype(np.uint8)
    three_d_image = three_d_image.swapaxes(0, 2)
    three_d_image = three_d_image.swapaxes(0, 1)
    img = three_d_image[:,:,(0,1,3)]

    full_image = original_im.ReadAsArray()
    full_image = full_image.swapaxes(0, 2)
    full_image = full_image.swapaxes(0, 1)

    segments_fz = felzenszwalb(img, scale=scale, sigma=sigma, min_size=minseg)
    median_image = np.zeros_like(full_image).astype(np.float32)
    for band in range(4):
        for seg in np.unique(segments_fz):
            values = full_image[:,:,band][segments_fz == seg]

            if band < 2:
	        if np.median(values) > 0:
                    med = np.median(values[values>0])
                else:
                    med = 0

            else:
		values[np.isnan(values)] = 0
#		values *= 1000
	        if np.median(values) > 0:
                    med = np.median(values[values>0])
                else:
                    med = 0
            median_image[:,:,band][segments_fz == seg] = med

    #Reshape
    s1, s2, s3 = median_image.shape

    median_image = median_image.swapaxes(1, 0)
    median_image = median_image.swapaxes(2, 0)
    save_raster(median_image, image, output, convdate)
    sys.exit()

def segment_km(image, output):
    original_im = gdal.Open(image)

    #Assign median values based on Felzenzwalb segmentation algorithm
    three_d_image = original_im.ReadAsArray().astype(np.uint8)
    three_d_image = three_d_image.swapaxes(0, 2)
    three_d_image = three_d_image.swapaxes(0, 1)
    img = three_d_image

    full_image = original_im.ReadAsArray()
    full_image = full_image.swapaxes(0, 2)
    full_image = full_image.swapaxes(0, 1)

    segments_slic = slic(img, n_segments=250, compactness=10, sigma=1)
    median_image = np.zeros_like(img).astype(np.float32)
    for band in range(3):
        for seg in np.unique(segments_slic):
            values = full_image[:,:,band][segments_slic == seg]
#            mode = scipy.stats.mode(values.astype(int)).mode[0]
#            if mode > 0:
            med = np.median(values[values>0])
#            else:
#                med = 0
            median_image[:,:,band][segments_slic == seg] = med

    save_raster(median_image, image, output)
    sys.exit()

def sieve(image, dst_filename, convdate):
    # 1. Remove all single pixels

    #First create a band in memory that's that's just 1s and 0s
    src_ds = gdal.Open( image, gdal.GA_ReadOnly )
    srcband = src_ds.GetRasterBand(1)
    srcarray = srcband.ReadAsArray()
    srcarray[srcarray > 0] = 1

    mem_rast = save_raster_memory(srcarray, image)
    mem_band = mem_rast.GetRasterBand(1)

    #Now the code behind gdal_sieve.py
    maskband = None
    drv = gdal.GetDriverByName('GTiff')
    dst_ds = drv.Create( dst_filename,src_ds.RasterXSize, src_ds.RasterYSize,1,
                         srcband.DataType )
    wkt = src_ds.GetProjection()
    if wkt != '':
        dst_ds.SetProjection( wkt )
    dst_ds.SetGeoTransform( src_ds.GetGeoTransform() )

    dstband = dst_ds.GetRasterBand(1)

    # Parameters
    prog_func = None
    threshold = 4
    connectedness = 8

    result = gdal.SieveFilter(mem_band, maskband, dstband,
                              threshold, connectedness,
                              callback = prog_func )

    sieved = dstband.ReadAsArray()
    sieved[sieved < 0] = 0

    src_new = gdal.Open(image)
    out_img = src_new.ReadAsArray().astype(np.float)

    out_img[np.isnan(out_img)] = 0 

    for b in range(out_img.shape[0]):
        out_img[b,:,:][sieved == 0] = 0




    dst_full = dst_filename.split('.')[0] + '_full.tif'

    save_raster(out_img, image, dst_full, convdate)
    sys.exit()


if __name__ == '__main__':
    args = docopt(__doc__, version='0.6.2')


image = args['<input>']
output = args['<output>']

if args['--sigma']:
    sigma = float(args['--sigma'])
else:
    sigma = .8

if args['--scale']:
    scale = float(args['--scale'])
else:
    scale = 20

if args['--seg']:
    minseg = args['--seg']
else:
    minseg = 4

convdate = False
if args['--convdate']:
    convdate = True

if args['sieve']:
    sieve(image, output, convdate)
elif args['fz']:
    segment_fz(image, output, scale, sigma, minseg, convdate)
elif args['kmeans']:
    segment_km(image, output)
