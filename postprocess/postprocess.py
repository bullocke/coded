#!/usr/bin/env python
# -*- coding: UTF-8 -*-
""" Postprocess Continuous Degradation Detection (CDD) results.

Usage: postprocess_cdd.py [options] (sieve | fz | kmeans) <input> <output>

  --seg=<SEG_SIZE>        Minimum segment size 
  --sigma=<SIGMA>         Sigma value for FZ test
  --scale=<SCALE>         Scale value for FZ test
  --convdate              Convert date to year
  --change_dif            Convert NFDI bef/after change to difference

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
import time

def buffer_nonforest(image, array, 
		       bef_int_band=4, mag_ind=1, mag_thresh=-.3,
		       change_ind=9,change_thresh=.1):

    """ 
    Iterate over years, and buffer previous non-forest or change areas
    If a low-magnitude pixel does not lie within the buffer area, remove it
    """

    out_array = np.zeros_like(array)
    #Non-forest pixels are masked, so use this infomration to create forest/nf raster
    bef_int_band = 4
    im_open = gdal.Open(image)
    mask_array = im_open.GetRasterBand(bef_int_band).ReadAsArray()
    mask_array[~np.isnan(mask_array)] = 0
    mask_array[np.isnan(mask_array)] = 1
    mask_array = raster_buffer(image, mask_array, dist=5)
    high_mag = np.logical_or(array[mag_ind,:,:] < mag_thresh, array[change_ind,:,:] > change_thresh)
    mask_array[high_mag] = 1


    for i in range(array.shape[0]):
	out_array[i,:,:][mask_array == 1] = array[i,:,:][mask_array == 1]
	out_array[i,:,:][high_mag] = array[i,:,:][high_mag]

    for year in range(2001, 2017):
	newmask = np.zeros_like(mask_array)
	newmask[out_array[0,:,:] == (year-1)] = 1
        newmask = raster_buffer(image, newmask, dist=5)
	mask_array += newmask
        for i in range(array.shape[0]):
	    out_array[i,:,:][mask_array == 1] = array[i,:,:][mask_array == 1]

    return out_array


def raster_buffer(raster_filepath, in_array, dist=1):
    """ Binary dilation using scikit image """
    struct = ndimage.generate_binary_structure(2, 2)
    out_array = ndimage.binary_dilation(in_array, structure=struct,iterations=dist).astype(in_array.dtype)
    #save_raster_simple(out_array, raster_filepath, 'test_50dilation.tif')
    return out_array


def convert_date(array):
    array[0,:,:][array[0,:,:] > 0] += 1970
    array[0,:,:] = array[0,:,:].astype(np.int)
    return array

def get_geom_feats(array):
    # Add extra bands to the array for:
	# 1. Max magnitude in 5 pixel window
	# 2. Min magnitude in 5 pixel window
	# 3. Mean magnitude in 5 pixel window
	# 4+ TODO: area, shape, etc? 

    mag = array[1,:,:]

    max_mag = ndimage.generic_filter(mag, np.max, size=5)

    min_mag = ndimage.generic_filter(mag, np.min, size=5)

    mean_mag = ndimage.generic_filter(mag, np.mean, size=5)

    dim1, dim2, dim3 = np.shape(array)

    newar = np.zeros((dim1+3,dim2,dim3))

    newar[0:dim1,:,:] = array

    newar[-3,:,:] = max_mag

    newar[-2,:,:] = min_mag

    newar[-1,:,:] = mean_mag

    newar[-3,:,:][newar[0, :, :] == 0] = 0
    newar[-2,:,:][newar[0, :, :] == 0] = 0
    newar[-1,:,:][newar[0, :, :] == 0] = 0

    return newar

def convert_change_dif(array):
    # add NFDI difference band
    dim1, dim2, dim3 = np.shape(array)
    newar = np.zeros((dim1+1,dim2,dim3))
    newar[0:dim1,:,:] = array

    NFDI_ind_bef = 3
    NFDI_ind_aft = 2

    newar[-1,:,:] = array[NFDI_ind_bef,:,:] - array[NFDI_ind_aft, :, :]

    return newar
    
def save_raster_simple(array, path, dst_filename):


    example = gdal.Open(path)
    x_pixels = array.shape[1]  # number of pixels in x
    y_pixels = array.shape[0]  # number of pixels in y
    bands = 1
    driver = gdal.GetDriverByName('GTiff')
    dataset = driver.Create(dst_filename,x_pixels, y_pixels, bands ,gdal.GDT_Int32)

    geotrans=example.GetGeoTransform()  #get GeoTranform from existed 'data0'
    proj=example.GetProjection() #you can get from a exsited tif or import 
    dataset.SetGeoTransform(geotrans)
    dataset.SetProjection(proj)

    dataset.GetRasterBand(1).WriteArray(array[:,:])

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


def save_raster(array, path, dst_filename, convdate, geomfeats, change_dif):

    #Convert date from years since 1970 to year
    if convdate:
        array = convert_date(array)

    # Get geometry features
    if geomfeats:
        array = get_geom_feats(array)

    if change_dif:
	array = convert_change_dif(array)

    buffer = True
    if buffer:
        array = buffer_nonforest(path, array) 

    #Multiply to save space
    array[1:,:,:]*=1000

    example = gdal.Open(path)
    x_pixels = array.shape[2]  # number of pixels in x
    y_pixels = array.shape[1]  # number of pixels in y
    bands = array.shape[0]
    driver = gdal.GetDriverByName('GTiff')
    dataset = driver.Create(dst_filename,x_pixels, y_pixels, bands ,gdal.GDT_Int32)

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

def sieve(image, dst_filename, convdate, change_dif):

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
    dst_sieve = dst_filename.split('.')[0] + '_sieveFilter.tif'

    dst_ds = drv.Create( dst_filename,src_ds.RasterXSize, src_ds.RasterYSize,1,
                         srcband.DataType )
    wkt = src_ds.GetProjection()
    if wkt != '':
        dst_ds.SetProjection( wkt )
    dst_ds.SetGeoTransform( src_ds.GetGeoTransform() )

    dstband = dst_ds.GetRasterBand(1)

    # Parameters
    prog_func = None
    threshold = 6
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


   # Get buffered f/nf mask
#    buf_nf = buffer_nonforest(image, out_img) 


    dst_full = dst_filename.split('.')[0] + '_sieved.tif'

    save_raster(out_img, image, dst_full, convdate, False, change_dif)
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

if args['--change_dif']:
    change_dif = True
else:
    change_dif = False

convdate = False
if args['--convdate']:
    convdate = True

if args['sieve']:
    sieve(image, output, convdate,change_dif)
elif args['fz']:
    segment_fz(image, output, scale, sigma, minseg, convdate)
elif args['kmeans']:
    segment_km(image, output)
