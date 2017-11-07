#!/usr/bin/env python
# -*- coding: UTF-8 -*-

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
from skimage.measure import label, regionprops
import time

def extend_nonforest(config, buffered_array, full_array):
   """ After buffering forest, get changes that are spatially connected i
       to previous change or non-forest
   """

   # convert full array into connected labels
   ones_array = np.copy(full_array[0,:,:])
   ones_array[ones_array > 0] = 1
   ones_array[ones_array != 1] = 0

   label_image = label(ones_array, neighbors=8)
   labels = np.unique(label_image)

   for _label in labels:
	indices = np.where(label_image == _label)
        if ones_array[indices][0] > 0:
	    connected_parts = buffered_array[0,:,:][indices].sum()
	    if connected_parts > 0:
	        for i in range(full_array.shape[0]):
	            buffered_array[i,:,:][indices] = full_array[i,:,:][indices]

   return buffered_array
#   skimage.measure.label
#   skimage.measure.regionprobs
   # maybe loop over each label? if any in corresponding values in 
   #   buffered array are non-forest or previous change add change
   #   data to out array

   # compare to change map

def buffer_nonforest(config, image, array, mask_array, change_array):

    """ 
    Iterate over years, and buffer previous non-forest or change areas
    If a low-magnitude pixel does not lie within the buffer area, remove it
    """

    out_array = np.zeros_like(array)
    #Non-forest pixels are masked, so use this infomration to create forest/nf raster
    im_open = gdal.Open(image)

    forest_value = config['classification']['forestlabel']
    dist = config['postprocessing']['buffer']['distance']
    mag_ind = config['general']['mag_band'] - 1
    mag_thresh = float(config['postprocessing']['buffer']['mag_thresh'])
    change_thresh = float(config['postprocessing']['buffer']['change_thresh'])

    #mask array = land cover classification before disturbance
    mask_array[mask_array == forest_value] = 0
    mask_array[mask_array > 0] = 1

    mask_array = raster_buffer(image, mask_array, dist=dist)

    high_mag = np.logical_or(array[mag_ind,:,:] < mag_thresh, change_array < change_thresh)
    mask_array[high_mag] = 1


    for i in range(array.shape[0]):
	out_array[i,:,:][mask_array == 1] = array[i,:,:][mask_array == 1]
	out_array[i,:,:][high_mag] = array[i,:,:][high_mag]

    for year in range(2001, 2017):
	newmask = np.zeros_like(mask_array)
	newmask[out_array[0,:,:] == (year)] = 1
        newmask = raster_buffer(image, newmask, dist=5)
	mask_array += newmask
        for i in range(array.shape[0]):
	    out_array[i,:,:][mask_array == 1] = array[i,:,:][mask_array == 1]

#    for i in range(array.shape[0]):
#	out_array[i,:,:][high_mag] = array[i,:,:][high_mag]

    return out_array


def raster_buffer(raster_filepath, in_array, dist=1):
    """ Binary dilation using scikit image """
    struct = ndimage.generate_binary_structure(2, 2)
    out_array = ndimage.binary_dilation(in_array, structure=struct,iterations=dist).astype(in_array.dtype)
    #save_raster_simple(out_array, raster_filepath, 'test_50dilation.tif')
    return out_array


def convert_date(config, array):
    date_band = config['general']['date_band'] - 1
    array[date_band,:,:][array[date_band,:,:] > 0] += 1970
    array[date_band,:,:] = array[date_band,:,:].astype(np.int)
    return array

def get_geom_feats(config, array):
    # Add extra bands to the array for:
	# 1. Max magnitude in X pixel window
	# 2. Min magnitude in X pixel window
	# 3. Mean magnitude in X pixel window
	# 4+ TODO: area, shape, etc? 

    mag = config['general']['mag_band'] - 1
    mag = array[mag_band,:,:]

    window = config['postprocessing']['window']['window_size']
 
    max_mag = ndimage.generic_filter(mag, np.max, size=window)

    min_mag = ndimage.generic_filter(mag, np.min, size=window)

    mean_mag = ndimage.generic_filter(mag, np.mean, size=window)

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

def convert_change_dif(config, array):

    NFDI_ind_bef = config['general']['nfdi_before'] - 1
    NFDI_ind_aft = config['general']['nfdi_after'] -1

    # Get percent change
    newar = (array[NFDI_ind_aft,:,:] / array[NFDI_ind_bef,:, :]) * 100 
    newar[np.isnan(newar)] = 0

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


def save_raster(array, path, dst_filename):


    #Multiply to save space
#    array[1:,:,:]*=1000

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

def sieve(config, image):

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

    #Need to output sieved file
    dst_path = config['postprocessing']['sieve']['sieve_file']
    dst_filename = dst_path + '/' + image.split('/')[-1].split('.')[0] + '_sieve.tif'

    dst_ds = drv.Create( dst_filename,src_ds.RasterXSize, src_ds.RasterYSize,1,
                         srcband.DataType )
    wkt = src_ds.GetProjection()
    if wkt != '':
        dst_ds.SetProjection( wkt )
    dst_ds.SetGeoTransform( src_ds.GetGeoTransform() )

    dstband = dst_ds.GetRasterBand(1)

    # Parameters
    prog_func = None

    threshold = config['postprocessing']['sieve']['threshold']
    connectedness = config['postprocessing']['sieve']['connectedness']
    if connectedness not in [8, 4]:
	print "connectness only supports value of 4 or 8"
	sys.exit()

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


    dst_full = config['postprocessing']['sieve']['sieved_output']

    if dst_full:
        save_raster(out_img, image, dst_full, convdate, False, change_dif)

    return out_img

def get_deg_magnitude(config, ftf, sieved, dif):

    date_band = config['general']['date_band'] - 1
    mag_band = config['general']['mag_band'] - 1

    rows, cols = np.shape(ftf)

    deg_array = np.zeros((3, rows, cols))

    deg_array[0,:,:] = sieved[date_band,:,:] * ftf

    deg_array[1,:,:] = (sieved[mag_band,:,:] * ftf) * 10000

    deg_array[2,:,:] = dif * ftf

    return deg_array

def min_max_years(config, image):
    min_year = int(config['postprocessing']['minimum_year'])
    if not min_year:
	min_year = 1990

    max_year = int(config['postprocessing']['maximum_year'])
    if not max_year:
	max_year = 2200

    bad_indices = np.logical_or(image[0,:,:] < min_year, image[0,:,:] > max_year)
    for i in range(image.shape[0]):
        image[i,:,:][bad_indices] = 0

    return image
