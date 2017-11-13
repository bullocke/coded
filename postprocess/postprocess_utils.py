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

def do_deg_classification(config, input, deg_mag, before_copy, raw_input):
    """ Classify degradation event based on degradation class (fire, logging,
        water, noise, etc. Not being utilized at the moment. """

    # Get proximity features 
    ag_label = int(config['postprocessing']['deg_class']['ag_label'])
    water_label = int(config['postprocessing']['deg_class']['water_label'])
    dev_label = int(config['postprocessing']['deg_class']['dev_label'])

    ag_prox = do_proximity(config, input, before_copy, ag_label, 'ag')
    water_prox = do_proximity(config, input, before_copy,water_label, 'water')
    dev_prox = do_proximity(config, input, before_copy, dev_label, 'dev')

    # Get shape features
    area, perimeter = get_shape_features(config, deg_mag, input)

    # Get change vector information
    changev_classes = config['postprocessing']['deg_class']['changev_classes']
    changev_classes = [int(i) -1 for i in changev_classes.split(', ')]

    #create stack to classify
    stack = np.vstack((deg_mag[1,:,:][np.newaxis,:,:], 
		        deg_mag[2,:,:][np.newaxis,:,:],
			raw_input[changev_classes[0],:,:][np.newaxis,:,:],
			raw_input[changev_classes[1],:,:][np.newaxis,:,:],
			raw_input[changev_classes[2],:,:][np.newaxis,:,:],
			raw_input[changev_classes[3],:,:][np.newaxis,:,:],
 		 	ag_prox[np.newaxis,:,:], 
			water_prox[np.newaxis,:,:], 
			dev_prox[np.newaxis,:,:],
			area[np.newaxis,:,:],
			perimeter[np.newaxis,:,:]))
    raw_input = None

    deg_training = config['postprocessing']['deg_class']['deg_classifier']

def get_shape_features(config, deg_mag, _input):
    """ Get shape features of change polygons """
    
    # convert full array into connected labels
    ones_array = np.copy(deg_mag[0,:,:])
    ones_array[ones_array > 0] = 1
    ones_array[ones_array != 1] = 0

    # How to get label image?
    segment = config['postprocessing']['segmentation']['segment']
    if segment:
        seg_method = config['postprocessing']['segmentation']['method']
        if seg_method == 'fz':
            scale = int(config['postprocessing']['segmentation']['scale'])
            sigma = float(config['postprocessing']['segmentation']['sigma'])
            minseg = int(config['postprocessing']['segmentation']['minseg'])
            seg_image = segment_fz(deg_mag, scale, sigma, minseg)
	    seg_image[seg_image > 0] = 1
            label_image = label(seg_image, neighbors=8)
    else:
        label_image = label(ones_array, neighbors=8)

    props = regionprops(label_image)

    out_area = np.copy(label_image)
    out_perim = np.copy(label_image)

    for i in props:
        _label = i.label
        out_area[label_image == _label] = i.area
        out_perim[label_image == _label] = i.perimeter

    save_raster_simple(out_area, _input, 'test_area.tif')
    save_raster_simple(out_perim, _input, 'test_perim.tif')
    return out_area, out_perim

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

    return out_array

def raster_buffer(raster_filepath, in_array, dist=1):
    """ Binary dilation using scikit image """
    struct = ndimage.generate_binary_structure(2, 2)
    out_array = ndimage.binary_dilation(in_array, structure=struct,iterations=dist).astype(in_array.dtype)
    return out_array

def convert_date(config, array):
    """ Convert date from years since 1970 to year """
    date_band = config['general']['date_band'] - 1
    array[date_band,:,:][array[date_band,:,:] > 0] += 1970
    array[date_band,:,:] = array[date_band,:,:].astype(np.int)
    return array

def do_proximity(config, image, srcarray, label, _class):
    """ Get proximity of each pixel to to certain value """

    #First create a band in memory that's that's just 1s and 0s
    src_ds = gdal.Open( image, gdal.GA_ReadOnly )
    srcband = src_ds.GetRasterBand(1)
    mem_rast = save_raster_memory(srcarray, image)
    mem_band = mem_rast.GetRasterBand(1)

    #Now the code behind gdal_sieve.py
    maskband = None
    drv = gdal.GetDriverByName('GTiff')

    #Need to output sieved file
    dst_path = config['postprocessing']['deg_class']['prox_dir']
    
    dst_filename = dst_path + '/' + image.split('/')[-1].split('.')[0] + '_' + _class + '.tif'
    dst_ds = drv.Create( dst_filename,src_ds.RasterXSize, src_ds.RasterYSize,1,
                         srcband.DataType )

    wkt = src_ds.GetProjection()
    if wkt != '':
        dst_ds.SetProjection( wkt )
    dst_ds.SetGeoTransform( src_ds.GetGeoTransform() )

    dstband = dst_ds.GetRasterBand(1)

    # Parameters
    prog_func = None

    options = []
    options.append( 'VALUES=' + str(label) )

    result = gdal.ComputeProximity(mem_band, dstband, options)#, 
                              #callback = prog_func )

    proximity = dstband.ReadAsArray()

    return proximity

     


def get_geom_feats(config, array, before_class, input):

    # Add extra bands to the array for:
	# 1. Max magnitude in X pixel window
	# 2. Min magnitude in X pixel window
	# 3. Mean magnitude in X pixel window
	# 4+ TODO: area, shape, etc? 

    mag_band = config['general']['mag_band'] - 1
    mag = array[mag_band,:,:]

    forestlabel = int(config['classification']['forestlabel'])

    # create window
    before_class = before_class.astype(np.float)
    before_class[before_class == forestlabel] = np.nan
    before_class[before_class == 0] = np.nan

    window = config['postprocessing']['deg_class']['window_size']
 
    max_mag = ndimage.generic_filter(before_class, np.nanmax, size=window)

    max_mag[np.isnan(max_mag)] = 0

    save_raster_simple(max_mag, input, 'test_classwindow.tif')

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
    """ Turn difference in NDFI into percent change """

    NFDI_ind_bef = config['general']['nfdi_before'] - 1
    NFDI_ind_aft = config['general']['nfdi_after'] -1

    # Get percent change
    newar = (array[NFDI_ind_aft,:,:] / array[NFDI_ind_bef,:, :]) * 100 
    newar[np.isnan(newar)] = 0

    return newar
    
def save_raster_simple(array, path, dst_filename):
    """ Save an array base on an existing raster """

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

def save_raster(array, path, dst_filename):
    """ Save the final multiband array based on an existing raster """

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

def save_raster_memory(array, path):
    """ Save a raster into memory """
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

def segment_fz(image, scale, sigma, minseg):
    """ Segment array rater using Felzenwalb segentation"""

    three_d_image = image.swapaxes(0, 2)
    three_d_image = three_d_image.swapaxes(0, 1)
    image = three_d_image[:,:,(0,1,2)]

    segments_fz = felzenszwalb(image, scale=scale, sigma=sigma, min_size=minseg)

    return segments_fz

def segment_km(image, output):

    #Assign median values based on Felzenzwalb segmentation algorithm
    three_d_image = image.swapaxes(0, 2)
    three_d_image = three_d_image.swapaxes(0, 1)
    image = three_d_image[:,:,(0,1,2)]

    segments_slic = slic(image, n_segments=250, compactness=10, sigma=1)

    return segments_slic

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
    """ Convert raw data into a usable output"""
    #TODO: misleading function name

    date_band = config['general']['date_band'] - 1
    mag_band = config['general']['mag_band'] - 1

    rows, cols = np.shape(ftf)

    deg_array = np.zeros((3, rows, cols))

    deg_array[0,:,:] = sieved[date_band,:,:] * ftf

    deg_array[1,:,:] = (sieved[mag_band,:,:] * ftf) * 10000

    deg_array[2,:,:] = dif * ftf

    return deg_array

def min_max_years(config, image):
    """ Exclude data outside of min and max year desired """
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
