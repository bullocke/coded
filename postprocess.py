import cv2, sys
import pymeanshift as pms
import numpy as np
import gdal
import scipy.stats
from skimage.color import rgb2gray
#from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, slic, quickshift#, watershed
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
import time

#image="/projectnb/landsat/users/bullocke/Rondonia/Done_YATSM/232066/GEE/GE_23266_2000_2010_subset_3.tif"
#output="/projectnb/landsat/users/bullocke/Rondonia/Done_YATSM/232066/GEE/GE_23266_2000_2010_subset_3_fz_3.tif"
#output="/projectnb/landsat/users/bullocke/Rondonia/Done_YATSM/232066/GEE/GE_23266_2000_2010_subset_3_km.tif"
image="/projectnb/landsat/users/bullocke/Rondonia/Done_YATSM/232066/GEE/GE_232066_2000_2010_4_subset.tif"
output="/projectnb/landsat/users/bullocke/Rondonia/Done_YATSM/232066/GEE/GE_232066_2000_2010_4_subset_fz.tif"


def save_raster(array, path, dst_filename):
    example = gdal.Open(path)
    x_pixels = array.shape[1]  # number of pixels in x
    y_pixels = array.shape[0]  # number of pixels in y
    driver = gdal.GetDriverByName('GTiff')
    dataset = driver.Create(dst_filename,x_pixels, y_pixels, 3,gdal.GDT_Int32) #TODO: bands
#    dataset.GetRasterBand(1).WriteArray(array[:,:])
    dataset.GetRasterBand(1).WriteArray(array[:,:,0])
    dataset.GetRasterBand(2).WriteArray(array[:,:,1])
    dataset.GetRasterBand(3).WriteArray(array[:,:,2])

    # follow code is adding GeoTranform and Projection
    geotrans=example.GetGeoTransform()  #get GeoTranform from existed 'data0'
    proj=example.GetProjection() #you can get from a exsited tif or import 
    dataset.SetGeoTransform(geotrans)
    dataset.SetProjection(proj)
    dataset.FlushCache()
    dataset=None

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

def segment_fz(image, output):
    original_im = gdal.Open(image)

    #Assign median values based on Felzenzwalb segmentation algorithm
    three_d_image = original_im.ReadAsArray().astype(np.uint8)
    three_d_image = three_d_image.swapaxes(0, 2)
    three_d_image = three_d_image.swapaxes(0, 1)
    img = three_d_image

    full_image = original_im.ReadAsArray()
    full_image = full_image.swapaxes(0, 2)
    full_image = full_image.swapaxes(0, 1)

    segments_fz = felzenszwalb(img, scale=20, sigma=0.2, min_size=8)
#    save_raster(segments_fz, image, output)
#    sys.exit()
    median_image = np.zeros_like(img).astype(np.float32)
    for band in range(3):
        for seg in np.unique(segments_fz):
            values = full_image[:,:,band][segments_fz == seg]
	    if np.median(values) > 0:
                med = np.median(values[values>0])
            else:
                med = 0
            median_image[:,:,band][segments_fz == seg] = med

    save_raster(median_image, image, output)
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
#            med = np.mean(values)
            mode = scipy.stats.mode(values.astype(int)).mode[0]
            if mode > 0:
                med = np.mean(values[values>0])
            else:
                med = 0
            median_image[:,:,band][segments_slic == seg] = med

    save_raster(median_image, image, output)
    sys.exit()

def sieve(image, dst_filename):
    #First create a band in memory that's that's just 1s and 0s
    src_ds = gdal.Open( image, gdal.GA_ReadOnly )
    srcband = src_ds.GetRasterBand(1)
    srcarray = srcband.ReadAsArray()
    srcarray[srcarray > 0] = 10
    #save_raster(srcarray, image, dst_filename)
    #sys.exit()
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

    prog_func = None

    threshold = 5

    connectedness = 4

    result = gdal.SieveFilter(mem_band, maskband, dstband,
                              threshold, connectedness,
                              callback = prog_func )
    import pdb; pdb.set_trace()


#sieve(image, output)
segment_fz(image, output)
#segment_km(image, output)
