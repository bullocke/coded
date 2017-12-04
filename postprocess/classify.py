"""
Classify a multi-band, satellite image.
Script from Github user machinalis: 
https://github.com/machinalis/satimg/blob/master/classify.py
"""
import logging
import numpy as np
import os
import sys

from docopt import docopt
from osgeo import gdal, ogr
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.externals import joblib


logger = logging.getLogger(__name__)



def create_mask_from_vector(vector_data_path, cols, rows, geo_transform, 
			    projection, target_value=1,
                            output_fname='', dataset_format='MEM'):

    """
    Rasterize the given vector (wrapper for gdal.RasterizeLayer). 
    Return a gdal.Dataset.
    :param vector_data_path: Path to a shapefile
    :param cols: Number of columns of the result
    :param rows: Number of rows of the result
    :param geo_transform: Returned value of gdal.Dataset.GetGeoTransform 
	(coefficients for transforming between pixel/line (P,L) raster space,
	 and projection coordinates (Xp,Yp) space.
    :param projection: Projection definition string (Returned by 
	gdal.Dataset.GetProjectionRef)
    :param target_value: Pixel value for the pixels. Must be a valid 
	gdal.GDT_UInt16 value.
    :param output_fname: If the dataset_format is GeoTIFF, this is the output 
	file name
    :param dataset_format: The gdal.Dataset driver name. [default: MEM]
    """

    driver = ogr.GetDriverByName('ESRI Shapefile')
    data_source = driver.Open(vector_data_path, 0)
    if data_source is None:
        report_and_exit("File read failed: %s", vector_data_path)
    layer = data_source.GetLayer(0)
    driver = gdal.GetDriverByName(dataset_format)
    target_ds = driver.Create(output_fname, cols, rows, 1, gdal.GDT_UInt16)
    target_ds.SetGeoTransform(geo_transform)
    target_ds.SetProjection(projection)
    gdal.RasterizeLayer(target_ds, [1], layer, burn_values=[target_value])
    return target_ds


def vectors_to_raster(file_paths, rows, cols, geo_transform, projection):

    """
    Rasterize, in a single image, all the vectors in the given directory.
        The data of each file will be assigned the same pixel value. This value is 
        defined by the order of the file in file_paths, starting with 1: so the 
        points/poligons/etc in the same file will be
        marked as 1, those in the second file will be 2, and so on.
    :param file_paths: Path to a directory with shapefiles
    :param rows: Number of rows of the result
    :param cols: Number of columns of the result
    :param geo_transform: Returned value of gdal.Dataset.GetGeoTransform 
	(coefficients for transforming between pixel/line (P,L) raster space, 
	and projection coordinates (Xp,Yp) space.
    :param projection: Projection definition string (Returned by 
	gdal.Dataset.GetProjectionRef)
    """

    labeled_pixels = np.zeros((rows, cols))
    for i, path in enumerate(file_paths):
        label = i+1
        logger.debug("Processing file %s: label (pixel value) %i", path, label)
        ds = create_mask_from_vector(path, cols, rows, geo_transform, projection,
                                     target_value=label)
        band = ds.GetRasterBand(1)
        a = band.ReadAsArray()
        logger.debug("Labeled pixels: %i", len(a.nonzero()[0]))
        labeled_pixels += a
        ds = None
    return labeled_pixels


def write_geotiff(fname, data, geo_transform, projection, data_type=gdal.GDT_Byte):

    """
    Create a GeoTIFF file with the given data.
    :param fname: Path to a directory with shapefiles
    :param data: Number of rows of the result
    :param geo_transform: Returned value of 
	gdal.Dataset.GetGeoTransform (coefficients for transforming between 
	pixel/line (P,L) raster space, and projection coordinates (Xp,Yp) space.
    :param projection: Projection definition string (Returned by 
	gdal.Dataset.GetProjectionRef)
    """

    driver = gdal.GetDriverByName('GTiff')
    rows, cols = data.shape
    dataset = driver.Create(fname, cols, rows, 1, data_type)
    dataset.SetGeoTransform(geo_transform)
    dataset.SetProjection(projection)
    band = dataset.GetRasterBand(1)
    band.WriteArray(data)

    metadata = {
        'TIFFTAG_COPYRIGHT': 'CC BY 4.0',
        'TIFFTAG_DOCUMENTNAME': 'classification',
        'TIFFTAG_IMAGEDESCRIPTION': 'Supervised classification.',
        'TIFFTAG_SOFTWARE': 'Python, GDAL, scikit-learn'
    }
    dataset.SetMetadata(metadata)

    dataset = None  # Close the file
    return


def report_and_exit(txt, *args, **kwargs):

    """ Report error and exit program """

    logger.error(txt, *args, **kwargs)
    exit(1)

def do_classify(config, raster_data_path):

    """ Master function for doing classification """

    trained_classifier = config['classification']['classifier']
    try:
        classifier = joblib.load(trained_classifier) 
    except:
        sys.exit() #TODO

    forestlabel = int(config['classification']['forestlabel'])

    gdal.UseExceptions()

    try:
        raster_dataset = gdal.Open(raster_data_path, gdal.GA_ReadOnly)
    except:
	print "Could not open input file in classify: %s" % raster_data_path
        sys.exit()

    geo_transform = raster_dataset.GetGeoTransform()
    proj = raster_dataset.GetProjectionRef()
    bands_data_before = []
    bands_data_after = []

    goodbands = config['classification']['before_bands']
    bands_to_use = [int(i) for i in goodbands.split(', ')]
    for b in bands_to_use:
        band = raster_dataset.GetRasterBand(b)
        bands_data_before.append(band.ReadAsArray())

    goodbands = config['classification']['after_bands']
    bands_to_use = [int(i) for i in goodbands.split(', ')]
    for b in bands_to_use:
        band = raster_dataset.GetRasterBand(b)
        bands_data_after.append(band.ReadAsArray())


    bands_data_before = np.dstack(bands_data_before)
    bands_data_after = np.dstack(bands_data_after)
    rows_before, cols_before, n_bands_before = bands_data_before.shape
    rows_after, cols_after, n_bands_after = bands_data_after.shape

    # sample.
    n_samples_before = rows_before*cols_before
    n_samples_after = rows_after*cols_after

    logger.debug("Process the training data")

    flat_pixels_before = bands_data_before.reshape((n_samples_before, 
						    n_bands_before))
    flat_pixels_after = bands_data_after.reshape((n_samples_after, 
						  n_bands_after))

    flat_pixels_before[np.isnan(flat_pixels_before)] = 0
    flat_pixels_after[np.isnan(flat_pixels_after)] = 0
    # Perform classification

    logger.debug("Classifing...")
    result_before = classifier.predict(flat_pixels_before)
    result_after = classifier.predict(flat_pixels_after)

    # Reshape the result: split the labeled pixels into rows to create an image
    classification_before = result_before.reshape((rows_before, cols_before))
    classification_after = result_after.reshape((rows_after, cols_after))

    # Turn non-change pixels to 0
    classification_before_full = np.copy(classification_before)

    classification_before[bands_data_before[:,:,0] == 0] = 0
    classification_after[bands_data_after[:,:,0] == 0] = 0
    classification_before[classification_after == 0] = 0

    deg_array = np.copy(classification_before)

    forest_to_forest = np.logical_and(classification_before == forestlabel, 
				      classification_after == forestlabel)

    deg_array[forest_to_forest] = 1
    deg_array[~forest_to_forest] = 0

    def_array = np.copy(classification_before)

    deforestation = np.logical_and(classification_before == forestlabel, 
				   classification_after != forestlabel)

    def_array[deforestation] = 1
    def_array[~deforestation] = 0

    output1 = config['classification']['before_output']
    output2 = config['classification']['after_output']
    output3 = config['classification']['f_t_f_output']
    output4 = config['classification']['def_output']
    #write output
    if output1:
        dstpath1 = output1 + '/' + raster_data_path.split('/')[-1].split('.')[0] + '_before.tif'
        write_geotiff(dstpath1, classification_before_full, geo_transform, proj)
    if output2:
        dstpath2 = output2 + '/' + raster_data_path.split('/')[-1].split('.')[0] + '_after.tif'
        write_geotiff(dstpath2, classification_after, geo_transform, proj)
    if output3:
        dstpath3 = output3 + '/' + raster_data_path.split('/')[-1].split('.')[0] + '_ftf.tif'
        write_geotiff(dstpath3, deg_array, geo_transform, proj)
    if output4:
        dstpath4 = output4 + '/' + raster_data_path.split('/')[-1].split('.')[0] + '_deforestation.tif'
        write_geotiff(dstpath4, def_array, geo_transform, proj)

    return classification_before_full, classification_before, classification_after, deg_array, def_array
