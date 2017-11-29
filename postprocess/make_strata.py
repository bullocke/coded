#!/usr/bin/env python
# -*- coding: UTF-8 -*-
""" Make strata for validation of degradation (CDD) results. The sampling scheme is a 
two-stage cluster sample. The first stage samples areas in which high resolution imagery 
is available on Google Earth within a certain amount of time after a disturbance. The 
second stage selects pixels within those clusters. This script creates the clusters for 
the first stage of sampling.

Inputs:
    shape:   shapefile containing extent and date of historical Google Earth Imagery
	     for now (TODO), the date must be in an attribute named "Descriptio".
    raster:  results from ge-cdd degradation monitoring. For now, the date of
              disturbance should be in the first band (TODO). 
Output:
     output:  geotiff image containing location of clusters for the first stage
	      of sampling. 

Usage: make_strata.py [options] <shape> <raster> <output>

  --days=<DAYS>           Days before the high-res image the disturbance can be

"""

from docopt import docopt
import gdal
from osgeo import ogr, osr
import os
import numpy as np
from datetime import datetime as dt

def main(shape, raster, maxdays):
    """ 

    Master function for creating strata for validation of ge-cdd (Google Earth Continuous
    Degradation Detection) results. 

    Inputs:
	shape:   shapefile containing extent and date of historical Google Earth Imagery
	         for now (TODO), the date must be in an attribute named "Descriptio".
	raster:  results from ge-cdd degradation monitoring. For now, the date of
		 disturbance should be in the first band (TODO). 
	maxdays: maximum amount of days after a disturbance allowable for a GE image
		 to be used for reference data
    Returns:
	out_ar:  output array containing locations in which high resolution imagery
		 is available within <DAYS> after disturbance

    """

    maxdays = 180

    im_op = gdal.Open(raster)
    im_ar = im_op.GetRasterBand(1).ReadAsArray() #TODO: Paramaterize band
    out_ar = np.zeros_like(im_ar)


    driver = ogr.GetDriverByName("ESRI Shapefile")
    dataSource = driver.Open(shape, 0)
    layer = dataSource.GetLayer()

    for iter in range(layer.GetFeatureCount()):
        feature = layer.GetFeature(iter)
        imagedate = feature.GetField("Descriptio")
	print imagedate
	imagedate = dt.strptime(imagedate,"%Y%m").toordinal()

        zoned, xoff, yoff, xcount, ycount = zonal_stats(feature, layer, gdal.Open(raster), 1, imagedate, maxdays)
	out_ar[yoff:(yoff+ycount),xoff(xoff+xcount)] = zoned

        # Turn areas not within window to 0

    layer.ResetReading()
    return out_ar


def zonal_stats(feat, lyr, raster, band, imagedate, maxdays):
    """ 

    Extract subset of raster for the region within a vector feature.

    Inputs:
	feat:      vector feature
	lyr:       vector layer
	raster:    input raster
	band:      band containing disturbance date
	imagedate: date of google earth image associated with current feature
	maxdays:   maximum days beyond disturbance that google earth image can 
		   be used
    Returns:
        outraster: raster within feature region within date range
	xoff:      pixel distance from top left corner of image region on x axis
	yoff:      pixel distance from top left corner of image region on y axis
        xcount:    shape of subset on x axis
        ycount:    shape of subset on y axis

    """

    # Open data

    # Get raster georeference info
    transform = raster.GetGeoTransform()
    xOrigin = transform[0]
    yOrigin = transform[3]
    pixelWidth = transform[1]
    pixelHeight = transform[5]

    # Reproject vector geometry to same projection as raster
    sourceSR = lyr.GetSpatialRef()
    targetSR = osr.SpatialReference()
    targetSR.ImportFromWkt(raster.GetProjectionRef())
    coordTrans = osr.CoordinateTransformation(sourceSR,targetSR)
   # feat = lyr.GetNextFeature()
    geom = feat.GetGeometryRef()
    geom.Transform(coordTrans)

    # Get extent of feat
    geom = feat.GetGeometryRef()
    if (geom.GetGeometryName() == 'MULTIPOLYGON'):
        count = 0
        pointsX = []; pointsY = []
        for polygon in geom:
            geomInner = geom.GetGeometryRef(count)
            ring = geomInner.GetGeometryRef(0)
            numpoints = ring.GetPointCount()
            for p in range(numpoints):
                    lon, lat, z = ring.GetPoint(p)
                    pointsX.append(lon)
                    pointsY.append(lat)
            count += 1
    elif (geom.GetGeometryName() == 'POLYGON'):
        ring = geom.GetGeometryRef(0)
        numpoints = ring.GetPointCount()
        pointsX = []; pointsY = []
        for p in range(numpoints):
                lon, lat, z = ring.GetPoint(p)
                pointsX.append(lon)
                pointsY.append(lat)

    else:
        sys.exit("ERROR: Geometry needs to be either Polygon or Multipolygon")

    xmin = min(pointsX)
    xmax = max(pointsX)
    ymin = min(pointsY)
    ymax = max(pointsY)

    # Specify offset and rows and columns to read
    xoff = int((xmin - xOrigin)/pixelWidth)
    yoff = int((yOrigin - ymax)/pixelWidth)

    xcount = int((xmax - xmin)/pixelWidth)+1
    ycount = int((ymax - ymin)/pixelWidth)+1

    # Create memory target raster
    target_ds = gdal.GetDriverByName('MEM').Create('', xcount, ycount, 1, gdal.GDT_Byte)
    target_ds.SetGeoTransform((
        xmin, pixelWidth, 0,
        ymax, 0, pixelHeight,
    ))

    # Create for target raster the same projection as for the value raster
    raster_srs = osr.SpatialReference()
    raster_srs.ImportFromWkt(raster.GetProjectionRef())
    target_ds.SetProjection(raster_srs.ExportToWkt())

    # Rasterize zone polygon to raster
    gdal.RasterizeLayer(target_ds, [1], lyr, burn_values=[1])

    # Read raster as arrays
    banddataraster = raster.GetRasterBand(band)

    rastX = raster.RasterXSize
    rastY = raster.RasterYSize

   
    dataraster = banddataraster.ReadAsArray(xoff, yoff, xcount, ycount).astype(np.float)
    
    bandmask = target_ds.GetRasterBand(1)
    datamask = bandmask.ReadAsArray(0, 0, xcount, ycount).astype(np.float)

    zoneraster = dataraster * datamask
    outraster = np.zeros_like(zoneraster)

    #Do the date stuff here
    #Ugly loop - but need to convert dates
    s1, s2 = zoneraster.shape
    for ind1 in range(s1):
        for ind2 in range(s2):
	    if zoneraster[ind1, ind2] == 0:
    	        continue
            try:
                pixeldate = dt.strptime(str(int(zoned[ind1, ind2])), '%Y%j').toordinal()
            except:
                pixeldate = dt.strptime(str(int(zoneraster[ind1, ind2]+ 1)), '%Y%j').toordinal()
            dif_dates = imagedate - pixeldate
	    if (dif_dates < 0) or (dif_dates > maxdays):
		continue 
	    else:
		outraster[ind1, ind2] = zoneraster[ind1, ind2]

    return outraster, xoff, yoff, xcount, ycount

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

if __name__ == '__main__':
    args = docopt(__doc__, version='0.6.2')

shape=args['<shape>']
raster=args['<raster>']
outputname=args['<output>']

if args['--days']:
    maxdays = args['--days']
else:
    maxdays = 180

output = main(shape, raster)

save_raster_simple(output, raster, outputname)
