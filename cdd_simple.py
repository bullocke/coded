#!/usr/bin/env python
# -*- coding: UTF-8 -*-
""" Simple version of CDD for faster processing

Usage: GE_cdd.py [options] <output>

  --cloud=CLOUD     cloud threshold
  --path=PATH       path
  --row=ROW         row
  --consec=CONSEC   consecutive obs to trigger change (default: 5)
  --thresh=THRESH   change threshold (default: 3.5)
  --forest=FOREST   forest % cover threshold (default: 30)
  --aoi             Use an area of interest (must hard code)
  --cf=CF_THRESH    Cloud frqction threshold
  --rmse=RMSE       RMSE threshold for forest classification
  --mag=MAG         Magnitude threshold for forest classification
  --trend=TREND     Trend (slope) threshold: default: .05
  --year=YEAR       Year to stop training and start monitoring
  --asset           Save to EE Asset instead of drive
  --folder=FOLDER   Folder name to save to

"""

from docopt import docopt
import os,sys
import numpy as np
import datetime
import pandas as pd
import matplotlib.dates as mdates
from matplotlib import dates
from pylab import *

#Google earth engine API
import ee

# Initialize Earth Engine
ee.Initialize()
print('Earth Engine Initialized')

# Parse arguments
if __name__ == '__main__':
    args = docopt(__doc__, version='0.6.2')

path = None
row = None
pathrow = False

# Lots of ugly arg parsing, TODO

if args['--path']:
    path = int(args['--path'])
    pathrow = True

if args['--row']:
    row = int(args['--row'])
elif pathrow:
    print('need to supply row with path')
    sys.exit()

if args['--consec']:
    consec = int(args['--consec'])
else:
    consec = 5

if args['--trend']:
    trend = float(args['--trend'])
else:
    trend = .05

if args['--thresh']:
    thresh = float(args['--thresh'])
else:
    thresh = 3.5

if args['--forest']:
    forest_threshold = int(args['--forest'])
else:
    forest_threshold = 30

if args['--cloud']:
    cloud_score = int(args['--cloud'])
else:
    cloud_score = 30

if args['--cf']:
    cf_thresh = float(args['--cf'])
else:
    cf_thresh = .2

if args['--rmse']:
    rmse_thresh = float(args['--rmse'])
else:
    rmse_thresh = .3

if args['--mag']:
    mag_thresh = float(args['--mag'])
else:
    mag_thresh = .6

if args['--year']:
    year = int(args['--year'])
else:
    year = 2000

aoi = False
if args['--aoi']:
    aoi = True

asset = False
if args['--asset']:
    asset = True

if args['--folder']:
    folder_name = args['--folder']
else:
    if asset:
        print('Need to specify asset folder')
        sys.exit()
    else:
        folder_name = None


output=str(args['<output>'])
print(output)

#GLOBALS
global AOI

#232 66 MC
AOI = ee.Geometry.Polygon(
        [[[-63.3806848526001, -8.67814233247136],
          [-63.38132858276367, -8.684421036203078],
          [-63.37589979171753, -8.684463459519577],
          [-63.375492095947266, -8.677781728053864]]])

#Try 2
gv= [500, 900, 400, 6100, 3000, 1000]
npv= [1400, 1700, 2200, 3000, 5500, 3000]
soil= [2000, 3000, 3400, 5800, 6000, 5800]
shade= [0, 0, 0, 0, 0, 0]
cloud = [9000, 9600, 8000, 7800, 7200, 6500]

# Hansen forest cover
if aoi:
    forest2000 = ee.Image('UMD/hansen/global_forest_change_2015_v1_3').select('treecover2000').clip(AOI)
else:
    forest2000 = ee.Image('UMD/hansen/global_forest_change_2015_v1_3').select('treecover2000')

# ** FUNCTIONS **

# Collection map functions

def unmix(image):
  unmixi = ee.Image(image).unmix([gv, shade, npv, soil, cloud], True, True)
  newimage = ee.Image(image).addBands(unmixi)
  mask = ee.Image(newimage).select('band_4').lt(cf_thresh)
  return newimage.updateMask(mask)

# NFDI functions
def get_nfdi(image):
  newimage = ee.Image(image).expression(
      '((GV / (1 - SHADE)) - (NPV + SOIL)) / ((GV / (1 - SHADE)) + NPV + SOIL)', {
        'GV': ee.Image(image).select('band_0'),
        'SHADE': ee.Image(image).select('band_1'),
        'NPV': ee.Image(image).select('band_2'),
        'SOIL': ee.Image(image).select('band_3')
      })
  return ee.Image(image).addBands(ee.Image(newimage).rename(['NFDI'])).select(['band_0','band_1','band_2','band_3','NFDI'])

# Regression functions

# This function computes the predictors and the response from the input.
def makeVariables(image):
  # Compute time of the image in fractional years relative to the Epoch.
  year = ee.Image(image.date().difference(ee.Date('1970-01-01'), 'year')) # this is the years since 1970
  # Compute the season in radians, one cycle per year.
  season = year.multiply(2 * np.pi)
  # Return an image of the predictors followed by the response.
  return image.select().addBands(ee.Image(1)).addBands(
#    year.rename(['t'])).addBands(
    season.sin().rename(['sin'])).addBands(
    season.cos().rename(['cos'])).addBands(
    image.select(['NFDI'])).toFloat()

def makeVariables_gv(image):
  # Compute time of the image in fractional years relative to the Epoch.
  year = ee.Image(image.date().difference(ee.Date('1970-01-01'), 'year')) # this is the years since 1970
  # Compute the season in radians, one cycle per year.
  season = year.multiply(2 * np.pi)
  # Return an image of the predictors followed by the response.
  return image.select().addBands(ee.Image(1)).addBands(
#    year.rename(['t'])).addBands(
    season.sin().rename(['sin'])).addBands(
    season.cos().rename(['cos'])).addBands(
    image.select(['band_0'])).toFloat()

def makeVariables_npv(image):
  # Compute time of the image in fractional years relative to the Epoch.
  year = ee.Image(image.date().difference(ee.Date('1970-01-01'), 'year')) # this is the years since 1970
  # Compute the season in radians, one cycle per year.
  season = year.multiply(2 * np.pi)
  # Return an image of the predictors followed by the response.
  return image.select().addBands(ee.Image(1)).addBands(
#    year.rename(['t'])).addBands(
    season.sin().rename(['sin'])).addBands(
    season.cos().rename(['cos'])).addBands(
    image.select(['band_2'])).toFloat()

def makeVariables_soil(image):
  # Compute time of the image in fractional years relative to the Epoch.
  year = ee.Image(image.date().difference(ee.Date('1970-01-01'), 'year')) # this is the years since 1970
  # Compute the season in radians, one cycle per year.
  season = year.multiply(2 * np.pi)
  # Return an image of the predictors followed by the response.
  return image.select().addBands(ee.Image(1)).addBands(
#    year.rename(['t'])).addBands(
    season.sin().rename(['sin'])).addBands(
    season.cos().rename(['cos'])).addBands(
    image.select(['band_3'])).toFloat()

def makeVariables_shade(image):
  # Compute time of the image in fractional years relative to the Epoch.
  year = ee.Image(image.date().difference(ee.Date('1970-01-01'), 'year')) # this is the years since 1970
  # Compute the season in radians, one cycle per year.
  season = year.multiply(2 * np.pi)
  # Return an image of the predictors followed by the response.
  return image.select().addBands(ee.Image(1)).addBands(
#    year.rename(['t'])).addBands(
    season.sin().rename(['sin'])).addBands(
    season.cos().rename(['cos'])).addBands(
    image.select(['band_1'])).toFloat()

def makeVariables_fullem(image):
  # Compute time of the image in fractional years relative to the Epoch.
  year = ee.Image(image.date().difference(ee.Date('1970-01-01'), 'year')) # this is the years since 1970
  # Compute the season in radians, one cycle per year.
  season = year.multiply(2 * np.pi)
  # Return an image of the predictors followed by the response.
  #return image.select().addBands(ee.Image(1)).rename('constant').addBands(
  return image.select().addBands(ee.Image(1)).addBands(
#    year.rename(['t'])).addBands(
    season.sin().rename(['sin'])).addBands(
    season.cos().rename(['cos'])).addBands(
    image.select(['NFDI','band_0','band_1','band_2','band_3'])).toFloat()

#For the regrowth test
def makeVariables_trend(image):
  # Compute time of the image in fractional years relative to the Epoch.
  year = ee.Image(image.date().difference(ee.Date('1970-01-01'), 'year')) # this is the years since 1970
  # Compute the season in radians, one cycle per year.
  season = year.multiply(2 * np.pi)
  # Return an image of the predictors followed by the response.
  return image.select().addBands(ee.Image(1)).addBands(
    year.rename(['t'])).addBands(
    season.sin().rename(['sin'])).addBands(
    season.cos().rename(['cos'])).addBands(
    image.select(['NFDI'])).toFloat()


# Add coefficients to image
def addcoefs(image):
  # Add regression coeffiecients to an imge
  newimage = ee.Image(image).addBands(ee.Image(coefficientsImage))
  return newimage

# Prediction function
def predict_nfdi(image):
  # Predict NDFI based on regression coefficients
  pred_nfdi_first = ee.Image(image).expression(
    'constant + (coef_sin * sin) + (coef_cos * cos)', {
      't': ee.Image(image).select('t'),
      'constant': ee.Image(image).select('coef_constant'),
      'coef_sin': ee.Image(image).select('coef_sin'),
      'sin': ee.Image(image).select('sin'),
      'coef_cos': ee.Image(image).select('coef_cos'),
      'cos': ee.Image(image).select('cos')
    })
  return ee.Image(image).addBands(ee.Image(pred_nfdi_first).rename(['Predict_NFDI']))

# For the regrowth test
def predict_nfdi_trend(image):
  # Predict NDFI with regression coefficients with trend (slope) term
  pred_nfdi_first = ee.Image(image).expression(
    'constant + (coef_trend * t) + (coef_sin * sin) + (coef_cos * cos)', {
      't': ee.Image(image).select('t'),
      'constant': ee.Image(image).select('coef_constant'),
      'coef_trend': ee.Image(image).select('coef_trend'),
      'coef_sin': ee.Image(image).select('coef_sin'),
      'sin': ee.Image(image).select('sin'),
      'coef_cos': ee.Image(image).select('coef_cos'),
      'cos': ee.Image(image).select('cos')
    })
  return ee.Image(image).addBands(ee.Image(pred_nfdi_first).rename(['Predict_NFDI']))

#Prediction function for retrain period - remove seasonaility
def pred_middle_retrain(retrain_middle, retrain_coefs):
  # Predict NFDI at middle of retraining period
  image = ee.Image(retrain_middle).addBands(retrain_coefs)
  pred_nfdi_imd = ee.Image(image).select('Intercept')
  if aoi:
      return pred_nfdi_imd.clip(AOI)
  else:
      return pred_nfdi_imd

# Add standard deviation band
def addmean(image):
  # Add RMSE band to image
  newimage = ee.Image(image).addBands(ee.Image(train_nfdi_mean))
  return newimage

def get_mean_residuals(image):
  # Return RMSE of time series residuals
  res = ee.Image(image).select('NFDI').subtract(image.select('Predict_NFDI'))
  
  sq_res = ee.Image(res).multiply(ee.Image(res)).rename(['sq_res'])

  full_image = ee.Image(image).addBands(ee.Image(sq_res))
  return full_image.select('sq_res')

def get_mean_residuals_norm(image):
  # Get normalized residuals
  res = image.select('NFDI').subtract(image.select('Predict_NFDI')).divide(ee.Image(train_nfdi_mean)).rename(['res'])
  return image.addBands(res)

def add_changedate_mask(image):
  # Mask if before changedate
  image_date = image.metadata('system:time_start').divide(ee.Image(315576e5))
  eq_change = image_date.eq(ee.Image(change_dates))
  new_band = ee.Image(1).multiply(image.select('NFDI')).updateMask(eq_change).rename('Change')
  return image.addBands(new_band)

def get_abs(image):
  # Get absulte value of residuals
  abs = image.select('res').abs().rename('abs_res')
  return image.addBands(abs)

# Main monitoring function
# status = 4 band image with status of change detection:
    # 1. Change (1) or no change (0). Used as mask. Default: 0
    # 2. Consecutive observations passed threshold. Default: 0
    # 3. Date of change if 1 = 1. Default: 0
    # 4. Magnitude of change
    # 5. iterator 
    
# image = image in collection 
def monitor_func(image, new_image):
  # Apply mask
  #define working mask. 1 if no change has been detected and no coud
  zero_mask = ee.Image(0)

  #MASKED WITH ZERO MASK
  zero_mask_nc = ee.Image(new_image).select('band_1').eq(ee.Image(1)).multiply( # band 1 != 0
                 ee.Image(new_image).select('band_2').lt(ee.Image(consec))).unmask() # not passed consec thresh
               
  cloud_mask = ee.Image(image).select('NFDI').mask().neq(ee.Image(0))
  image_monitor = image.unmask()

  # Get normalized residual
  norm_dif = image_monitor.select('NFDI').subtract(image_monitor.select('Predict_NFDI'))
  norm_res = ee.Image(norm_dif).divide(ee.Image(train_nfdi_mean))

  # passed_thresh = Find if it is passed threshold (1 = passed threshold)
  # 1 if norm_res > threshold
  # 0 if norm res < threshold, cloud mask
  # 
  gt_thresh = ee.Image(norm_res).abs().gt(ee.Image(thresh)).multiply(zero_mask_nc).multiply(cloud_mask)
  
  # Band 2: Consecutive observations beyond threshold
  # = # consecutive observations > threshold
  _band_2 = ee.Image(new_image).select('band_2').add(gt_thresh).multiply(ee.Image(new_image).select('band_1'))
  band_2_increase = _band_2.gt(ee.Image(new_image).select('band_2')).Or(cloud_mask.eq(ee.Image(0))) #This was added to not reset w/cloud
  band_2 = _band_2.multiply(band_2_increase)

  # flag_change = where band 2 of status is at consecutive threshold (5)
  # 1 if consec is reached
  # 0 if not consec
  flag_change = band_2.eq(ee.Image(consec))
  
  # Change status
  # 1 = no change
  # 0 = change
  band_1 = ee.Image(new_image).select('band_1').eq(ee.Image(1)).And(flag_change.eq(ee.Image(0)))

  # add date of image to band 3 of status where change has been flagged
  # time is stored in milliseconds since 01-01-1970, so scale to be days since then 
  #change_date = image.metadata('system:time_start').multiply(flag_change).divide(ee.Image(8.64e7))

  # New: actual date
  change_date = image.metadata('system:time_start').multiply(gt_thresh).divide(ee.Image(315576e5)).multiply(ee.Image(new_image).select('band_3').eq(ee.Image(0)))

  band_3 = ee.Image(new_image).select('band_3').add(change_date).multiply(ee.Image((gt_thresh.eq(1).Or(cloud_mask.eq(0)).Or(ee.Image(new_image).select('band_1').eq(0)))))

  # Magnitude and endmembers
  magnitude = ee.Image(norm_dif).multiply(gt_thresh).multiply(zero_mask_nc)
  endm1 = ee.Image(image_monitor).select('band_0').multiply(gt_thresh).multiply(zero_mask_nc)
  endm2 = ee.Image(image_monitor).select('band_1').multiply(gt_thresh).multiply(zero_mask_nc)
  endm3 = ee.Image(image_monitor).select('band_2').multiply(gt_thresh).multiply(zero_mask_nc)
  endm4 = ee.Image(image_monitor).select('band_3').multiply(gt_thresh).multiply(zero_mask_nc)

  #Keep mag if already changed or in process
  is_changing = band_1.eq(ee.Image(0)).Or(band_2).gt(ee.Image(0))

  band_4 = ee.Image(new_image).select('band_4').add(magnitude).multiply(is_changing)

  # Add one to iteration band
  band_5 = ee.Image(new_image).select('band_5').add(ee.Image(1))
  
  # Add endmember bands
  band_6 = ee.Image(new_image).select('band_6').add(endm1).multiply(is_changing)
  band_7 = ee.Image(new_image).select('band_7').add(endm2).multiply(is_changing)
  band_8 = ee.Image(new_image).select('band_8').add(endm3).multiply(is_changing)
  band_9 = ee.Image(new_image).select('band_9').add(endm4).multiply(is_changing)

  new_image = band_1.addBands([band_2,band_3,band_4,band_5,band_6,band_7,band_8,band_9])
  
  return new_image.rename(['band_1','band_2','band_3','band_4','band_5','band_6','band_7','band_8','band_9'])

def mask_57(img):
  # Cloud mask for Landsat 5 and 7
  mask = img.select(['cfmask']).neq(4).And(img.select(['cfmask']).neq(2)).And(img.select('B1').gt(ee.Image(0)))
  if aoi:
    return img.updateMask(mask).select(['B1','B2', 'B3','B4','B5','B7']).clip(AOI)
  else:
    return img.updateMask(mask).select(['B1','B2', 'B3','B4','B5','B7'])

def mask_8(img):
  # Cloud mask for Landsat 8
  mask = img.select(['cfmask']).neq(4).And(img.select(['cfmask']).neq(2)).And(img.select('B2').gt(ee.Image(0)))
  if aoi:
    return ee.Image(img.updateMask(mask).select(['B2', 'B3','B4','B5','B6','B7']).rename(['B1','B2','B3','B4','B5','B7'])).clip(AOI)
  else:
    return ee.Image(img.updateMask(mask).select(['B2', 'B3','B4','B5','B6','B7']).rename(['B1','B2','B3','B4','B5','B7']))

def get_inputs_training(_year, path, row):
  
  # Get inputs for training period
  year = _year - 1

  #Get 10 years worth of trianing data
  train_year_start = str(_year - 10) 
  train_start = train_year_start + '-01-01'
  train_end = str(year) + '-12-31'

  if pathrow:  
    train_collection7 = ee.ImageCollection('LANDSAT/LE7_SR'
      ).filterDate(train_start, train_end
      ).filter(ee.Filter.eq('WRS_PATH', path)
      ).filter(ee.Filter.eq('WRS_ROW', row))
  
    train_collection5 = ee.ImageCollection('LANDSAT/LT5_SR'
      # Filter to get only two years of data.
      ).filterDate(train_start, train_end
      ).filter(ee.Filter.eq('WRS_PATH', path)
      ).filter(ee.Filter.eq('WRS_ROW', row))
  else:
    train_collection7 = ee.ImageCollection('LANDSAT/LE7_SR'
      ).filterDate(train_start, train_end
      ).filterBounds(AOI)
 
    train_collection5 = ee.ImageCollection('LANDSAT/LT5_SR'
      # Filter to get only two years of data.
      ).filterDate(train_start, train_end
      ).filterBounds(AOI)
  # Mask clouds
                   
  train_col7_noclouds = train_collection7.map(mask_57).map(add_cloudscore7)
                   
  train_col5_noclouds = train_collection5.map(mask_57).map(add_cloudscore5)
  
  train_col_noclouds = train_col7_noclouds.merge(train_col5_noclouds)

  # Training collection unmixed
  
  train_col_unmix = train_col_noclouds.map(unmix)

  # Collection NFDI
  train_nfdi = ee.ImageCollection(train_col_unmix.map(get_nfdi)).sort('system:time_start')

  return train_nfdi

def get_inputs_monitoring(year, path, row):
  # Get inputs for monitoring period

  monitor_start = str(year + 1) + '-01-01'
  #TODO: paramterize this
  monitor_end = '2015' + '-12-31'
  
  if pathrow: 
    collection8 = ee.ImageCollection('LANDSAT/LC8_SR'
      # Filter to get only two years of data.
      ).filterDate(monitor_start, monitor_end
      ).filter(ee.Filter.eq('WRS_PATH', path)
      ).filter(ee.Filter.eq('WRS_ROW', row))
  
    collection7 = ee.ImageCollection('LANDSAT/LE7_SR'
      # Filter to get only two years of data.
      ).filterDate(monitor_start, monitor_end
      ).filter(ee.Filter.eq('WRS_PATH', path)
      ).filter(ee.Filter.eq('WRS_ROW', row))
  
    collection5 = ee.ImageCollection('LANDSAT/LT5_SR'
      # Filter to get only two years of data.
      ).filterDate(monitor_start, monitor_end
      ).filter(ee.Filter.eq('WRS_PATH', path)
      ).filter(ee.Filter.eq('WRS_ROW', row))
  else:
    collection8 = ee.ImageCollection('LANDSAT/LC8_SR'
      # Filter to get only two years of data.
      ).filterDate(monitor_start, monitor_end
      ).filterBounds(AOI)
  
    collection7 = ee.ImageCollection('LANDSAT/LE7_SR'
      # Filter to get only two years of data.
      ).filterDate(monitor_start, monitor_end
      ).filterBounds(AOI)
  
    collection5 = ee.ImageCollection('LANDSAT/LT5_SR'
      # Filter to get only two years of data.
      ).filterDate(monitor_start, monitor_end
      ).filterBounds(AOI)
  
  # Mask clouds
  col8_noclouds = collection8.map(mask_8).map(add_cloudscore8)

  col7_noclouds = collection7.map(mask_57).map(add_cloudscore7)
                   
  col5_noclouds = collection5.map(mask_57).map(add_cloudscore5)
  
  # merge
  col_l87noclouds = col8_noclouds.merge(col7_noclouds)
  col_noclouds = col_l87noclouds.merge(col5_noclouds)

  # testing: don't include l8
#  col_noclouds = col7_noclouds.merge(col5_noclouds)

  # Training collection unmixed
  col_unmix = col_noclouds.map(unmix)

  # Collection NFDI
  nfdi = ee.ImageCollection(col_unmix.map(get_nfdi)).sort('system:time_start')

  return nfdi

def get_inputs_retrain(year, path, row):
  # Get inputs for monitoring period

  monitor_start = str(year) + '-01-01'
  year_end = year + 1
  monitor_end = str(year_end) + '-12-31'
  
  if pathrow: 
    collection8 = ee.ImageCollection('LANDSAT/LC8_SR'
      # Filter to get only two years of data.
      ).filterDate(monitor_start, monitor_end
      ).filter(ee.Filter.eq('WRS_PATH', path)
      ).filter(ee.Filter.eq('WRS_ROW', row))
  
    collection7 = ee.ImageCollection('LANDSAT/LE7_SR'
      # Filter to get only two years of data.
      ).filterDate(monitor_start, monitor_end
      ).filter(ee.Filter.eq('WRS_PATH', path)
      ).filter(ee.Filter.eq('WRS_ROW', row))
  
    collection5 = ee.ImageCollection('LANDSAT/LT5_SR'
      # Filter to get only two years of data.
      ).filterDate(monitor_start, monitor_end
      ).filter(ee.Filter.eq('WRS_PATH', path)
      ).filter(ee.Filter.eq('WRS_ROW', row))
  else:
    collection8 = ee.ImageCollection('LANDSAT/LC8_SR'
      # Filter to get only two years of data.
      ).filterDate(monitor_start, monitor_end
      ).filterBounds(AOI)
  
    collection7 = ee.ImageCollection('LANDSAT/LE7_SR'
      # Filter to get only two years of data.
      ).filterDate(monitor_start, monitor_end
      ).filterBounds(AOI)
  
    collection5 = ee.ImageCollection('LANDSAT/LT5_SR'
      # Filter to get only two years of data.
      ).filterDate(monitor_start, monitor_end
      ).filterBounds(AOI)

  # Mask clouds
  col8_noclouds = collection8.map(mask_8).map(add_cloudscore8) 

  col7_noclouds = collection7.map(mask_57).map(add_cloudscore7)
                   
  col5_noclouds = collection5.map(mask_57).map(add_cloudscore5)
  
  # merge
  col_l87noclouds = col8_noclouds.merge(col7_noclouds)
  col_noclouds = col_l87noclouds.merge(col5_noclouds)

  # Testing:
#  col_noclouds = col7_noclouds.merge(col5_noclouds)

  # Training collection unmixed
  col_unmix = col_noclouds.map(unmix)

  # Collection NFDI
  nfdi = ee.ImageCollection(col_unmix.map(get_nfdi)).sort('system:time_start')

  return nfdi


def get_regression_coefs(train_array):
  # Get regression coefficients for the training period
  
  # Define the axes of iation in the collection array.
  imageAxis = 0
  bandAxis = 1

  # Check the length of the image axis (number of images).
  arrayLength = train_array.arrayLength(imageAxis)
  # Update the mask to ensure that the number of images is greater than or
  # equal to the number of predictors (the linear model is solveable).
  train_array = train_array.updateMask(arrayLength.gt(3))

  # Get slices of the array according to positions along the band axis.
  predictors = train_array.arraySlice(bandAxis, 0, 3)
  response = train_array.arraySlice(bandAxis, 3)
    
  # coefficients = predictors.matrixSolve(response)
  coefficients = predictors.matrixPseudoInverse().matrixMultiply(response)

  # Turn the results into a multi-band image.

 # global coefficientsImage (old, probably can delete)
  coefficientsImage = coefficients.arrayProject([0]).arrayFlatten([['coef_constant', 'coef_sin', 'coef_cos']])

  return coefficientsImage

def get_regression_coefs_trend(train_array):
  # Get regression coefficients for the training period
  
  # Define the axes of iation in the collection array.
  imageAxis = 0
  bandAxis = 1

  # Check the length of the image axis (number of images).
  arrayLength = train_array.arrayLength(imageAxis)
  # Update the mask to ensure that the number of images is greater than or
  # equal to the number of predictors (the linear model is solveable).
  train_array = train_array.updateMask(arrayLength.gt(4))

  # Get slices of the array according to positions along the band axis.
  predictors = train_array.arraySlice(bandAxis, 0, 4)
  response = train_array.arraySlice(bandAxis, 4)
    
  # coefficients = predictors.matrixSolve(response)
  coefficients = predictors.matrixPseudoInverse().matrixMultiply(response)

  # Turn the results into a multi-band image.
#  global coefficientsImage (old, probably can delete)
  coefficientsImage = coefficients.arrayProject([0]).arrayFlatten([['coef_constant','coef_trend', 'coef_sin', 'coef_cos']])

  return coefficientsImage

def get_coefficient_image(collection, band):
  # Turn regression array information into an image
  array = ee.ImageCollection(collection).map(makeVariables_fullem).select(['constant','sin','cos',band]).toArray()
  band_coefs_image = get_regression_coefs(array)

  return band_coefs_image

def deg_monitoring(year, ts_status, path, row, old_coefs, train_nfdi, first, tmean):
 # Main function for monitoring, should be looped over for each year

  # * REGRESSION

  # train array = nfdi collection as arrays
  train_array = ee.ImageCollection(train_nfdi).select('NFDI').map(makeVariables).toArray()
  train_array_trend = ee.ImageCollection(train_nfdi).select('NFDI').map(makeVariables_trend).toArray()

  # train_all = train array with temporal iables attached
  train_all = ee.ImageCollection(train_nfdi).select('NFDI').map(makeVariables)
 
  # Do it again with trend coefficient

  # coefficients image = image with regression coefficients (intercept, slope, sin, cos) for each pixel
  global coefficientsImage
  coefficientsImage = get_regression_coefs(train_array)
  coefficientsImage_trend = ee.Image(get_regression_coefs_trend(train_array_trend))
  
  # Do it again with GV and shade - for forest classification
  train_array_gv = ee.ImageCollection(train_nfdi).select('band_0').map(makeVariables_gv).toArray()
  coefficientsImage_gv = get_regression_coefs(train_array_gv)

  train_array_shade = ee.ImageCollection(train_nfdi).select('band_1').map(makeVariables_shade).toArray()
  coefficientsImage_shade = get_regression_coefs(train_array_shade)

  train_array_npv = ee.ImageCollection(train_nfdi).select('band_2').map(makeVariables_npv).toArray()
  coefficientsImage_npv = get_regression_coefs(train_array_npv)

  train_array_soil = ee.ImageCollection(train_nfdi).select('band_3').map(makeVariables_soil).toArray()
  coefficientsImage_soil = get_regression_coefs(train_array_soil)

  # check change status. If mid-change - use last year's coefficients. 

  #Get Tmean = mean NFDI residuals
  # If not in the middle of a change - use tmean for current training period
  # Else - use last year's
  train_coefs = ee.ImageCollection(train_all).map(addcoefs)

  # predict_nfdi_train = predicted NFDI based on regression coefficients for each image in training period
  predict_nfdi_train = ee.ImageCollection(train_coefs).map(predict_nfdi)

  # train_nfdi_mean = mean normalized residuals for the training period
  #Current year tmean = mean abs() residuals
  global train_nfdi_mean
  train_nfdi_sq_res = predict_nfdi_train.map(get_mean_residuals).mean()
  train_nfdi_mean = ee.Image(train_nfdi_sq_res).sqrt().rename(['mean_res'])

  # ** MONITORING PERIOD **

  # Monitor_NFDI = NFDI for monitoring period
  monitor_nfdi = get_inputs_monitoring(year, path, row)

  # monitor_collection = nfdi for monitoring period with temporal features
  monitor_collection = ee.ImageCollection(monitor_nfdi).map(makeVariables_fullem)

  # monitor_collection_coefs = monitor_collection with temporal coefficients attached
  monitor_collection_coefs = ee.ImageCollection(monitor_collection).map(addcoefs)

  # predict_nfdi_monitor = predicted NFDI based on regression coefficients
  predict_nfdi_monitor = ee.ImageCollection(monitor_collection_coefs).map(predict_nfdi)
  
  # predict_nfdi_monitor_mean = predict_nfdi_monitor with mean residual from training period attached
  predict_nfdi_monitor_mean = ee.ImageCollection(predict_nfdi_monitor).map(addmean)

  # monitor_nfdi_prs = subset of predict_nfdi_monitor_mean with just predicted nfdi, real nfdi, mean training residuals, and endmembers
  monitor_nfdi_prs = ee.ImageCollection(predict_nfdi_monitor_mean).select(['NFDI','Predict_NFDI','mean_res','band_0','band_1','band_2','band_3'])

  # monitor_abs_res = normalized residuals for predicted versus real nfdi
  monitor_abs_res = ee.ImageCollection(monitor_nfdi_prs).map(get_mean_residuals_norm)
    
  # results = results of change detection iteration
  results = ee.Image(monitor_nfdi_prs.iterate(monitor_func, ts_status))

  # combine monitoring nfdi with training
  new_training = ee.ImageCollection(train_nfdi).merge(monitor_nfdi).sort('system:time_start')

  return ee.List([results, coefficientsImage, new_training, train_nfdi_mean, coefficientsImage_trend, coefficientsImage_gv, coefficientsImage_shade, coefficientsImage_npv, coefficientsImage_soil])
#  return ee.List([results, coefficientsImage, new_training, train_nfdi_mean, coefficientsImage_trend])

# Retraining
def mask_nochange(image):
  # mask retrain stack if there has been no change
  ischanged = ee.Image(change_dates).gt(ee.Image(0))
  if aoi:
      return ee.Image(image).updateMask(ischanged).clip(AOI)
  else:
      return ee.Image(image).updateMask(ischanged)

def mask_beforechange(image):
  # mask retrain stack before change
  im_date = ee.Image(image).metadata('system:time_start').divide(ee.Image(315576e5))
  skip_year = change_dates.add(ee.Image(1))
  af_change = im_date.gt(skip_year)
  if aoi:
      return ee.Image(image).updateMask(af_change).clip(AOI)
  else:
      return ee.Image(image).updateMask(af_change)

def regression_retrain(original_collection, year, path, row):
  # get a few more years data
  y1_data = get_inputs_retrain(year, path, row)
  
  full_data_nomask = original_collection.merge(y1_data).sort('system:time_start')

  stack_nochange_masked = full_data_nomask.map(mask_nochange)
  
  stack_masked= stack_nochange_masked.map(mask_beforechange)
  
  #run regression on data after a change
  train_iables = ee.ImageCollection(stack_nochange_masked).map(makeVariables)
  
  train_array = train_iables.toArray()

  # coefficients image = image with regression coefficients (intercept, slope, sin, cos) for each pixel
  _coefficientsImage = get_regression_coefs(train_array)
  
  coefficientsImage = _coefficientsImage 
  
  retrain_with_coefs = ee.ImageCollection(train_iables).map(addcoefs)
  
  retrain_predict1 = ee.ImageCollection(retrain_with_coefs).map(predict_nfdi)
  
  retrain_predict2 = ee.ImageCollection(retrain_predict1).map(mask_nochange)
  
  retrain_predict = ee.ImageCollection(retrain_predict1).map(mask_beforechange)

  # do it again for fractions
  retrain_array_gv = ee.ImageCollection(stack_nochange_masked).select('band_0').map(makeVariables_gv).toArray()
  r_coefficientsImage_gv = ee.Image(get_regression_coefs(retrain_array_gv)).select(['coef_constant'])

  retrain_array_shade = ee.ImageCollection(stack_nochange_masked).select('band_1').map(makeVariables_shade).toArray()
  r_coefficientsImage_shade = ee.Image(get_regression_coefs(retrain_array_shade)).select(['coef_constant'])

  retrain_array_npv = ee.ImageCollection(stack_nochange_masked).select('band_2').map(makeVariables_npv).toArray()
  r_coefficientsImage_npv = ee.Image(get_regression_coefs(retrain_array_npv)).select(['coef_constant'])

  retrain_array_soil = ee.ImageCollection(stack_nochange_masked).select('band_3').map(makeVariables_soil).toArray()
  r_coefficientsImage_soil = ee.Image(get_regression_coefs(retrain_array_soil)).select(['coef_constant'])
  
  return ee.List([_coefficientsImage.rename(['Intercept', 'Sin','Cos']), retrain_predict, r_coefficientsImage_gv, r_coefficientsImage_shade, r_coefficientsImage_npv, r_coefficientsImage_soil])

def add_cloudscore5(image):
   # Add cloud score for Landsat 5
   thedate = image.date()
   date_bef = thedate.advance(-1, 'day')
   date_aft = thedate.advance(1,'day')
   toa = ee.ImageCollection('LANDSAT/LT05/C01/T1_TOA'
      ).filterDate(date_bef, date_aft
      ).filter(ee.Filter.eq('WRS_PATH', path)
      ).filter(ee.Filter.eq('WRS_ROW', row)
      ).first()
   cs = ee.Algorithms.If(
    ee.Image(toa),
    ee.Image(ee.Algorithms.Landsat.simpleCloudScore(ee.Image(toa))).select('cloud'),
    ee.Image(0).rename(['cloud']))

   mask = ee.Image(cs).lt(ee.Image(cloud_score))
   if aoi:
     return image.updateMask(mask).select(['B1','B2', 'B3','B4','B5','B7']).clip(AOI)
   else:
     return image.updateMask(mask).select(['B1','B2', 'B3','B4','B5','B7'])

def add_cloudscore7(image):
   # Add cloud score for Landsat 7
   thedate = image.date()
   date_bef = thedate.advance(-1, 'day')
   date_aft = thedate.advance(1,'day')
   toa = ee.ImageCollection('LANDSAT/LE07/C01/T1_TOA'
      ).filterDate(date_bef, date_aft
      ).filter(ee.Filter.eq('WRS_PATH', path)
      ).filter(ee.Filter.eq('WRS_ROW', row)
      ).first()
   cs = ee.Algorithms.If(
    ee.Image(toa),
    ee.Image(ee.Algorithms.Landsat.simpleCloudScore(ee.Image(toa))).select('cloud'),
    ee.Image(0).rename(['cloud']))

   mask = ee.Image(cs).lt(ee.Image(cloud_score))
   if aoi:
     return image.updateMask(mask).select(['B1','B2', 'B3','B4','B5','B7']).clip(AOI)
   else:
     return image.updateMask(mask).select(['B1','B2', 'B3','B4','B5','B7'])

def add_cloudscore8(image):
   # Add cloud score for Landsat 8
   thedate = image.date()
   date_bef = thedate.advance(-1, 'day')
   date_aft = thedate.advance(1,'day')
   toa = ee.ImageCollection('LANDSAT/LC08/C01/T1_TOA'
      ).filterDate(date_bef, date_aft
      ).filter(ee.Filter.eq('WRS_PATH', path)
      ).filter(ee.Filter.eq('WRS_ROW', row)
      ).first()
   cs = ee.Algorithms.If(
    ee.Image(toa),
    ee.Image(ee.Algorithms.Landsat.simpleCloudScore(ee.Image(toa))).select('cloud'),
    ee.Image(0).rename(['cloud']))
     
   mask = ee.Image(cs).lt(ee.Image(cloud_score))
   if aoi:
     return image.updateMask(mask).select(['B1','B2', 'B3','B4','B5','B7']).clip(AOI)
   else:
     return image.updateMask(mask).select(['B1','B2', 'B3','B4','B5','B7'])

# ** MAIN WORK **

# ** DEFINE GLOBALS

coefficientsImage = ""
train_nfdi_mean = ""
change_dates = ""

# ts_status = initial image for change detection iteration. 
# Bands (Depricated: TODO):
    # 1. Change (1) or no change (0). Used as mask. Default: 0
    # 2. Consecutive observations passed threshold. Default: 0
    # 3. Date of change if 1 = 1. Default: 0
    # 4. Magnitude of change
    # 5. iterator 
    
ts_status = ee.Image(1).addBands([ee.Image(0),ee.Image(0),ee.Image(0),ee.Image(1),ee.Image(0),ee.Image(0),ee.Image(0),ee.Image(0)]).rename(['band_1','band_2','band_3','band_4','band_5','band_6','band_7','band_8','band_9']).unmask()

# Do the monitoring for each year.

old_coefs = ee.Image(0)

# First year inputs

train_nfdi = get_inputs_training(year, path, row)

results = deg_monitoring(2000, ts_status, path, row, old_coefs, train_nfdi, True, None)

tmean = results.get(3)
before_rmse = ee.Image(tmean) #RMSE for training period

final_results = ee.Image(results.get(0)) 
old_coefs = results.get(1) #Coefficients without trend
final_train = ee.ImageCollection(results.get(2)) #TODO

# I think I can delete this TODO:
change_output = final_results.select('band_1').eq(ee.Image(0))

# Regression information for classification
coefficientsImage_trend = ee.Image(results.get(4))
coefficientsImage_gv = ee.Image(results.get(5)).select(['coef_constant'])
coefficientsImage_shade = ee.Image(results.get(6)).select(['coef_constant'])
coefficientsImage_npv = ee.Image(results.get(7)).select(['coef_constant'])
coefficientsImage_soil = ee.Image(results.get(8)).select(['coef_constant'])

global change_dates
change_dates = final_results.select('band_3')

# Retrain
retrain_regression = regression_retrain(final_train, 2016, path, row)

retrain_coefs = ee.Image(retrain_regression.get(0))
retrain_predict = ee.ImageCollection(retrain_regression.get(1))
retrain_gv = ee.Image(retrain_regression.get(2))
retrain_shade = ee.Image(retrain_regression.get(3))
retrain_npv = ee.Image(retrain_regression.get(4))
retrain_soil = ee.Image(retrain_regression.get(5))
retrain_predict_last = ee.Image(retrain_predict.toList(1000).get(-1))

# Get predicted NFDI at middle of time series
retrain_last = ee.Image(retrain_predict.toList(1000).get(-1))

if aoi:
  retrain_last_date = ee.Image(retrain_last).metadata('system:time_start').divide(ee.Image(31557600000)).clip(AOI)
else:
  retrain_last_date = ee.Image(retrain_last).metadata('system:time_start').divide(ee.Image(31557600000))

# get the date at the middle of the retrain time series
retrain_middle = ee.Image(ee.Image(retrain_last_date).subtract(ee.Image(change_dates)).divide(ee.Image(2)).add(ee.Image(change_dates))).rename(['years'])
predict_middle = pred_middle_retrain(retrain_middle, retrain_coefs)


# Get prediction for year 2000
original_middle = ee.Image(30).rename(['years'])
predict_middle_original = pred_middle_retrain(original_middle, ee.Image(old_coefs).rename(['Intercept','Sin', 'Cos']))
predict_middle_original = ee.Image(predict_middle_original)

# Prepare output

# Normalize magnitude
# st_magnitude = short-term change magnitude
st_magnitude = final_results.select('band_4').divide(ee.Image(consec)).multiply(change_dates.gt(ee.Image(0)))

# Endmembers 
change_endm1 = final_results.select('band_6').divide(ee.Image(consec)).multiply(change_dates.gt(ee.Image(0)))
change_endm2 = final_results.select('band_7').divide(ee.Image(consec)).multiply(change_dates.gt(ee.Image(0)))
change_endm3 = final_results.select('band_8').divide(ee.Image(consec)).multiply(change_dates.gt(ee.Image(0)))
change_endm4 = final_results.select('band_9').divide(ee.Image(consec)).multiply(change_dates.gt(ee.Image(0)))

#Forest mask
# Right now I am not doing this, classification instead
#forest_mask = predict_middle_original.gt(ee.Image(mag_thresh)).And(otmean.lt(ee.Image(rmse_thresh))).multiply(forest2000.gt(ee.Image(forest_threshold)))

# save_output:
#bands 1-6: Change information
save_output = change_dates.rename(['change_date']).addBands( #change date
  [ee.Image(st_magnitude).rename(['magnitude']) #Change magnitude
  , ee.Image(change_endm1).rename(['change_v_gv']) #Change vector GV 
  , ee.Image(change_endm2).rename(['change_v_npv']) #Change vector NPV
  , ee.Image(change_endm3).rename(['change_v_soil']) #Change vector Soil
  , ee.Image(change_endm4).rename(['change_v_shade']) #Change ctor Shade

#bands 7-13: Pre-disturbance training information
  , ee.Image(old_coefs).select('coef_constant').rename(['mag_training']) #mag training
  , ee.Image(old_coefs).select('coef_sin').rename(['sine_training']) #sine training
  , ee.Image(old_coefs).select('coef_cos').rename(['cos_training']) #cos training
  , ee.Image(coefficientsImage_gv).rename(['pre_gv']) # Training GV
  , ee.Image(coefficientsImage_shade).rename(['pre_shade']) #Training Shade
  , ee.Image(coefficientsImage_npv).rename(['pre_npv']) #Training NPV
  , ee.Image(coefficientsImage_soil).rename(['pre_soil']) #Training SOIL

#Bands 14-20: Post-disturbance retrain information
  , ee.Image(predict_middle).rename(['nfdi_retrain']) #Mag after
  , ee.Image(retrain_coefs).select('Sin').rename(['retrain_sin']) #Sine after
  , ee.Image(retrain_coefs).select('Cos').rename(['retrain_cos']) #Cos after
  , ee.Image(retrain_gv).rename(['after_gv']) #After GV
  , ee.Image(retrain_shade).rename(['after_shade']) #After Shade
  , ee.Image(retrain_npv).rename(['after_npv']) #After NPV
  , ee.Image(retrain_soil).rename(['after_soil'])]).toFloat() #After Soil

print('Submitting task')

task_config = {
  'description': output,
  'scale': 30
  }

project_root='users/bullocke'
folder_name='sample4'

# Save to Engine asset or Drive?
if asset:
  task = ee.batch.Export.image.toAsset(
        image=save_output, 
        description=output,
        assetId='{}/{}/{}'.format(project_root,folder_name,output), 
        scale=30)
else:
  task = ee.batch.Export.image(save_output, output, task_config)
task.start()


