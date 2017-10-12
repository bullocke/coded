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

"""

from docopt import docopt
import os,sys
import numpy as np
import datetime
import pandas as pd
import matplotlib.dates as mdates
from matplotlib import dates
from pylab import *

#Import pycc and earth engine
import ee
import ccd

# Initialize Earth Engine
ee.Initialize()
print('Earth Engine Initialized')

# Parse arguments
if __name__ == '__main__':
    args = docopt(__doc__, version='0.6.2')

path = None
row = None
pathrow = False
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

output=str(args['<output>'])
print(output)

#GLOBALS
global AOI


#Missed change 299 69
AOI = ee.Geometry.Polygon(
        [[[-60.01581072807312, -12.736319129682075],
          [-60.0158429145813, -12.74218984418972],
          [-60.00924468040466, -12.742252631845096],
          [-60.00914812088013, -12.736224945988106]]])

#Try 2
gv= [500, 900, 400, 6100, 3000, 1000]
npv= [1400, 1700, 2200, 3000, 5500, 3000]
soil= [2000, 3000, 3400, 5800, 6000, 5800]
shade= [0, 0, 0, 0, 0, 0]
#cloud = [6000, 5500, 5000, 4500, 4200, 4000]
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
  return ee.Image(image).addBands(ee.Image(newimage).rename(['NFDI'])).select(['band_0','NFDI'])

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
  newimage = ee.Image(image).addBands(ee.Image(coefficientsImage))
  return newimage

# Prediction function
def predict_nfdi(image):
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
  newimage = ee.Image(image).addBands(ee.Image(train_nfdi_mean))
  return newimage

#def get_mean_residuals(image):
#  res = image.select('NFDI').subtract(image.select('Predict_NFDI')).abs()
#  return res

def get_mean_residuals(image):
  res = ee.Image(image).select('NFDI').subtract(image.select('Predict_NFDI'))
  
  sq_res = ee.Image(res).multiply(ee.Image(res)).rename(['sq_res'])

  full_image = ee.Image(image).addBands(ee.Image(sq_res))
  return full_image.select('sq_res')

def get_mean_residuals_norm(image):
  res = image.select('NFDI').subtract(image.select('Predict_NFDI')).divide(ee.Image(train_nfdi_mean)).rename(['res'])
  return image.addBands(res)

def add_changedate_mask(image):
#  image_date = image.metadata('system:time_start').divide(ee.Image(8.64e7))
  image_date = image.metadata('system:time_start').divide(ee.Image(315576e5))
  eq_change = image_date.eq(ee.Image(change_dates))
  new_band = ee.Image(1).multiply(image.select('NFDI')).updateMask(eq_change).rename('Change')
  return image.addBands(new_band)

def get_abs(image):
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
  change_date = image.metadata('system:time_start').multiply(flag_change).divide(ee.Image(315576e5))
  band_3 = ee.Image(new_image).select('band_3').add(change_date)

  # Magnitude
#  magnitude = ee.Image(norm_dif).abs().multiply(gt_thresh).multiply(zero_mask_nc)
  magnitude = ee.Image(norm_dif).multiply(gt_thresh).multiply(zero_mask_nc)

  #Keep mag if already changed or in process
  is_changing = band_1.eq(ee.Image(0)).Or(band_2).gt(ee.Image(0))


  band_4 = ee.Image(new_image).select('band_4').add(magnitude).multiply(is_changing)

  # Add one to iteration band
  band_5 = ee.Image(new_image).select('band_5').add(ee.Image(1))
  
  new_image = band_1.addBands([band_2,band_3,band_4,band_5])
  
  return new_image.rename(['band_1','band_2','band_3','band_4','band_5'])

def mask_57(img):
  mask = img.select(['cfmask']).neq(4).And(img.select(['cfmask']).neq(2)).And(img.select('B1').gt(ee.Image(0)))
  if aoi:
    return img.updateMask(mask).select(['B1','B2', 'B3','B4','B5','B7']).clip(AOI)
  else:
    return img.updateMask(mask).select(['B1','B2', 'B3','B4','B5','B7'])

def mask_8(img):
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
  #monitor_end = str(year + 1) + '-12-31'
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
  global coefficientsImage
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
  global coefficientsImage
  coefficientsImage = coefficients.arrayProject([0]).arrayFlatten([['coef_constant','coef_trend', 'coef_sin', 'coef_cos']])

  return coefficientsImage

def deg_monitoring(year, ts_status, path, row, old_coefs, train_nfdi, first, tmean):
 # Main function for monitoring, should be looped over for each year

  # * REGRESSION

  # train array = nfdi collection as arrays
  train_array = ee.ImageCollection(train_nfdi).map(makeVariables).toArray()
  train_array_trend = ee.ImageCollection(train_nfdi).map(makeVariables_trend).toArray()

  # train_all = train array with temporal iables attached
  train_all = ee.ImageCollection(train_nfdi).map(makeVariables)
 
  # Do it again with trend coefficient

  # coefficients image = image with regression coefficients (intercept, slope, sin, cos) for each pixel
  global coefficientsImage
  coefficientsImage = get_regression_coefs(train_array)
  coefficientsImage_trend = ee.Image(get_regression_coefs_trend(train_array_trend)).select('coef_trend')
  
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
  monitor_collection = ee.ImageCollection(monitor_nfdi).map(makeVariables)

  # monitor_collection_coefs = monitor_collection with temporal coefficients attached
  monitor_collection_coefs = ee.ImageCollection(monitor_collection).map(addcoefs)

  # predict_nfdi_monitor = predicted NFDI based on regression coefficients
  predict_nfdi_monitor = ee.ImageCollection(monitor_collection_coefs).map(predict_nfdi)
  
  # predict_nfdi_monitor_mean = predict_nfdi_monitor with mean residual from training period attached
  predict_nfdi_monitor_mean = ee.ImageCollection(predict_nfdi_monitor).map(addmean)

  # monitor_nfdi_prs = subset of predict_nfdi_monitor_mean with just predicted nfdi, real nfdi, and mean training residuals
  monitor_nfdi_prs = ee.ImageCollection(predict_nfdi_monitor_mean).select(['NFDI','Predict_NFDI','mean_res'])

  # monitor_abs_res = normalized residuals for predicted versus real nfdi
  monitor_abs_res = ee.ImageCollection(monitor_nfdi_prs).map(get_mean_residuals_norm)
    
  # results = results of change detection iteration
  results = ee.Image(monitor_nfdi_prs.iterate(monitor_func, ts_status))

  # combine monitoring nfdi with training
  new_training = ee.ImageCollection(train_nfdi).merge(monitor_nfdi).sort('system:time_start')

  return ee.List([results, coefficientsImage_trend, new_training, train_nfdi_mean, coefficientsImage_trend])


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
  
  return ee.List([_coefficientsImage.rename(['Intercept', 'Sin','Cos']), retrain_predict])


def add_cloudscore5(image):
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
# Bands:
    # 1. Change (1) or no change (0). Used as mask. Default: 0
    # 2. Consecutive observations passed threshold. Default: 0
    # 3. Date of change if 1 = 1. Default: 0
    # 4. Magnitude of change
    # 5. iterator 
    
ts_status = ee.Image(1).addBands([ee.Image(0),ee.Image(0),ee.Image(0),ee.Image(1)]).rename(['band_1','band_2','band_3','band_4','band_5']).unmask()

# Do the monitoring for each year.

old_coefs = ee.Image(0)

# First year inputs

train_nfdi = get_inputs_training(year, path, row)

results = deg_monitoring(2000, ts_status, path, row, old_coefs, train_nfdi, True, None)

tmean = results.get(3)
otmean = ee.Image(tmean)
final_results = ee.Image(results.get(0))
old_coefs = results.get(1)
final_train = ee.ImageCollection(results.get(2))
change_output = final_results.select('band_1').eq(ee.Image(0))
coefficientsImage_trend = ee.Image(results.get(4))

global change_dates
change_dates = final_results.select('band_3')


# Retrain

retrain_regression = regression_retrain(final_train, 2016, path, row)

retrain_coefs = ee.Image(retrain_regression.get(0))
retrain_predict = ee.ImageCollection(retrain_regression.get(1))
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
predict_middle_original = pred_middle_retrain(original_middle, ee.Image(old_coefs).rename(['Intercept']))
predict_middle_original = ee.Image(predict_middle_original)


# Prepare output

# Normalize magnitude
# st_magnitude = short-term change magnitude
st_magnitude = final_results.select('band_4').divide(ee.Image(consec)).multiply(change_dates.gt(ee.Image(0)))

#Forest mask
forest_mask = predict_middle_original.gt(ee.Image(mag_thresh)).And(otmean.lt(ee.Image(rmse_thresh))).multiply(forest2000.gt(ee.Image(forest_threshold)))


# save_output:
# Bands:
    # 1. Change date
    # 2. Short-term change magnitude
    # 3. Predicted NFDI: End of retrain period
    # 4. Regression constant (intercept)
# Mask:
    # Hansen 2000 forest mask according to % canopy cover threshold (forest_threshold)


save_output = change_dates.rename(['change_date']).addBands(
  [ee.Image(st_magnitude).rename(['magnitude'])
  , ee.Image(predict_middle).rename(['nfdi_retrain'])
  , ee.Image(predict_middle_original).rename(['nfdi_2000'])
  , coefficientsImage_trend.rename(['slope_training'])]).updateMask(forest_mask).toFloat()

print('Submitting task')

task_config = {
  'description': output,
  'scale': 30
  }
task = ee.batch.Export.image(save_output, output, task_config)
task.start()



