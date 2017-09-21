#!/usr/bin/env python
# -*- coding: UTF-8 -*-
""" Continuous Degradation Detection Using Google Earth Engine


Usage: GE_cdd.py [options] <output>

  --path=PATH       path
  --row=ROW         row
  --consec=CONSEC   consecutive obs to trigger change (default: 5)
  --mag=MAG         size of long-term magnitude window (default: 10)
  --thresh=THRESH   change threshold (default: 3.5)
  --forest=FOREST   forest % cover threshold (default: 30)
  --aoi             Use an area of interest (must hard code)
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

if args['--path']:
    path = int(args['--path'])
else:
    print('need to supply path')
    sys.exit()

if args['--row']:
    row = int(args['--row'])
else:
    print('need to supply row')
    sys.exit()

if args['--consec']:
    consec = int(args['--consec'])
else:
    consec = 5

if args['--mag']:
    mag_window = int(args['--mag'])
else:
    mag_window = 10

if args['--thresh']:
    thresh = float(args['--thresh'])
else:
    thresh = 3.5

if args['--forest']:
    forest_threshold = int(args['--forest'])
else:
    forest_threshold = 30

aoi = False
if args['--aoi']:
    aoi = True

output=str(args['<output>'])
print(output)

#GLOBALS
global AOI

AOI = ee.Geometry.Polygon(
        [[[-63.18941116333008, -9.415532407445827],
          [-63.18941116333008, -9.427555957905986],
          [-63.16263198852539, -9.426878585912847],
          [-63.1629753112793, -9.414008265523616]]])

# spectral endmembers from Souza (2005).
gv= [500, 900, 400, 6100, 3000, 1000]
shade= [0, 0, 0, 0, 0, 0]
npv= [1400, 1500, 1300, 3000, 7800, 2000]
soil= [2000, 3000, 3400, 5800, 7900, 7000]

# Hansen forest cover
if aoi:
    forest2000 = ee.Image('UMD/hansen/global_forest_change_2015_v1_3').select('treecover2000').clip(AOI)
else:
    forest2000 = ee.Image('UMD/hansen/global_forest_change_2015_v1_3').select('treecover2000')

# ** FUNCTIONS **

# Collection map functions

def unmix(image):
  unmixi = ee.Image(image).unmix([gv, shade, npv, soil], True, True)
  newimage = ee.Image(image).addBands(unmixi)
  return newimage

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

# Add standard deviation band
def addmean(image):
  newimage = ee.Image(image).addBands(ee.Image(train_nfdi_mean))
  return newimage

def get_mean_residuals(image):
  res = image.select('NFDI').subtract(image.select('Predict_NFDI')).abs()
  return res

def get_mean_residuals_norm(image):
  res = image.select('NFDI').subtract(image.select('Predict_NFDI')).divide(ee.Image(train_nfdi_mean)).rename(['res'])
  return image.addBands(res)

def add_changedate_mask(image):
  image_date = image.metadata('system:time_start').divide(ee.Image(8.64e7))
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
    # 6. Consecutive observations passed threshold w/o resetting
    # 7. long-term change magnitude
    
# image = image in collection 
def monitor_func(image, new_image):
  # Apply mask
  #define working mask. 1 if no change has been detected and no coud
  zero_mask = ee.Image(0)

  #MASKED WITH ZERO MASK
  zero_mask_nc = ee.Image(new_image).select('band_1').eq(ee.Image(1)).multiply( # band 1 != 0
                 ee.Image(new_image).select('band_2').lt(ee.Image(consec))).unmask() # not passed consec thresh
               
  cloud_mask = ee.Image(image).select('NFDI').mask().neq(ee.Image(0))
               
  mag_mask_nc = ee.Image(new_image).select('band_1').eq(ee.Image(0)).multiply(
                ee.Image(new_image).select('band_8').eq(ee.Image(1))).unmask()

  image_monitor = image.unmask()

  # Get normalized residual
  norm_dif = image_monitor.select('NFDI').subtract(image_monitor.select('Predict_NFDI'))
  norm_res = ee.Image(norm_dif).divide(ee.Image(train_nfdi_mean))

  # passed_thresh = Find if it is passed threshold (1 = passed threshold)
  # 1 if norm_res > threshold
  # 0 if norm res < threshold, cloud mask
  gt_thresh = ee.Image(norm_res).abs().gt(ee.Image(thresh)).multiply(zero_mask_nc).multiply(cloud_mask)
  #gt_thresh_mag = ee.Image(norm_res).abs().gt(ee.Image(0)).multiply(mag_mask_nc)
  gt_thresh_mag = mag_mask_nc.multiply(cloud_mask)
  
  # Band 2: Consecutive observations beyond threshold
  # = # consecutive observations > threshold
  _band_2 = ee.Image(new_image).select('band_2').add(gt_thresh).multiply(ee.Image(new_image).select('band_1'))
  band_2_increase = _band_2.gt(ee.Image(new_image).select('band_2')).Or(cloud_mask.eq(ee.Image(0))) #This was added to not reset w/cloud
  band_2 = _band_2.multiply(band_2_increase)
  band_6 = ee.Image(new_image).select('band_6').add(gt_thresh_mag).multiply(ee.Image(new_image).select('band_8'))

  # flag_change = where band 2 of status is at consecutive threshold (5)
  # 1 if consec is reached
  # 0 if not consec
  flag_change = band_2.eq(ee.Image(consec))
  flag_mag = band_6.eq(ee.Image(mag_window)) # 1 if reaches threshold
  
  # Change status
  # 1 = no change
  # 0 = change
  band_1 = ee.Image(new_image).select('band_1').eq(ee.Image(1)).And(flag_change.eq(ee.Image(0)))
  band_8 = ee.Image(new_image).select('band_8').eq(ee.Image(1)).And(flag_mag.eq(ee.Image(0)))

  # add date of image to band 3 of status where change has been flagged
  # time is stored in milliseconds since 01-01-1970, so scale to be days since then 
  change_date = image.metadata('system:time_start').multiply(flag_change).divide(ee.Image(8.64e7))
  band_3 = ee.Image(new_image).select('band_3').add(change_date)

  # Magnitude
  magnitude = norm_res.abs().multiply(gt_thresh).multiply(zero_mask_nc)
  magnitude_long = norm_res.abs().multiply(band_8).multiply(mag_mask_nc)

  #Keep mag if already changed or in process
 # is_changing = ee.Image(new_image).select('band_1').eq(ee.Image(0)).Or(ee.Image(new_image).select('band_2').gt(ee.Image(0)))
  is_changing = band_1.eq(ee.Image(0)).Or(band_2).gt(ee.Image(0));

 # mag_changing = ee.Image(new_image).select('band_8').eq(ee.Image(0)).Or(ee.Image(new_image).select('band_6').gt(ee.Image(0)))
  mag_changing = band_8.eq(ee.Image(0)).Or(band_6).gt(ee.Image(0));

  band_4 = ee.Image(new_image).select('band_4').add(magnitude).multiply(is_changing)
  band_7 = ee.Image(new_image).select('band_7').add(magnitude_long).multiply(mag_changing)

  # Add one to iteration band
  band_5 = ee.Image(new_image).select('band_5').add(ee.Image(1))
  
  new_image = band_1.addBands([band_2,band_3,band_4,band_5, band_6, band_7, band_8])
  
  return new_image.rename(['band_1','band_2','band_3','band_4','band_5', 'band_6', 'band_7','band_8'])

def mask_57(img):
  mask = img.select(['cfmask']).neq(4).And(img.select(['cfmask']).neq(2))
  if aoi:
    return img.updateMask(mask).select(['B1','B2', 'B3','B4','B5','B7']).clip(AOI)
  else:
    return img.updateMask(mask).select(['B1','B2', 'B3','B4','B5','B7'])

def mask_8(img):
  mask = img.select(['cfmask']).neq(4).And(img.select(['cfmask']).neq(2))
  if aoi:
    return ee.Image(img.updateMask(mask).select(['B2', 'B3','B4','B5','B6','B7']).rename(['B1','B2','B3','B4','B5','B7'])).clip(AOI)
  else:
    return ee.Image(img.updateMask(mask).select(['B2', 'B3','B4','B5','B6','B7']).rename(['B1','B2','B3','B4','B5','B7']))

def get_inputs_training(_year, path, row):
  
  # Get inputs for training period
  year = _year - 1
  train_year_start = str(_year - 6) 
  train_start = train_year_start + '-01-01'
  train_end = str(year) + '-12-31'
  
  train_collection7 = ee.ImageCollection('LANDSAT/LE7_SR'
    ).filterDate(train_start, train_end
    ).filter(ee.Filter.eq('WRS_PATH', path)
    ).filter(ee.Filter.eq('WRS_ROW', row))
  
  train_collection5 = ee.ImageCollection('LANDSAT/LT5_SR'
    # Filter to get only two years of data.
    ).filterDate(train_start, train_end
    ).filter(ee.Filter.eq('WRS_PATH', path)
    ).filter(ee.Filter.eq('WRS_ROW', row))
  
  # Mask clouds
                   
  train_col7_noclouds = train_collection7.map(mask_57)
                   
  train_col5_noclouds = train_collection5.map(mask_57)
  
  train_col_noclouds = train_col7_noclouds.merge(train_col5_noclouds)

  # Training collection unmixed
  
  train_col_unmix = train_col_noclouds.map(unmix)

  # Collection NFDI
  train_nfdi = ee.ImageCollection(train_col_unmix.map(get_nfdi)).sort('system:time_start')

  return train_nfdi

def get_inputs_monitoring(year, path, row):

  # Get inputs for monitoring period

  monitor_start = str(year) + '-01-01'
  monitor_end = str(year) + '-12-31'
  
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
  
  
  # Mask clouds
  col8_noclouds = collection8.map(mask_8) 

  col7_noclouds = collection7.map(mask_57)
                   
  col5_noclouds = collection5.map(mask_57)
  
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
  
  # Define the axes of variation in the collection array.
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
  coefficientsImage = coefficients.arrayProject([0]).arrayFlatten([['coef_constant', 'coef_trend', 'coef_sin', 'coef_cos']])

  return coefficientsImage

def deg_monitoring(year, ts_status, path, row, old_coefs, train_nfdi, first, tmean):
 # Main function for monitoring, should be looped over for each year

  # * REGRESSION

  # train array = nfdi collection as arrays
  train_array = ee.ImageCollection(train_nfdi).map(makeVariables).toArray()

  # train_all = train array with temporal variables attached
  train_all = ee.ImageCollection(train_nfdi).map(makeVariables)

  # coefficients image = image with regression coefficients (intercept, slope, sin, cos) for each pixel
  _coefficientsImage = get_regression_coefs(train_array)
  
  # check change status. If mid-change - use last year's coefficients. 
  is_changing = ee.Image(ts_status).select("band_2").gt(ee.Image(0)).Or(ee.Image(ts_status).select('band_1').eq(ee.Image(0)))

  not_changing = ee.Image(is_changing).eq(ee.Image(0))
  
  old_changing_coefs = ee.Image(is_changing).multiply(ee.Image(old_coefs))
  current_coefs_nochange = ee.Image(not_changing).multiply(ee.Image(_coefficientsImage))

  global coefficientsImage
  coefficientsImage = old_changing_coefs.add(current_coefs_nochange)

  train_intercept = coefficientsImage.select('coef_constant')

  #Get Tmean = mean NFDI residuals
  # If not in the middle of a change - use tmean for current training period
  # Else - use last year's
  if first:
    train_coefs = ee.ImageCollection(train_all).map(addcoefs)

    # predict_nfdi_train = predicted NFDI based on regression coefficients for each image in training period
    predict_nfdi_train = ee.ImageCollection(train_coefs).map(predict_nfdi)

    # train_nfdi_mean = mean normalized residuals for the training period
    #Current year tmean = mean abs() residuals
    global train_nfdi_mean
    train_nfdi_mean = predict_nfdi_train.map(get_mean_residuals).mean().rename(['mean_res'])
    _train_nfdi_mean = train_nfdi_mean

  else:
    #Check if it is in the middle of a change - if so use last year's tmean

    is_changing_mag = ee.Image(ts_status).select("band_2").gt(ee.Image(0)).Or(ee.Image(ts_status).select('band_1').eq(ee.Image(0))).Or(ee.Image(ts_status).select('band_6').gt(ee.Image(0)))

    not_changing_mag = ee.Image(is_changing_mag).eq(ee.Image(0))

    train_coefs = ee.ImageCollection(train_all).map(addcoefs)

    # predict_nfdi_train = predicted NFDI based on regression coefficients for each image in training period
    predict_nfdi_train = ee.ImageCollection(train_coefs).map(predict_nfdi)

    # train_nfdi_mean = mean normalized residuals for the training period
    #Current year tmean = mean abs() residuals
    _train_nfdi_mean = predict_nfdi_train.map(get_mean_residuals).mean().rename(['mean_res'])

    old_changing_tmean = ee.Image(is_changing_mag).multiply(ee.Image(tmean))
    current_tmean_nochange = ee.Image(not_changing_mag).multiply(ee.Image(_train_nfdi_mean))

    global train_nfdi_mean
    train_nfdi_mean = old_changing_tmean.add(current_tmean_nochange).rename(['mean_res'])


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

  return ee.List([results, coefficientsImage, new_training, _train_nfdi_mean])


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
    # 6. Consecutive observations passed threshold w/o resetting
    # 7. long-term change magnitude
    
ts_status = ee.Image(1).addBands([ee.Image(0),ee.Image(0),ee.Image(0),ee.Image(1), ee.Image(0), ee.Image(0), ee.Image(1)]).rename(['band_1','band_2','band_3','band_4','band_5','band_6','band_7','band_8']).unmask()

# Do the monitoring for each year.

old_coefs = ee.Image(0)

# First year inputs

train_nfdi = get_inputs_training(2003, path, row)

results = deg_monitoring(2003, ts_status, path, row, old_coefs, train_nfdi, True, None)

ts_status = results.get(0)
old_coefs = results.get(1)
train_nfdi = results.get(2)
tmean = results.get(3)

results = deg_monitoring(2004, ts_status, path, row, old_coefs, train_nfdi, False, tmean)

ts_status = results.get(0)
old_coefs = results.get(1)
train_nfdi = results.get(2)
tmean = results.get(3)

results = deg_monitoring(2005, ts_status, path, row, old_coefs, train_nfdi, False, tmean)

ts_status = results.get(0)
old_coefs = results.get(1)
train_nfdi = results.get(2)
tmean = results.get(3)

results = deg_monitoring(2006, ts_status, path, row, old_coefs, train_nfdi, False, tmean)

ts_status = results.get(0)
old_coefs = results.get(1)
train_nfdi = results.get(2)
tmean = results.get(3)

results = deg_monitoring(2007, ts_status, path, row, old_coefs, train_nfdi, False, tmean)

ts_status = results.get(0)
old_coefs = results.get(1)
train_nfdi = results.get(2)
tmean = results.get(3)

results = deg_monitoring(2008, ts_status, path, row, old_coefs, train_nfdi, False, tmean)

ts_status = results.get(0)
old_coefs = results.get(1)
train_nfdi = results.get(2)
tmean = results.get(3)

results = deg_monitoring(2009, ts_status, path, row, old_coefs, train_nfdi, False, tmean)

ts_status = results.get(0)
old_coefs = results.get(1)
train_nfdi = results.get(2)
tmean = results.get(3)

results = deg_monitoring(2010, ts_status, path, row, old_coefs, train_nfdi, False, tmean)

final_results = ee.Image(results.get(0))

change_output = final_results.select('band_1').eq(ee.Image(0))

global change_dates
change_dates = final_results.select('band_3')

# Normalize magnitude
# st_magnitude = short-term change magnitude
st_magnitude = final_results.select('band_4').divide(ee.Image(consec)).multiply(change_dates.gt(ee.Image(0)))

# lt_magnitude = long-term change magnitude
done_lt = final_results.select('band_6').eq(ee.Image(0)).And(final_results.select('band_8').eq(ee.Image(0)))
not_done_lt = final_results.select('band_1').eq(ee.Image(0)).And(final_results.select('band_6').neq(ee.Image(0)))
lt_normalizer = ee.Image(mag_window).multiply(done_lt).add(not_done_lt.multiply(final_results.select("band_6")))
lt_magnitude = final_results.select('band_7').divide(lt_normalizer).multiply(change_dates.gt(ee.Image(0)))

# save_output:
# Bands:
    # 1. Change date
    # 2. Short-term change magnitude
    # 3. Long-term change magnitude
# Mask:
    # Hansen 2000 forest mask according to % canopy cover threshold (forest_threshold)

save_output = change_dates.addBands([st_magnitude, lt_magnitude]).multiply(forest2000.gt(ee.Image(forest_threshold)))
print('Submitting task')
task_config = {
  'description': output,
  'scale': 30
  }
task = ee.batch.Export.image(save_output, output, task_config)
task.start()



