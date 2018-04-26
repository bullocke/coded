#!/usr/bin/env python
# -*- coding: UTF-8 -*-

""" Simple version of the Google Earth Engine Continuous Degradation i
(CDD) algorithm for faster processing. CDD is built upon the Google 
Earth Engine's Python API and storage of the Landsat data archive. CDD 
utilizes a similar change detection approach to the Continuous Change 
Detection and Classification (CCDC) algorithm (Zhu & Woodcock, 2014). 
However, CDD is modified to specifically monitor for degradation and in
 doing so, deforestation. CDD is also simplified to be applicable on the 
Earth Engine. The change detection inputs are spectral endmembers 
computed for every image in the study region. The endmembers are based
on fraction green vegetation (GV), non-photosynthetic vegetation (NPV), 
soil, and shade. Additionally, an index on endmembers as defined in Souza
 et al. (2005) called the Normalized Degradation Fraction Index is 
utilized. For more information, see the CDD algoritm description on 
Github:

https://github.com/bullocke/ge-cdd/blob/master/cdd_simple_description.md  

Note that the formatting is designed for conversion to javascript with
Jiphy

Usage: GE_cdd.py [options] <path> <row> <param> <output>

"""

from docopt import docopt
import os,sys
import numpy as np
import datetime
import pandas as pd
import matplotlib.dates as mdates
from matplotlib import dates
from pylab import *
import yaml
import ee

ee.Initialize()
print('Earth Engine Initialized')


# COMMAND LINE OPTIONS
if __name__ == '__main__':
    args = docopt(__doc__, version='0.6.2')

output=str(args['<output>'])
params = str(args['<param>'])
path = int(args['<path>'])
row = int(args['<row>'])

# PARAMETERS
try:
    with open(params, 'r') as f:
        opts = yaml.safe_load(f)
except:
    print("Invalid config file")
    sys.exit()

consec = opts['thresholds']['consec']
thresh = opts['thresholds']['thresh']
cloud_score = opts['thresholds']['cloud_score']
cf_thresh = opts['thresholds']['cf_thresh']
year = opts['input']['year']
train_length = int(opts['input']['train_length'])
noharm = opts['thresholds']['noharm']
aoi = opts['input']['aoi']
country = opts['input']['country']
if aoi:
    aoi = ee.Geometry.Polygon(aoi)
    pathrow = False
elif country:
    aoi = ee.FeatureCollection('ft:1tdSwUL7MVpOauSgRzqVTOwdfy17KDbw-1d9omPw'
      ).filter(ee.Filter.eq('Country', country)).geometry()
    pathrow = False
else:
    pathrow = True
    

asset = opts['output']['asset']
folder_name = opts['output']['folder_name']

gv = opts['data']['gv']
npv = opts['data']['npv']
soil = opts['data']['soil']
shade = opts['data']['shade']
cloud = opts['data']['cloud']
reverse = opts['input']['reverse']


# GLOBALS
global aoi
global reverse
global train_length

# FUNCTIONS 
def unmix(image):
  """ Do spectral unmixing on a single image """
  unmixi = ee.Image(image).unmix([gv, shade, npv, soil, cloud], True, 
				 True)
  newimage = ee.Image(image).addBands(unmixi)
  mask = ee.Image(newimage).select('band_4').lt(cf_thresh)
  return newimage.updateMask(mask)


# NDFI functions
def get_ndfi(image):
  """ Get Normalized Degradation Fraction Index (NDFI) for an image """
  newimage = ee.Image(image).expression(
      '((GV / (1 - SHADE)) - (NPV + SOIL)) / ((GV / (1 - SHADE)) + NPV + SOIL)', {
        'GV': ee.Image(image).select('band_0'),
        'SHADE': ee.Image(image).select('band_1'),
        'NPV': ee.Image(image).select('band_2'),
        'SOIL': ee.Image(image).select('band_3')
      })
  return ee.Image(image).addBands(ee.Image(newimage).rename(['NDFI'])
    ).select(['band_0','band_1','band_2','band_3','NDFI'])


def makeVariables_noharm(image):
  """ Computes the predictors and the response from the input. """
  # Compute time of the image in fractional years relative to the Epoch.
  year = ee.Image(image.date().difference(ee.Date('1970-01-01'), 'year')) 
  # Compute the season in radians, one cycle per year.
  season = year.multiply(2 * np.pi)
  # Return an image of the predictors followed by the response.
  return image.select().addBands(ee.Image(1)).addBands(
    ee.Image(0).rename(['sin'])).addBands(
    ee.Image(0).rename(['cos'])).addBands(
    image.select(['NDFI'])).toFloat()


def makeVariables(image):
  """ Computes the predictors and the response from the input. """
  # Compute time of the image in fractional years relative to the Epoch.
  year = ee.Image(image.date().difference(ee.Date('1970-01-01'), 'year')) 
  # Compute the season in radians, one cycle per year.
  season = year.multiply(2 * np.pi)
  # Return an image of the predictors followed by the response.
  return image.select().addBands(ee.Image(1)).addBands(
    season.sin().rename(['sin'])).addBands(
    season.cos().rename(['cos'])).addBands(
    image.select(['NDFI'])).toFloat()


def makeVariables_gv(image):
  """ Make variables for GV regression model """
  year = ee.Image(image.date().difference(ee.Date('1970-01-01'), 'year')) 
  season = year.multiply(2 * np.pi)
  return image.select().addBands(ee.Image(1)).addBands(
    season.sin().rename(['sin'])).addBands(
    season.cos().rename(['cos'])).addBands(
    image.select(['band_0'])).toFloat()


def makeVariables_npv(image):
  """ Make variables for NPV regression model """
  year = ee.Image(image.date().difference(ee.Date('1970-01-01'), 'year')) 
  season = year.multiply(2 * np.pi)
  return image.select().addBands(ee.Image(1)).addBands(
    season.sin().rename(['sin'])).addBands(
    season.cos().rename(['cos'])).addBands(
    image.select(['band_2'])).toFloat()


def makeVariables_soil(image):
  """ Make variables for soil regression model """
  year = ee.Image(image.date().difference(ee.Date('1970-01-01'), 'year')) 
  season = year.multiply(2 * np.pi)
  return image.select().addBands(ee.Image(1)).addBands(
    season.sin().rename(['sin'])).addBands(
    season.cos().rename(['cos'])).addBands(
    image.select(['band_3'])).toFloat()


def makeVariables_shade(image):
  """ Make variables for shade regression model """
  year = ee.Image(image.date().difference(ee.Date('1970-01-01'), 'year')) 
  season = year.multiply(2 * np.pi)
  return image.select().addBands(ee.Image(1)).addBands(
    season.sin().rename(['sin'])).addBands(
    season.cos().rename(['cos'])).addBands(
    image.select(['band_1'])).toFloat()


def makeVariables_fullem(image):
  """ Make variables with all end members """
  year = ee.Image(image.date().difference(ee.Date('1970-01-01'), 'year')) 
  season = year.multiply(2 * np.pi)
  return image.select().addBands(ee.Image(1)).addBands(
    season.sin().rename(['sin'])).addBands(
    season.cos().rename(['cos'])).addBands(
    image.select(['NDFI','band_0','band_1','band_2','band_3'])).toFloat()


def makeVariables_trend(image):
  """ Make variables with trend coefficient """
  year = ee.Image(image.date().difference(ee.Date('1970-01-01'), 'year')) 
  season = year.multiply(2 * np.pi)
  return image.select().addBands(ee.Image(1)).addBands(
    year.rename(['t'])).addBands(
    season.sin().rename(['sin'])).addBands(
    season.cos().rename(['cos'])).addBands(
    image.select(['NDFI'])).toFloat()


def addcoefs(image):
  """ Add coefficients to an image """
  newimage = ee.Image(image).addBands(ee.Image(coefficientsImage))
  return newimage


def predict_ndfi(image):
  """ Predict NDFI based on regression coefficients """
  if noharm:
    pred_ndfi_first = ee.Image(image).expression(
      'constant + (coef_sin * sin) + (coef_cos * cos)', {
        't': ee.Image(image).select('t'),
        'constant': ee.Image(image).select('coef_constant'),
        'coef_sin': ee.Image(0),
        'sin': ee.Image(0), 
        'coef_cos': ee.Image(0), 
        'cos': ee.Image(0) 
      })
  else:
    pred_ndfi_first = ee.Image(image).expression(
      'constant + (coef_sin * sin) + (coef_cos * cos)', {
        't': ee.Image(image).select('t'),
        'constant': ee.Image(image).select('coef_constant'),
        'coef_sin': ee.Image(image).select('coef_sin'),
        'sin': ee.Image(image).select('sin'),
        'coef_cos': ee.Image(image).select('coef_cos'),
        'cos': ee.Image(image).select('cos')
    })
  return ee.Image(image).addBands(ee.Image(pred_ndfi_first).rename(
    ['Predict_NDFI']))


def predict_ndfi_trend(image):
  """ Predict NDFI based on regression coefficients including trend """
  pred_ndfi_first = ee.Image(image).expression(
    'constant + (coef_trend * t) + (coef_sin * sin) + (coef_cos * cos)', {
      't': ee.Image(image).select('t'),
      'constant': ee.Image(image).select('coef_constant'),
      'coef_trend': ee.Image(image).select('coef_trend'),
      'coef_sin': ee.Image(image).select('coef_sin'),
      'sin': ee.Image(image).select('sin'),
      'coef_cos': ee.Image(image).select('coef_cos'),
      'cos': ee.Image(image).select('cos')
    })
  return ee.Image(image).addBands(ee.Image(pred_ndfi_first).rename(
    ['Predict_NDFI']))


def pred_middle_retrain(retrain_middle, retrain_coefs):
  """ Predict middle NDFI for retraining period after disturbance """
  image = ee.Image(retrain_middle).addBands(retrain_coefs)
  pred_ndfi_imd = ee.Image(image).select('Intercept')
  if aoi:
      return pred_ndfi_imd.clip(aoi)
  else:
      return pred_ndfi_imd


def addmean(image):
  """ Add mean residual or RMSE to an image """
  newimage = ee.Image(image).addBands(ee.Image(train_ndfi_mean))
  return newimage


def get_mean_residuals(image):
  """ Get RMSE of time series residuals """
  res = ee.Image(image).select('NDFI').subtract(image.select('Predict_NDFI'))
  sq_res = ee.Image(res).multiply(ee.Image(res)).rename(['sq_res'])
  full_image = ee.Image(image).addBands(ee.Image(sq_res))
  return full_image.select('sq_res')


def get_mean_residuals_norm(image):
  """ Get normalized residuals """
  res = image.select('NDFI').subtract(image.select('Predict_NDFI')).divide(
          ee.Image(train_ndfi_mean)).rename(['res'])
  return image.addBands(res)


def add_changedate_mask(image):
  """ Mask if before change date in time series """
  image_date = image.metadata('system:time_start').divide(
                 ee.Image(315576e5))
  eq_change = image_date.eq(ee.Image(change_dates))
  new_band = ee.Image(1).multiply(image.select('NDFI')).updateMask(
               eq_change).rename('Change')
  return image.addBands(new_band)


def get_abs(image):
  """ Get absolute value of residuals """
  abs = image.select('res').abs().rename('abs_res')
  return image.addBands(abs)

    
def monitor_func(image, new_image):
  """ Main monitoring function to loop over images sequentially
  arguments: 
  image = 9 band image with status of change detection:
      # 1. change (1) or no change (0). used as mask. default: 0
      # 2. consecutive observations passed threshold. default: 0
      # 3. date of change if 1 = 1. default: 0
      # 4. magnitude of change
      # 5. iteprator 
      # 6. change vector information for GV
      # 7. change vector information for shade
      # 8. change vector information for npv
      # 9. change vector information for soil
  new_image: current spectrally unmixed landsat image
  """
  # define working mask. 1 if no change has been detected and no coud
  zero_mask = ee.Image(0)
  # band 1 != 0 and not passed consec
  zero_mask_nc = ee.Image(new_image).select('band_1').eq(
                   ee.Image(1)).multiply(ee.Image(new_image).select(
                   'band_2').lt(ee.Image(consec))).unmask() 
  cloud_mask = ee.Image(image).select('NDFI').mask().neq(ee.Image(0))
  image_monitor = image.unmask()
  # Get normalized residual
  norm_dif = image_monitor.select('NDFI').subtract(
               image_monitor.select('Predict_NDFI'))
  norm_res = ee.Image(norm_dif).divide(ee.Image(train_ndfi_mean))
  # 1 if norm_res > threshold
  # 0 if norm res < threshold, cloud mask
  gt_thresh = ee.Image(norm_res).abs().gt(ee.Image(thresh)).multiply(
                zero_mask_nc).multiply(cloud_mask)
  # Band 2: Consecutive observations beyond threshold
  _band_2 = ee.Image(new_image).select('band_2').add(gt_thresh).multiply(
              ee.Image(new_image).select('band_1'))
  band_2_increase = _band_2.gt(ee.Image(new_image).select('band_2')).Or(
                      cloud_mask.eq(ee.Image(0))) # dont reset w/cloud
  band_2 = _band_2.multiply(band_2_increase)
  # 1 if consec is reached
  # 0 if not consec
  flag_change = band_2.eq(ee.Image(consec))
  # 1 = no change
  # 0 = change
  band_1 = ee.Image(new_image).select('band_1').eq(ee.Image(1)).And(
             flag_change.eq(ee.Image(0)))
  # add date of image to band 3 of status where change has been flagged
  # time is stored in milliseconds since 01-01-1970, so scale to be days since then 
  change_date = image.metadata('system:time_start').multiply(gt_thresh).divide(
                  ee.Image(315576e5)).multiply(ee.Image(new_image).select(
                  'band_3').eq(ee.Image(0)))
  band_3 = ee.Image(new_image).select('band_3').add(change_date).multiply(
             ee.Image((gt_thresh.eq(1).Or(cloud_mask.eq(0)).Or(ee.Image(
             new_image).select('band_1').eq(0)))))
  # Magnitude and endmembers
  magnitude = ee.Image(norm_dif).multiply(gt_thresh).multiply(zero_mask_nc)
  endm1 = ee.Image(image_monitor).select('band_0').multiply(
            gt_thresh).multiply(zero_mask_nc)
  endm2 = ee.Image(image_monitor).select('band_1').multiply(
            gt_thresh).multiply(zero_mask_nc)
  endm3 = ee.Image(image_monitor).select('band_2').multiply(
            gt_thresh).multiply(zero_mask_nc)
  endm4 = ee.Image(image_monitor).select('band_3').multiply(
            gt_thresh).multiply(zero_mask_nc)
  # Keep mag if already changed or in process
  is_changing = band_1.eq(ee.Image(0)).Or(band_2).gt(ee.Image(0))
  band_4 = ee.Image(new_image).select('band_4').add(magnitude).multiply(
             is_changing)
  # Add one to iteration band
  band_5 = ee.Image(new_image).select('band_5').add(ee.Image(1))
  # Add endmember bands
  band_6 = ee.Image(new_image).select('band_6').add(endm1).multiply(
             is_changing)
  band_7 = ee.Image(new_image).select('band_7').add(endm2).multiply(
             is_changing)
  band_8 = ee.Image(new_image).select('band_8').add(endm3).multiply(
             is_changing)
  band_9 = ee.Image(new_image).select('band_9').add(endm4).multiply(
             is_changing)
  new_image = band_1.addBands([band_2,band_3,band_4,band_5,band_6,
                band_7,band_8,band_9])
  return new_image.rename(['band_1','band_2','band_3','band_4','band_5',
           'band_6','band_7','band_8','band_9'])


def mask_57(img):
  """ Cloud mask for Landsat 5 and 7 """
  mask = img.select(['cfmask']).neq(4).And(img.select(['cfmask']).neq(2)).And(
           img.select('B1').gt(ee.Image(0)))
  if aoi:
    return img.updateMask(mask).select(['B1','B2', 'B3','B4','B5','B7']).clip(aoi)
  else:
    return img.updateMask(mask).select(['B1','B2', 'B3','B4','B5','B7'])


def mask_8(img):
  """ Cloud mask for Landsat 8 """
  mask = img.select(['cfmask']).neq(4).And(img.select(['cfmask']).neq(2)).And(
           img.select('B2').gt(ee.Image(0)))
  if aoi:
    return ee.Image(img.updateMask(mask).select(['B2', 'B3','B4','B5','B6','B7']
             ).rename(['B1','B2','B3','B4','B5','B7'])).clip(aoi)
  else:
    return ee.Image(img.updateMask(mask).select(['B2', 'B3','B4','B5','B6','B7']
             ).rename(['B1','B2','B3','B4','B5','B7']))


def get_inputs_training(_year, path, row):
  """ Get inputs for training period """
  if reverse:
      train_year_start = _year + 1 
      train_start = str(train_year_start) + '-01-01'
      train_end = str(_year+6) + '-12-31'
  else:
      year = _year - 1
      #Get 5 years worth of trianing data
      train_year_start = str(_year - train_length) 
      train_start = train_year_start + '-01-01'
      train_end = str(year-1) + '-12-31'
  if pathrow:  
    train_collection7 = ee.ImageCollection('LANDSAT/LE7_SR'
      ).filterDate(train_start, train_end
      ).filter(ee.Filter.eq('WRS_PATH', path)
      ).filter(ee.Filter.eq('WRS_ROW', row))
    train_collection5 = ee.ImageCollection('LANDSAT/LT5_SR'
      ).filterDate(train_start, train_end
      ).filter(ee.Filter.eq('WRS_PATH', path)
      ).filter(ee.Filter.eq('WRS_ROW', row))
  else:
    train_collection7 = ee.ImageCollection('LANDSAT/LE7_SR'
      ).filterDate(train_start, train_end
      ).filterBounds(aoi)
 
    train_collection5 = ee.ImageCollection('LANDSAT/LT5_SR'
      ).filterDate(train_start, train_end
      ).filterBounds(aoi)                   
  # Mask clouds
  train_col7_noclouds = train_collection7.map(mask_57).map(add_cloudscore7)
  train_col5_noclouds = train_collection5.map(mask_57).map(add_cloudscore5)
  train_col_noclouds = train_col7_noclouds.merge(train_col5_noclouds)
  # Training collection unmixed
  train_col_unmix = train_col_noclouds.map(unmix)
  # Collection NDFI
  train_ndfi = ee.ImageCollection(train_col_unmix.map(get_ndfi)
                 ).sort('system:time_start')
  if reverse:
      train_nfdi = ee.ImageCollection(train_ndfi.toList(10000000).reverse())
  return train_ndfi


def get_inputs_monitoring(year, path, row):
  """ Get inputs for monitoring period """
  if reverse:
      train_year_start = year + 1 
      monitor_end = str(train_year_start) + '-01-01'
      monitor_start = str(year-21) + '-12-31'
  else:
      monitor_start = str(year) + '-01-01'
      monitor_end = str(year+14) + '-12-31'
  if pathrow: 
    collection8 = ee.ImageCollection('LANDSAT/LC8_SR'
      ).filterDate(monitor_start, monitor_end
      ).filter(ee.Filter.eq('WRS_PATH', path)
      ).filter(ee.Filter.eq('WRS_ROW', row))
    collection7 = ee.ImageCollection('LANDSAT/LE7_SR'
      ).filterDate(monitor_start, monitor_end
      ).filter(ee.Filter.eq('WRS_PATH', path)
      ).filter(ee.Filter.eq('WRS_ROW', row))
    collection5 = ee.ImageCollection('LANDSAT/LT5_SR'
      ).filterDate(monitor_start, monitor_end
      ).filter(ee.Filter.eq('WRS_PATH', path)
      ).filter(ee.Filter.eq('WRS_ROW', row))
  else:
    collection8 = ee.ImageCollection('LANDSAT/LC8_SR'
      ).filterDate(monitor_start, monitor_end
      ).filterBounds(aoi)
    collection7 = ee.ImageCollection('LANDSAT/LE7_SR'
      ).filterDate(monitor_start, monitor_end
      ).filterBounds(aoi)
    collection5 = ee.ImageCollection('LANDSAT/LT5_SR'
      ).filterDate(monitor_start, monitor_end
      ).filterBounds(aoi)
  # Mask clouds
  col8_noclouds = collection8.map(mask_8).map(add_cloudscore8)
  col7_noclouds = collection7.map(mask_57).map(add_cloudscore7)
  col5_noclouds = collection5.map(mask_57).map(add_cloudscore5)
  # Merge
  col_l87noclouds = col8_noclouds.merge(col7_noclouds)
  col_noclouds = col_l87noclouds.merge(col5_noclouds)
  # Training collection unmixed
  col_unmix = col_noclouds.map(unmix)
  # Collection NDFI
  ndfi = ee.ImageCollection(col_unmix.map(get_ndfi)).sort('system:time_start')
  return ndfi


def get_inputs_retrain(year, path, row):
  """ Get inputs for retraining period """
  if reverse:
      monitor_start = str(year-20) + '-01-01'
      # Add extra time for classification
      monitor_end = str(year) + '-12-31'
  else:
      monitor_start = str(year) + '-01-01'
      # Add extra time for classification
      year_end = year + 20
      monitor_end = str(year_end) + '-12-31'
  if pathrow: 
    collection8 = ee.ImageCollection('LANDSAT/LC8_SR'
      ).filterDate(monitor_start, monitor_end
      ).filter(ee.Filter.eq('WRS_PATH', path)
      ).filter(ee.Filter.eq('WRS_ROW', row))
    collection7 = ee.ImageCollection('LANDSAT/LE7_SR'
      ).filterDate(monitor_start, monitor_end
      ).filter(ee.Filter.eq('WRS_PATH', path)
      ).filter(ee.Filter.eq('WRS_ROW', row))
    collection5 = ee.ImageCollection('LANDSAT/LT5_SR'
      ).filterDate(monitor_start, monitor_end
      ).filter(ee.Filter.eq('WRS_PATH', path)
      ).filter(ee.Filter.eq('WRS_ROW', row))
  else:
    collection8 = ee.ImageCollection('LANDSAT/LC8_SR'
      ).filterDate(monitor_start, monitor_end
      ).filterBounds(aoi)
    collection7 = ee.ImageCollection('LANDSAT/LE7_SR'
      ).filterDate(monitor_start, monitor_end
      ).filterBounds(aoi)
    collection5 = ee.ImageCollection('LANDSAT/LT5_SR'
      ).filterDate(monitor_start, monitor_end
      ).filterBounds(aoi)
  # Mask clouds
  col8_noclouds = collection8.map(mask_8).map(add_cloudscore8) 
  col7_noclouds = collection7.map(mask_57).map(add_cloudscore7)
  col5_noclouds = collection5.map(mask_57).map(add_cloudscore5)
  # merge
  col_l87noclouds = col8_noclouds.merge(col7_noclouds)
  col_noclouds = col_l87noclouds.merge(col5_noclouds)
  # Training collection unmixed
  col_unmix = col_noclouds.map(unmix)
  # Collection NDFI
  ndfi = ee.ImageCollection(col_unmix.map(get_ndfi)).sort('system:time_start')
  if reverse:
      nfdi = ee.ImageCollection(ndfi.toList(10000000).reverse())
  return ndfi


def get_regression_coefs(train_array):
  """ Get regression coefficients for the training period """
  # Code from Google Earth Engine tutorial:
  # https://developers.google.com/earth-engine/reducers_regression
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
  coefficientsImage = coefficients.arrayProject([0]).arrayFlatten(
                        [['coef_constant', 'coef_sin', 'coef_cos']])
  return coefficientsImage


def get_regression_coefs_trend(train_array):
  """ Get regression coefficients for the training period """
  # See above for documentation
  imageAxis = 0
  bandAxis = 1
  arrayLength = train_array.arrayLength(imageAxis)
  train_array = train_array.updateMask(arrayLength.gt(4))
  predictors = train_array.arraySlice(bandAxis, 0, 4)
  response = train_array.arraySlice(bandAxis, 4)
  coefficients = predictors.matrixPseudoInverse().matrixMultiply(response)
  coefficientsImage = coefficients.arrayProject([0]).arrayFlatten(
                        [['coef_constant','coef_trend', 'coef_sin', 'coef_cos']])
  return coefficientsImage


def get_coefficient_image(collection, band):
  """ Turn regression array information into an image """
  array = ee.ImageCollection(collection).map(makeVariables_fullem).select(
            ['constant','sin','cos',band]).toArray()
  band_coefs_image = get_regression_coefs(array)
  return band_coefs_image


def deg_monitoring(year, ts_status, path, row, old_coefs, train_ndfi, 
                     first, tmean):
  """ Main function for monitoring, should be looped over for each year """
  #  REGRESSION
  # train array = ndfi collection as arrays
  train_array = ee.ImageCollection(train_ndfi).select('NDFI').map(
                            makeVariables).toArray()
  train_array_trend = ee.ImageCollection(train_ndfi).select('NDFI').map(
                        makeVariables_trend).toArray()
  # train_all = train array with temporal variables attached
  if noharm:
    train_all = ee.ImageCollection(train_ndfi).select('NDFI').map(makeVariables_noharm)
  else:
    train_all = ee.ImageCollection(train_ndfi).select('NDFI').map(makeVariables)
  # coefficients image = image with regression coefficients for each pixel
  global coefficientsImage
  coefficientsImage = get_regression_coefs(train_array)
  coefficientsImage_trend = ee.Image(get_regression_coefs_trend(
                              train_array_trend))
  # Do it again with GV and shade - for forest classification
  train_array_gv = ee.ImageCollection(train_ndfi).select('band_0').map(
                     makeVariables_gv).toArray()
  coefficientsImage_gv = get_regression_coefs(train_array_gv)
  train_array_shade = ee.ImageCollection(train_ndfi).select('band_1').map(
                        makeVariables_shade).toArray()
  coefficientsImage_shade = get_regression_coefs(train_array_shade)
  train_array_npv = ee.ImageCollection(train_ndfi).select('band_2').map(
                      makeVariables_npv).toArray()
  coefficientsImage_npv = get_regression_coefs(train_array_npv)
  train_array_soil = ee.ImageCollection(train_ndfi).select('band_3').map(
                       makeVariables_soil).toArray()
  coefficientsImage_soil = get_regression_coefs(train_array_soil)
  # check change status. If mid-change - use last year's coefficients. 
  train_coefs = ee.ImageCollection(train_all).map(addcoefs)
  # predict_ndfi_train = predicted NDFI based on regression coefficients 
  predict_ndfi_train = ee.ImageCollection(train_coefs).map(predict_ndfi)
  global train_ndfi_mean
  train_ndfi_sq_res = predict_ndfi_train.map(get_mean_residuals).mean()
  train_ndfi_mean = ee.Image(train_ndfi_sq_res).sqrt().rename(['mean_res'])
  #  MONITORING PERIOD 
  # Monitor_NDFI = NDFI for monitoring period
  monitor_ndfi = get_inputs_monitoring(year, path, row)
  # monitor_collection = ndfi for monitoring period with temporal features
  monitor_collection = ee.ImageCollection(monitor_ndfi).map(makeVariables_fullem)
  # monitor_collection_coefs = monitor_collection with temporal coefficients 
  monitor_collection_coefs = ee.ImageCollection(monitor_collection).map(addcoefs)
  # predict_ndfi_monitor = predicted NDFI based on regression coefficients
  predict_ndfi_monitor = ee.ImageCollection(monitor_collection_coefs).map(
                           predict_ndfi)
  # predict_ndfi_monitor_mean = predict_ndfi_monitor with mean residual from training
  predict_ndfi_monitor_mean = ee.ImageCollection(predict_ndfi_monitor).map(addmean)
  # monitor_ndfi_prs = subset of predict_ndfi_monitor_mean with just predicted ndfi, 
  # real ndfi, mean training residuals, and endmembers
  monitor_ndfi_prs = ee.ImageCollection(predict_ndfi_monitor_mean).select(
                       ['NDFI','Predict_NDFI','mean_res','band_0','band_1',
                       'band_2','band_3'])
  # monitor_abs_res = normalized residuals for predicted versus real ndfi
  monitor_abs_res = ee.ImageCollection(monitor_ndfi_prs).map(get_mean_residuals_norm)
  # results = results of change detection iteration
  results = ee.Image(monitor_ndfi_prs.iterate(monitor_func, ts_status))
  # combine monitoring ndfi with training
  new_training = ee.ImageCollection(train_ndfi).merge(monitor_ndfi).sort(
                   'system:time_start')
  return ee.List([results, 
                  coefficientsImage, 
                  new_training, 
                  train_ndfi_mean, 
                  coefficientsImage_trend, 
                  coefficientsImage_gv, 
                  coefficientsImage_shade, 
                  coefficientsImage_npv, 
                  coefficientsImage_soil])


def mask_nochange(image):
  """ mask retrain stack if there has been no change """
  ischanged = ee.Image(change_dates).gt(ee.Image(0))
  if aoi:
      return ee.Image(image).updateMask(ischanged).clip(aoi)
  else:
      return ee.Image(image).updateMask(ischanged)


def mask_beforechange(image):
  """ mask retrain stack before change """
  im_date = ee.Image(image).metadata('system:time_start').divide(
              ee.Image(315576e5))
  #Skip 2 years during regrowth
  skip_year = change_dates.add(ee.Image(2))
  af_change = im_date.gt(skip_year)
  if aoi:
      return ee.Image(image).updateMask(af_change).clip(aoi)
  else:
      return ee.Image(image).updateMask(af_change)


def regression_retrain(original_collection, year, path, row):
  """ main function for regression after disturbance """
  # get a few more years data
  y1_data = get_inputs_retrain(year, path, row)
  full_data_nomask = original_collection.merge(y1_data).sort('system:time_start')
  stack_nochange_masked = full_data_nomask.map(mask_nochange)
  stack_masked= stack_nochange_masked.map(mask_beforechange)
  #run regression on data after a change
  train_iables = ee.ImageCollection(stack_nochange_masked).map(makeVariables)
  train_array = train_iables.toArray()
  # coefficients image = image with regression coefficients for each pixel
  _coefficientsImage = get_regression_coefs(train_array)
  coefficientsImage = _coefficientsImage 
  retrain_with_coefs = ee.ImageCollection(train_iables).map(addcoefs)
  retrain_predict1 = ee.ImageCollection(retrain_with_coefs).map(predict_ndfi)
  retrain_predict2 = ee.ImageCollection(retrain_predict1).map(mask_nochange)
  retrain_predict = ee.ImageCollection(retrain_predict1).map(mask_beforechange)
  retrain_ndfi_sq_res = retrain_predict.map(get_mean_residuals).mean()
  retrain_ndfi_mean = ee.Image(retrain_ndfi_sq_res).sqrt().rename(['mean_res'])
  # do it again for fractions
  retrain_array_gv = ee.ImageCollection(stack_nochange_masked).select(
                       'band_0').map(makeVariables_gv).toArray()
  r_coefficientsImage_gv = ee.Image(get_regression_coefs(
                             retrain_array_gv)).select(['coef_constant'])
  retrain_array_shade = ee.ImageCollection(stack_nochange_masked).select(
                          'band_1').map(makeVariables_shade).toArray()
  r_coefficientsImage_shade = ee.Image(get_regression_coefs(
                                retrain_array_shade)).select(['coef_constant'])
  retrain_array_npv = ee.ImageCollection(stack_nochange_masked).select(
                        'band_2').map(makeVariables_npv).toArray()
  r_coefficientsImage_npv = ee.Image(get_regression_coefs(
                              retrain_array_npv)).select(['coef_constant'])
  retrain_array_soil = ee.ImageCollection(stack_nochange_masked).select(
                         'band_3').map(makeVariables_soil).toArray()
  r_coefficientsImage_soil = ee.Image(get_regression_coefs(
                               retrain_array_soil)).select(['coef_constant'])
  return ee.List([_coefficientsImage.rename(['Intercept', 'Sin','Cos']), 
                  retrain_predict, 
                  r_coefficientsImage_gv, 
                  r_coefficientsImage_shade, 
                  r_coefficientsImage_npv, 
                  r_coefficientsImage_soil,
                  retrain_ndfi_mean])


def add_cloudscore5(image):
   """ Add cloud score for Landsat 5 """
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
          ee.Image(ee.Algorithms.Landsat.simpleCloudScore(
            ee.Image(toa))).select('cloud'),
    ee.Image(0).rename(['cloud']))
   mask = ee.Image(cs).lt(ee.Image(cloud_score))
   if aoi:
     return image.updateMask(mask).select(['B1','B2', 'B3','B4','B5','B7']).clip(aoi)
   else:
     return image.updateMask(mask).select(['B1','B2', 'B3','B4','B5','B7'])


def add_cloudscore7(image):
   """ Add cloud score for Landsat 7 """
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
          ee.Image(ee.Algorithms.Landsat.simpleCloudScore(
          ee.Image(toa))).select('cloud'),
          ee.Image(0).rename(['cloud']))
   mask = ee.Image(cs).lt(ee.Image(cloud_score))
   if aoi:
     return image.updateMask(mask).select(['B1','B2', 'B3','B4','B5','B7']).clip(aoi)
   else:
     return image.updateMask(mask).select(['B1','B2', 'B3','B4','B5','B7'])


def add_cloudscore8(image):
   """ Add cloud score for Landsat 8 """
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
          ee.Image(ee.Algorithms.Landsat.simpleCloudScore(
          ee.Image(toa))).select('cloud'), 
          ee.Image(0).rename(['cloud']))
   mask = ee.Image(cs).lt(ee.Image(cloud_score))
   if aoi:
     return image.updateMask(mask).select(['B1','B2', 'B3','B4','B5','B7']).clip(aoi)
   else:
     return image.updateMask(mask).select(['B1','B2', 'B3','B4','B5','B7'])


#  MAIN WORK 

# JAVASCRIPT GLOBALS

coefficientsImage = ""
train_ndfi_mean = ""
change_dates = ""

ts_status = ee.Image(1).addBands([ee.Image(0),ee.Image(0),ee.Image(0),
                                    ee.Image(1),ee.Image(0),ee.Image(0),
                                    ee.Image(0),ee.Image(0)]).rename(
                                   ['band_1','band_2','band_3','band_4',
                                   'band_5','band_6','band_7','band_8',
                                   'band_9']).unmask()

old_coefs = ee.Image(0)

# First year inputs
train_ndfi = get_inputs_training(year, path, row)

results = deg_monitoring(year, ts_status, path, row, old_coefs, 
            train_ndfi, True, None)

# RMSE for training period
tmean = results.get(3) #TODO: Add to output

# RMSE for training period
before_rmse = ee.Image(tmean)

# Change detection results
final_results = ee.Image(results.get(0)) 

# Coefficients during training period
old_coefs = results.get(1) 

# Training data as image collection
final_train = ee.ImageCollection(results.get(2)) #TODO

# Regression information for classification
coefficientsImage_trend = ee.Image(results.get(4))
coefficientsImage_gv = ee.Image(results.get(5)).select(['coef_constant'])
coefficientsImage_shade = ee.Image(results.get(6)).select(['coef_constant'])
coefficientsImage_npv = ee.Image(results.get(7)).select(['coef_constant'])
coefficientsImage_soil = ee.Image(results.get(8)).select(['coef_constant'])

global change_dates
has_changed = ee.Image(final_results.select('band_1')).eq(ee.Image(0))
change_dates = ee.Image(final_results.select('band_3')).multiply(has_changed)

# Retrain after disturbance
retrain_regression = regression_retrain(final_train, year, path, row)

retrain_coefs = ee.Image(retrain_regression.get(0))
retrain_predict = ee.ImageCollection(retrain_regression.get(1))
retrain_gv = ee.Image(retrain_regression.get(2))
retrain_shade = ee.Image(retrain_regression.get(3))
retrain_npv = ee.Image(retrain_regression.get(4))
retrain_soil = ee.Image(retrain_regression.get(5))
retrain_predict_last = ee.Image(retrain_predict.toList(1000).get(-1))

# NDFI RMSE for retraining period
after_rmse = ee.Image(retrain_regression.get(6))

# Get predicted NDFI at middle of time series
retrain_last = ee.Image(retrain_predict.toList(1000).get(-1))

if aoi:
  retrain_last_date = ee.Image(retrain_last).metadata('system:time_start'
                        ).divide(ee.Image(31557600000)).clip(aoi)
else:
  retrain_last_date = ee.Image(retrain_last).metadata('system:time_start'
                        ).divide(ee.Image(31557600000))

# get the date at the middle of the retrain time series
retrain_middle = ee.Image(ee.Image(retrain_last_date).subtract(
                   ee.Image(change_dates)).divide(ee.Image(2)).add(
                   ee.Image(change_dates))).rename(['years'])
predict_middle = pred_middle_retrain(retrain_middle, retrain_coefs)

# OUTPUT

# Normalize magnitude
# st_magnitude = short-term change magnitude
st_magnitude = final_results.select('band_4').divide(
                 ee.Image(consec)).multiply(change_dates.gt(ee.Image(0)))

# Endmembers 
change_endm1 = final_results.select('band_6').divide(
                 ee.Image(consec)).multiply(change_dates.gt(ee.Image(0)))
change_endm2 = final_results.select('band_7').divide(
                 ee.Image(consec)).multiply(change_dates.gt(ee.Image(0)))
change_endm3 = final_results.select('band_8').divide(
                 ee.Image(consec)).multiply(change_dates.gt(ee.Image(0)))
change_endm4 = final_results.select('band_9').divide(
                 ee.Image(consec)).multiply(change_dates.gt(ee.Image(0)))

# save_output

# bands 1-6: change information
# band 1: change date
# band 2: change magnitude
# band 3: change vector GV
# band 4: change vector NPV
# band 5: change vector soil
# band 6: change vector shade
# bands 7-14: pre-disturbance training information
# band 7: mag NDFI training
# band 8: RMSE NDFI training #TODO
# band 9: sine training
# band 10: cos training
# band 11: GV training
# band 12: shade training
# band 13: NPV training
# band 14: soil training
# bands 15-22: post-disturbance retrain information
# band 15: mag NDFI after
# band 16: rmse NDFI after
# band 17: sin after
# band 18: cos after
# band 19: GV after
# band 20: shade after
# band 21: npv after
# band 22: soil after
save_output = change_dates.rename(['change_date']).addBands( 
                [ee.Image(st_magnitude).rename(['magnitude']) 
                , ee.Image(change_endm1).rename(['change_v_gv'])  
                , ee.Image(change_endm2).rename(['change_v_npv']) 
                , ee.Image(change_endm3).rename(['change_v_soil'])
                , ee.Image(change_endm4).rename(['change_v_shade'])
                , ee.Image(old_coefs).select('coef_constant').rename(['mag_training'])
                , ee.Image(before_rmse).rename(['rmse_training'])
                , ee.Image(old_coefs).select('coef_sin').rename(['sine_training'])
                , ee.Image(old_coefs).select('coef_cos').rename(['cos_training'])
                , ee.Image(coefficientsImage_gv).rename(['pre_gv'])
                , ee.Image(coefficientsImage_shade).rename(['pre_shade'])
                , ee.Image(coefficientsImage_npv).rename(['pre_npv'])
                , ee.Image(coefficientsImage_soil).rename(['pre_soil'])
                , ee.Image(predict_middle).rename(['ndfi_retrain'])
                , ee.Image(after_rmse).rename(['rmse_after'])
                , ee.Image(retrain_coefs).select('Sin').rename(['retrain_sin'])
                , ee.Image(retrain_coefs).select('Cos').rename(['retrain_cos'])
                , ee.Image(retrain_gv).rename(['after_gv'])
                , ee.Image(retrain_shade).rename(['after_shade'])
                , ee.Image(retrain_npv).rename(['after_npv'])
                , ee.Image(retrain_soil).rename(['after_soil'])]).toFloat()

print('Submitting task')

task_config = {
  'description': output,
  'scale': 30
  }

project_root='users/bullocke'

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


