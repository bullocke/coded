## Instructions for running the Continuous Degradation Detection (CODED) algorithm on the Google Earth Engine (GEE). Version 0.2 for the GEE Javascript API. 

### Parameters

Parameters are defined using a dictionary. 

| Parameter | Description | Example |
| --- | --- | --- |
| consec | consecutive observations beyond change threshold to trigger a change | 3 |
| thresh | change threshold defined as an observation's residual normalized by the training models RMSE | 5 |
| trainLength | number of years in training period | 3 |
| minYears | minimum years between disturbances | 3 | 
| start | beginning of study period | 1992 |
| end | end of study period | 1996 |
| trainDataStart | beginning of time period associated with training data | 1992 |
| trainDataEnd | end of time period associated with training data | 1996 |
| soil | soil endmember reflectance values for each band | see example | 
| gv | green vegetation endmember reflectance values for each band | see example |  
| npv | non-photosynthetic vegetation endmember reflectance values for each band | see example |  
| shade | shade endmember reflectance values for each band | see example |  
| cloud | cloud endmember reflectance values for each band | see example |  
| cfThreshold | minimum threshold to remove clouds based on cloud fraction | .05 |
| forestLabel | label of forest in the training data | 1 |
| minYears | minimum years between disturbances | 3 |
| numChanges | maximum number of changes to keep upon export | 3 |

### Change Detection

There are three inputs to the change detection algorithm: training data, the parameter file, and a FeatureCollection containing the boundary of the study area.

Training data should be a feature collection with each class containing a unique integer identifier in an attribute named "label". For example, all forest samples can have "1" for "label", all water can have "2", and so on. 

First, load the CODED module:

```javascript
var cddUtils = require('users/bullocke/coded:v0.2/changeDetection')
```

Define the study area, with 'region' being an import or path to a feature. 

```javascript
var saveRegion = ee.FeatureCollection(region)
```
Define the sample name and parameters:

```javascript
var sampleName = ee.FeatureCollection('users/bullockebu/amazon/samples/sample1')

var params = ee.Dictionary({
     'consec': 3,
     'thresh': 3,
     'start': 1990,
     'end': 2018,
     'trainDataEnd': 2016,
     'trainDataStart': 2013,
     'trainLength': 3,
     'soil': [2000, 3000, 3400, 5800, 6000, 5800],
     'gv': [500, 900, 400, 6100, 3000, 1000],
     'npv': [1400, 1700, 2200, 3000, 5500, 3000],
     'shade': [0, 0, 0, 0, 0, 0],
     'cloud': [9000, 9600, 8000, 7800, 7200, 6500],
     'cfThreshold': .05,
     'forestLabel': 1,
     'minYears': 3
      })
```

Call the main function of CODED to retrieve the results:

```javascript
var results = cddUtils.submitCODED(saveRegion, params, trainingData)
```

### Outputs

The output of the change detection is an [array image](https://developers.google.com/earth-engine/arrays_array_images). At every pixel location there is an array matrix, with a row for every year in the study period and the columns corresponding to the a change flag (1 = change, 0 = no change), change magnitude, post-change land cover, NDFI difference band, and a forest flag (1 = forest in training period). 

| Column | Description | Range |
| --- | --- | --- |
| 1 | Change flag with 1 indicating change | 0-1 |
| 2 | Change magnitude with higher values representing higher magnitude | 0-255 |
| 3 | Post-change land cover corrresponding to label in training data | 1 - # classes |
| 4 | Difference in NDFI expressed as percent NDFI magnitude after disturbance compared to before | 0-255 |
| 5 | Forest flag with 1 representing forest in training period | 0 - 1 |

The data array that CODED returns can not be saved as an output or asset. The array image needs to be projected as a saveable 3-D image (x, y, and band dimensions). 

First, it helps to define a function to create band names:

```javascript
var makeBands = function(start, end, prefix) {
  var bandSeq = ee.List.sequence(start, end)
  var bandList = bandSeq.map(function(i) {
    return ee.String(prefix).cat(ee.String(i).slice(0,4))
  })
  return bandList
}
```

Next, use a combination of arraySlice, arrayProject, and arrayFlatten to turn one column of the array image into a 3D image with one band for each year: 

```javascript
var makeImage = function(arrayImage, column, bandPrefix, start, end) {
  var bandList = makeBands(start, end, bandPrefix)
  return arrayImage.arraySlice(1, ee.Number(column), ee.Number(column).add(1))
                    .arrayProject([0])
                    .arrayFlatten([bandList])
}

var column = 0 // change flag
var bandPrefix = 'distFlag_' // band prefix
var start = params.get('start') // first year of study period
var end = params.get('end') // last year of study period

var distFlagImage = makeImage(results, column, bandPrefix, start, end)
```

If you create images of all the outputs the images will contain a lot of bands - likely not all of them are necessary. The dataUtils file contains a function to reduce the bands to 4 times the number of changes specified in the parameter dictionary. The output bands are the date of first change, magnitude of first change, the land cover after the first change, the difference in NDFI from before and after the change, the date of second change, and so on. In the following example the CODED output array is turned into 5 images, and then reduced to a smaller image while keeping all the forest flags and attaching the parameter dictionary to the image attributes.

```javascript
var disturbances = dataUtils.makeImage(results, 0, 'dist_', params.get('start'), params.get('end'))
var magnitude = dataUtils.makeImage(results, 1, 'mag_', params.get('start'), params.get('end'))
var postChange = dataUtils.makeImage(results, 2, 'post_', params.get('start'), params.get('end'))
var difference = dataUtils.makeImage(results, 3, 'dif_', params.get('start'), params.get('end'))
var forestFlag = dataUtils.makeImage(results, 4, 'forest_', params.get('start'), params.get('end'))

var disturbanceBands = disturbances.addBands([magnitude, postChange, difference])

var saveOutput = ee.Image(dataUtils.reduceBands(ee.Image(disturbanceBands), params)
                           .setMulti(params))
                           .addBands(forestFlag) 
```

The results can then be submitted as a task:

```javascript
Export.image.toAsset({
  image: saveOutput,
  description: 'imageDescription',
  assetId: 'path/imageDescription',
  maxPixels: 1000000000000,
  scale: 30,
  region: saveRegion.geometry(),
})
```
