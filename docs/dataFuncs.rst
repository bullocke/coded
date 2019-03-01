Utility Functions
======================

*Utility functions for running CODED and dealing with outputs.* 

To load all the necessary functions for the running CODED, the changeDetection and dataUtils files must be laoded:

.. code-block:: javascript

  var dataUtils = require('users/bullocke/coded:coded/dataUtils')
  var codedUtils = require('users/bullocke/coded:coded/changeDetection')

submitCODED
-----------

Usage:
  submitCODED(saveRegion, params, trainingData)

Description:
  Main function for running CODED.

Arguments:
  saveRegion (Feature Collection): Geographic region to run the analysis. 

  params (Dictionary): Dictionary defining CODED parameters.

  trainingData (Feature Collection): Feature points containing training data with unique land cover labels identified in the 'label' attribute. 

**Example:**

.. code-block:: javascript

   var results = codedUtils.submitCODED(saveRegion, params, trainingData)

makeImage
---------

Usage:
 makeImage(arrayImage, column, prefix, startYear, endYear)

Description:
  Make a saveable image from the data array output that is retured by CODED. 

Arguments:
  arrayImage (array): Output array image from CODED

  column (int): column of multi-dimensional array image to turn into image.

  prefix (string): Prefix to give to all bands in output image

  startYear (int): First year in study period.

  endYear (int): Last year in study period.

**Example:**

.. code-block:: javascript

   var disturbances = dataUtils.makeImage(results, 0, 'dist_', params.get('start'), params.get('end'))


reduceBands
-----------

Usage:
  reduceBands(changeBands, params)

Description:
  Reduce the number of output bands to just save information about the number of disturbances specified in the parameter file.

Arguments:
  changeBands (image): Images representing CODED output disturbances, magnitude, post-disturbance land cover, and change difference.

  parameters (dictionary): CODED parameter dictionary. 

**Example:**

.. code-block:: javascript

   var results = codedUtils.submitCODED(saveRegion, params, trainingData)

   var disturbances = dataUtils.makeImage(results, 0, 'dist_', params.get('start'), params.get('end'))
   var magnitude = dataUtils.makeImage(results, 1, 'mag_', params.get('start'), params.get('end'))
   var postChange = dataUtils.makeImage(results, 2, 'post_', params.get('start'), params.get('end'))
   var difference = dataUtils.makeImage(results, 3, 'dif_', params.get('start'), params.get('end'))

   var changeBands = disturbances.addBands([magnitude, postChange, difference])
   var save_output = ee.Image(dataUtils.reduceBands(changeBands, params))
