Parameters
==========

*Parameters for defining the study area*

+-------------------+----------+--------------------------------------------+
| Parameter         | Type     | Description                                |
+===================+==========+============================================+
| countryStudyArea  | boolean  | Use a country boundary for the study area  |
+-------------------+----------+--------------------------------------------+
| country           | string   | Name of country to use as study area       |
+-------------------+----------+--------------------------------------------+
| studyArea         | string   | Asset to use if countryStudyArea is false  |
+-------------------+----------+--------------------------------------------+

**Note**: *studyArea* will be ignored if *countryStudyArea* is true. 


*Parameters for filtering the input Landsat data*

+------------+----------+--------------------+
| Parameter  | Type     | Description        |
+============+==========+====================+
| startDoy   | integer  | Start day of year  |
+------------+----------+--------------------+
| endDoy     | integer  | End day of year    |
+------------+----------+--------------------+
| startYear  | integer  | Start year         |
+------------+----------+--------------------+
| endYear    | integer  | End year           |
+------------+----------+--------------------+


*Parameters for defining a forest mask*

+--------------------+----------+------------------------------------------------------------+
| Parameter          | Type     | Description                                                |
+====================+==========+============================================================+
| useMask            | boolean  | Whether or not to apply a forest mask                      |
+--------------------+----------+------------------------------------------------------------+
| getMaskFromHansen  | boolean  | Whether or not to generate a forest mask from UMD dataset  |
+--------------------+----------+------------------------------------------------------------+
| forestMask         | string   | Path to asset if using mask and not from UMD               |
+--------------------+----------+------------------------------------------------------------+
| focalMode          | integer  | Focal mode window size to apply to mask                    |
+--------------------+----------+------------------------------------------------------------+
| treeCoverThreshold | integer  | Tree cover threshold for UMD dataset                       |
+--------------------+----------+------------------------------------------------------------+

**Note**: All parameters will be ignored if *useMask* is false. *getMaskFromHansen*, *focalMode*, and *treeCoverThreshold* will be ignored if *forestMask* is defined. 


*Parameters for defining training data*

+------------------------+----------+-------------------------------------------------------------+
| Parameter              | Type     | Description                                                 |
+========================+==========+=============================================================+
| getTrainingFromHansen  | boolean  | Whether or not to sample the UMD dataset for training data  |
+------------------------+----------+-------------------------------------------------------------+
| training               | string   | Path to feature collection with training data               |
+------------------------+----------+-------------------------------------------------------------+
| prepTraining           | boolean  | Whether or not to cache predictor data and export asset     |
+------------------------+----------+-------------------------------------------------------------+
| forestValue            | number   | The number associated with forest points                    |
+------------------------+----------+-------------------------------------------------------------+
| numberOfPoints         | number   | Number of points to sample from UMD layer                   |
+------------------------+----------+-------------------------------------------------------------+

**Note**: *training* will be ignored if *getTrainingFromHansen* is true. *numberOfPoints* will be ignored if *getTrainingFromHansen* is false. The first time running CODED, *prepTraining* must be true. 


*Parameters for CODED change detection*

+-----------------------+----------+-------------------------------------------------+
| Parameter             | Type     | Description                                     |
+=======================+==========+=================================================+
| minObservations       | integer  | # of consecutive observations to flag a change  |
+-----------------------+----------+-------------------------------------------------+
| chiSquareProbability  | float    | Threshold that controls sensitivity to change   |
+-----------------------+----------+-------------------------------------------------+


*Parameters for exporting and saving results*

+--------------------------+---------+----------------------------------------------------------+
| Parameter                | Type    | Description                                              |
+==========================+=========+==========================================================+
| outName                  | string  | Output asset ID                                          |
+--------------------------+---------+----------------------------------------------------------+
| numberOfChangesToExport  | integer | # of disturbances to keep in output dataset              |
+--------------------------+---------+----------------------------------------------------------+
| dateInt                  | boolean | Standardized dates to be 8 bit integers                  |
+--------------------------+---------+----------------------------------------------------------+
| maskProb                 | boolean | Mask changes that do not have a change probability of 1  |
+--------------------------+---------+----------------------------------------------------------+
| flipMag                  | boolean | Make NDFI change magnitude positive                      |
+--------------------------+---------+----------------------------------------------------------+
| exportLayers             | object  | Layers to export in image stack                          |
+--------------------------+---------+----------------------------------------------------------+

**Note**: *dateInt* will convert dates so that the date = date - *startYear* + 1.


*Parameters for exporting the results in grid cells*

+----------------+----------+-----------------------------------------------------+
| Parameter      | Type     | Description                                         |
+================+==========+=====================================================+
| exportInGrids  | boolean  | Whether or not to split output into multiple tasks  |
+----------------+----------+-----------------------------------------------------+
| gridFolder     | string   | Path to folder to save gridded results              |
+----------------+----------+-----------------------------------------------------+
| gridSize       | number   | Length of grid edge in degrees                      |
+----------------+----------+-----------------------------------------------------+
| gridPrefix     | string   | Prefix for name to output grid assets               |
+----------------+----------+-----------------------------------------------------+
| gridMin        | number   | Index of first grid to export                       |
+----------------+----------+-----------------------------------------------------+
| gridMax        | number   | Index of last grid to export                        |
+----------------+----------+-----------------------------------------------------+
| predefinedGrid | string   | Path to feature collection with predefined grid     |
+----------------+----------+-----------------------------------------------------+

**Note**: All grid parameters will be ignored and the results will be exported in a single task if *exportInGrids* is false. 
