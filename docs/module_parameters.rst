Parameters
==========

*Parameters from running CODED* 

+----------------+-------+------------------------------------------------------------------------------+ 
| Parameter      | Type  | Description                                                                  |
+================+=======+==============================================================================+
|   thresh       |  int  | change threshold defined as a residual normalized by the training model RMSE |
+----------------+-------+------------------------------------------------------------------------------+ 
|   consec       |  int  | consecutive observations beyond change threshold to trigger a change         |
+----------------+-------+------------------------------------------------------------------------------+ 
| trainlength    |  int  | number of years in training period                                           |
+----------------+-------+------------------------------------------------------------------------------+ 
|   minYears     |  int  | minimum years between disturbances                                           |
+----------------+-------+------------------------------------------------------------------------------+ 
|    start       |  int  | beginning year of study period                                               |
+----------------+-------+------------------------------------------------------------------------------+ 
| end            |  int  | ending year of study period                                                  |
+----------------+-------+------------------------------------------------------------------------------+ 
| trainDataStart | int   | beginning year of time period associated with training data                  |
+----------------+-------+------------------------------------------------------------------------------+ 
| trainDataEnd   | int   | ending year of time period associated with training data                     |
+----------------+-------+------------------------------------------------------------------------------+ 
| soil           | list  | soil endmember reflectance value for each band                               |
+----------------+-------+------------------------------------------------------------------------------+ 
| gv             | list  | green vegetation endmember reflectance value for each band                   |
+----------------+-------+------------------------------------------------------------------------------+ 
| npv            | list  | non-photosynthetic vegetation endmember reflectance value for each band      |
+----------------+-------+------------------------------------------------------------------------------+ 
| shade          | list  | shade endmember reflectance value for each band                              |
+----------------+-------+------------------------------------------------------------------------------+ 
| cloud          | list  | cloud endmember reflectance value for each band                              |
+----------------+-------+------------------------------------------------------------------------------+ 
| cfThreshold    | float | minimum threshold to remove clouds based on cloud fraction                   |
+----------------+-------+------------------------------------------------------------------------------+ 
| forestLabel    | int   | label of forest in training data                                             |
+----------------+-------+------------------------------------------------------------------------------+ 
| numChanges     | int   | maximum number of changes to keep when exporting                             |
+----------------+-------+------------------------------------------------------------------------------+ 
| window         | int   | the max number of years to use in the monitoring period at any given time    |
+----------------+-------+------------------------------------------------------------------------------+ 
| minObs         | int   | the minimum number of observations to fit a model for training               |
+----------------+-------+------------------------------------------------------------------------------+ 

