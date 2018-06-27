## Instructions for running the Continuous Degradation Detection (CODED) algorithm on the Google Earth Engine (GEE). Version 2.0 for the GEE Javascript API. 

### Change Detection

Change detection is performed using the 'submit_change' file. To use the most recent version of CODED, a copy of 'submit_change' can be made locally with the utility paths directing towards my repository. Alternatively, the repository can be cloned, in which case the paths would have to be adjusted accordingly. 

There are 4 primary inputs to the algorithm: training data to perform land cover classification, a parameter file containing various model parameters, a FeatureCollection defining the study area, and the previous iteration of the algorithm, it it exists. 

#### Training Data

Training data must be a single FeatureCollection containing points with a unique class identifier named 'label'. The training data must represent the class during the entire training period. 

#### Parameters

| Parameter | Description | Example |
| --- | --- | --- |
| cloud_score | minimum threshold to remove clouds based on ee.simpleCloudScore | 80 |
| cfThreshold | minimum threshold to remove clouds based on cloud fraction | .05 |
| consenc | consecutive observations beyond change threshold to trigger a change | 3 |
| thresh | change threshold defined as an observation's residual normalized by the training models RMSE | 5 |
| minRMSE | minimum RMSE for creating the change score | .015 |
| trainStart | beginning of training period | 1992 |
| trainEnd | end of training period | 1996 |
| trainDataStart | beginning of time period associated with training data | 1992 |
| trainDataEnd | end of time period associated with training data | 1996 |
| monitorEnd | end of monitoring period | 2018 |
| iteration | the iteration of the change detection | 1 |
| soil | soil endmember reflectance values for each band | see example | 
| gv | green vegetation endmember reflectance values for each band | see example |  
| npv | non-photosynthetic vegetation endmember reflectance values for each band | see example |  
| shade | shade endmember reflectance values for each band | see example |  
| cloud | cloud endmember reflectance values for each band | see example |  
| forestLabel | label of forest in the training data | 1 |
| outBands | output bands to save | see example |

#### Region

The study region must be defined using a FeatureCollection named 'region'. The region can be defined in the Earth Engine by drawing a 'shape' and naming it 'region'. The region can cross Landsat Path/Rows, but too big of an area will likely cause errors.

#### Previous Iteration

To reduce computational intensity only one change is detected for every location over the monitoring period. However, multiple changes for the same location can be detected by submitting multiple jobs, with 'previousIteration' defined as the output of the previous change detection. For the first iteration 'previousIteration' should be set to 0. 

### Outputs

The output of the change detection is an image with various information. For every iteration, a set of the bands will be created with a number representing the iteration. 

| band name | Description |
| --- | --- |
| changeDate | date of change in years since 1970
| changeMag | change magnitude defined as the average change score during the change period |
| numObs | number of observations during monitoring period before change |
| postDistClass | post-disturbance landcover class according to 'label' in training data |
| difNDFI | NDFI magnitude after disturbance compared to before expressed as proportion * 100 | 
| firstRegrowth | first year after a disturbance classified as forest |
| forestTraining | value of 1 if the location was classified as forest during training period |
| cusumStat | CUSUM test statistic for omission test |
| cusumDate | date of maximum CUSUM test statistic | 
| TrainForestProb | RF's forest classification probability during training period
| postCuFProb | Forest probability after CUSUM detected change |
