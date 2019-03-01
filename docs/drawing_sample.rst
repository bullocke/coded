Sampling
========

*Instructions on generating a sample.*

All maps that were constructed from classification of remote sensing imagery will contain errors. These errors are inevitable and are due to many reasons including missed clouds, similarity between classes, and climate variability. Therefore, areas calculated directly from the map through "pixel counting" will be incorrect. It is instead recommended that calculation of areas, whether it be land cover or land cover change, be produced by applying an unbiased estimator to a reference sample [1]. Sample based estimation allows for the calculation of unbiased area estimates in addition to the quantification of uncertainties associated with the areas.  

Area estimation can be difficult when the class of interest is small relative to the population. This is generally the case when comparing the area of degraded forests to that of an entire country or region. Remote sensing data allows for the spatially explicit stratification of a landscape, with small land cover or change classes represented by unique strata. These strata can then be sampled to ensure that the small land classes are represented. This is called stratified sampling, with areas being calculated through the stratified estimator [2].

This section is incomplete. Tools and guidance for sampling and estimation on the Earth Engine can be found in the `Google Earth Engine Accuracy and Area Estimation Toolbox (AREA2)`_. 

.. _Google Earth Engine Accuracy and Area Estimation Toolbox (AREA2): https://github.com/bullocke/area2 

.. [1] GFOI, 2016. Integration of remote-sensing and ground-based observations for estimation of emissions and removals of greenhouse gases in forests: Methods and Guidance from the Global Forest Observations Initiative 2, 226.
.. [2] Cochran, W.G., 1977. Sampling techniques. New York John Wiley Sons. 
