# Simple Continuous Degradation Detection

Simple-CDD is a simplified version of the full CDD algorithm that is designed to be more computationally efficient. 
## Pre-processing

All available Landsat data is used a converted to surface reflectance using the standard LEDAPS surface reflectance product. The data is first filtered for clouds using two algorithms: 

  * CFmask product
  * Google Earth Engine simple cloud score

The data are then converted to spectral endmembers using endmembers developed in [Souza et al., 2005](http://www.sciencedirect.com/science/article/pii/S0034425705002385), in addition to a self-developed cloud endmember. The 5 endmembers are:

  * Green Vegetation
  * Non-Photosynthetic Vegetation
  * Shade
  * Soil
  * Cloud

The endmembers are transformed according to the methodology in Souza et al (2005) in to the Normalized Fraction Degradation Index (NFDI). NFDI is generally designed to highlight areas of degraded or cleared forests. NFDI was designed in the Amazon, and to my knowledge has not been validated elsewhere.  

## Forest classification and characterization

To find degraded or damaged forests, the original state of the forest must first be characterized. Generally, an NFDI value near or at 1 is indication of a forested landscape. However, the magnitude of NFDI will depend on the density of the forest. A training period is used to define the 'general' state of the forest. To account for clouds, sensor noise, and other factors that cause image-to-image variability, a regression model is fit for every pixel for all the observations in the training period. The regression model is made up of the following components:

  * A constant term, representing overall magnitude
  * A trend term, representing interannual variability
  * A sine and cosine term, representing seasonal, or intra-annual variability 
  * A noise term, summarized in the algorithm as the root-mean-squared-error 

