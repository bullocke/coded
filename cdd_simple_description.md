# Simple Continuous Degradation Detection

Simple-CDD is a simplified version of the full CDD algorithm that is designed to be more computationally efficient. 
## Pre-processing

All available Landsat data is used a converted to surface reflectance using the standard LEDAPS surface reflectance product. The data is first filtered for clouds using two algorithms: 

  * CFmask product
  * Google Earth Engine simple cloud score

The data are then converted through linear spectral unmixing to represent fractions of spectral endmembers developed in [Souza et al., 2005](http://www.sciencedirect.com/science/article/pii/S0034425705002385), in addition to a self-developed cloud endmember. The 5 endmembers are:

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

These regression components are used to differentiate a forest or forested grassland (including cerrado) from other land covers. The following chart shows how the land covers differ based on regression RMSE and magnitude based on 1250 training locations across the Amazon basin: 

![alt text](https://raw.githubusercontent.com/bullocke/ge-cdd/master/images/NFDI_landcover_classification.jpg)

The simple version of the algorithm (Simple-CDD), also excludes areas with high trend components to filter out actively changing landscapes (such as regrowing forests). The model is then refit over the training period without the trend term. The reason for removing the trend term is to be able to predict into the future without triggering a change when the gradual change process ends (such as the saturation, or natural halting of regrowth). The full CDD algorithm refits the model every year, accounting for these changes. However, the computational intensity of the regression makes the process significantly slower. 

An important step in this process is not just forest classification, but forest characterization. By calculating the training NFDI regression magnitude, change in NFDI can then be calculated relative to original condition. In this manner, degradation is defined as its relation to original state, not just the NFDI at the current time. A good example of this distinction is in forested cerrado, which exists naturally in a state of non-continuous canopy cover. The cerrado will naturally have a lower NFDI than a closed-canopy forest, but that does not mean it is degraded. This difference alludes to the difficulty in classifying a degraded forest based on a single image alone. 

To see the difference in forest characterization with canopy cover, see the difference in NFDI between a dense congruent canopy in Rondôni (top), and a thinner forested cerrado in Mato Gross (bottom):

![alt text](https://raw.githubusercontent.com/bullocke/ge-cdd/master/images/DenseForest_both2.jpg)
![alt text](https://raw.githubusercontent.com/bullocke/ge-cdd/master/images/ThinForest_both2.jpg)

## Change detection  

## NFDI Prediction 
