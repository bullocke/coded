# Continuous Degredation Detection (CODED)

CODED is an algorithm developed to monitor for low-magnitude forest disturbances using Landsat data. The algorithm is based upon previous developments in continuous land cover monitoring [(Zhu and Woodcock, 2014)](http://www.sciencedirect.com/science/article/pii/S0034425714000248) and tropical degradation monitoring using spectral unmixing models [(Souza et al., 2013)](http://www.mdpi.com/2072-4292/5/11/5493/html) and is built upon the [Google Earth Engine](https://earthengine.google.com/) processing and data storage system. CODED is designed to create a stratification for sample-based estimation of degraded forests. 

## Repository

An updated repository with Javascript code to run the algorithm can be found [here](https://code.earthengine.google.com/?accept_repo=users/bullocke/coded). Intructions for running the algorithm can be found in the [Javascript folder](https://github.com/bullocke/coded/blob/master/javascript/instructions.MD). An old and non-maintained version using the Python API can be found in the [python folder](https://github.com/bullocke/coded/tree/master/python). The repository also contains two
example files in the folder 'v0.2/examples'. One file contains example time series of different types of degradation events. The second file contains the code for running the algorithm in a region in Acre, Brazil.  

