
Versions
========

*Version history of CODED*

There are now three versions of CODED that will produce similar but not identical results: 

0. Version 0 is the original beta implementation, written entirely in Javascript using the Google Earth Engine API. This version is what is described in the `paper <https://doi.org/10.1016/j.rse.2018.11.011>`_. 
1. Version 1 was created to overcome technical limitations of Version 0. Put simply, Version 0 was not efficient enough to consistently run over large areas. Version 1 makes use of the `GEE CCDC implementation <https://developers.google.com/earth-engine/api_docs#eealgorithmstemporalsegmentationccdc>`_ for change monitoring, adapted to create the algorithm described in the CODED paper. Version 1 is what is implemented in the Forest Disturbance Mapping GUI.2. Version 2 is nearly identical to Version 1 but makes use of the CODED API. Version 2 is the most recent version and will be the only version modified moving forward. This version is what is applied in the Javascript tutorial. The main difference between Version 1 and 2 is that version 2 has more options for output layers. 
