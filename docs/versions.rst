
Versions
========

*Version history of CODED*

There are two versions of CODED that will produce similar but not identical results, referred to here as 'Version 0' and 'Version 1'. The original implementation, Version 0, was written entirely in Javascript using the Google Earth Engine Javascript API. This version is what is described in the `paper <https://doi.org/10.1016/j.rse.2018.11.011>`_. Version 1 was created to overcome technical limitations of Version 0. Put simply, Version 0 was not efficient enough to consistently run over large areas. Version 1 makes use of the `GEE CCDC implementation <https://developers.google.com/earth-engine/api_docs#eealgorithmstemporalsegmentationccdc>`_ for change monitoring, adapted to create the algorithm described in the CODED paper. 



*Version 0 tutorial*

I do not recommend using Version 0, but it is still available if users wish to use it. 

:doc:`module_parameters`
:doc:`running`
:doc:`dataFuncs`
:doc:`acre`

