""" Convert date band to year of first disturbance

Usage: convert_date_band.py [options] <input> <output>

  --dband=<DBAND1>  date band  (default 2)

"""

from docopt import docopt
import gdal
from osgeo import ogr, osr
import os
import numpy as np
import pandas as pd
from postprocess_utils import save_raster_simple

def main(raster, output, dband):

    """ Main function for getting class area of a raster by year """

    # Open raster
    rast_op = gdal.Open(raster)
    date_array1 = rast_op.GetRasterBand(dband).ReadAsArray().astype(np.str)

    # convert to just years
    year_image1 = date_array1.view(np.chararray).ljust(4)
    year_image1 = np.array(year_image1).astype(np.float) 

    save_raster_simple(year_image1, raster, output)


if __name__ == '__main__':
    args = docopt(__doc__, version='0.6.2')

raster=args['<input>']
output=args['<output>']


if args['--dband']:
    dband = int(args['--dband'])
else:
    dband = 2


main(raster, output, dband)


