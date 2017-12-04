""" Get yearly mapped class areas

Usage: get_yearly_area.py [options] <years> <input> <output>

  --km=<KM>      output area in km instead of hectares
  --dband=<BAND>  date band (default 1)
  --sband=<BAND>  strata band (default 1)

"""

from docopt import docopt
import gdal
from osgeo import ogr, osr
import os
import numpy as np
import pandas as pd

def main(raster, years, output, dband, sband):

    """ Main function for getting class area of a raster by year """

    # Open raster
    rast_op = gdal.Open(raster)
    date_array = rast_op.GetRasterBand(dband).ReadAsArray().astype(np.str)

    # convert to just years
    year_image = date_array.view(np.chararray).ljust(4)
    year_image = np.array(year_image).astype(np.float) 

    # open strata band
    rast_op = gdal.Open(raster)
    strata_array = rast_op.GetRasterBand(sband).ReadAsArray()
    stratas = np.unique(strata_array)
    stratas = stratas[stratas != 0]

    multiplier = .09 #hectares

    for i in stratas:
        out_df = pd.DataFrame()
	out_name = output + '_strata_' + str(i) + '.csv'
        for year in years:
	     out_df[year] = [np.logical_and(year_image == int(year), 
					     strata_array == i).sum() * multiplier]
        out_df.to_csv(out_name)


if __name__ == '__main__':
    args = docopt(__doc__, version='0.6.2')

raster=args['<input>']
output=args['<output>']
years=args['<years>'].split(' ')

if args['--dband']:
    dband = int(args['--dband'])
else:
    dband = 1

if args['--sband']:
    sband = int(args['--sband'])
else:
    sband = 4


main(raster, years, output, dband,sband)


