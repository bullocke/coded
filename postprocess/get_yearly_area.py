""" Get yearly mapped class areas

Usage: get_yearly_area.py [options] <input> <output>

  --km=<KM>          output area in km instead of hectares
  --sband=<sBAND>     strata band (default 1)
  --dband1=<DBAND1>  date band 1 (default 2)
  --dband2=<DBAND2>  date band 2 (default 3)

"""

from docopt import docopt
import gdal
from osgeo import ogr, osr
import os
import numpy as np
import pandas as pd

def main(raster, years, output, dband1, dband2, sband):

    """ Main function for getting class area of a raster by year """

    # Open raster
    rast_op = gdal.Open(raster)
    date_array1 = rast_op.GetRasterBand(dband1).ReadAsArray().astype(np.str)
    date_array2 = rast_op.GetRasterBand(dband2).ReadAsArray().astype(np.str)

    # convert to just years
    year_image1 = date_array1.view(np.chararray).ljust(4)
    year_image1 = np.array(year_image1).astype(np.float) 

    year_image2 = date_array2.view(np.chararray).ljust(4)
    year_image2 = np.array(year_image2).astype(np.float) 

    # open strata band
    rast_op = gdal.Open(raster)
    strata_array = rast_op.GetRasterBand(sband).ReadAsArray()
    stratas = np.unique(strata_array)
    stratas = stratas[stratas != 0]

    multiplier = .09 #hectares

    # Deg
    out_df = pd.DataFrame()
    out_name = output + '_strata_deg.csv'
    for year in years:
        just_deg = [np.logical_and(year_image1 == int(year),
				   strata_array == 4).sum() * multiplier]
        deg2deg = [np.logical_and(year_image2 == int(year),
				   strata_array == 4).sum() * multiplier]
        deg2def =  [np.logical_and(year_image1 == int(year),
				   strata_array == 5).sum() * multiplier]

        out_df[year] = [just_deg[0] + deg2def[0] + deg2deg[0]]

    out_df.to_csv(out_name)


    # Def
    out_df = pd.DataFrame()
    out_name = output + '_strata_def.csv'
    for year in years:
        just_def = [np.logical_and(year_image1 == int(year),
				   strata_array == 3).sum() * multiplier]
        deg2def =  [np.logical_and(year_image2 == int(year),
				   strata_array == 5).sum() * multiplier]

        out_df[year] = [just_def[0] + deg2def[0]]

    out_df.to_csv(out_name)

if __name__ == '__main__':
    args = docopt(__doc__, version='0.6.2')

raster=args['<input>']
output=args['<output>']
#years=args['<years>'].split(' ')
years=range(1990,2014)


if args['--sband']:
    sband = int(args['--sband'])
else:
    sband = 1

if args['--dband1']:
    dband1 = int(args['--dband1'])
else:
    dband1 = 2

if args['--dband2']:
    dband2 = int(args['--dband2'])
else:
    dband2 = 3

main(raster, years, output, dband1, dband2, sband)


