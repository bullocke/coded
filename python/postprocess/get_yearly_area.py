""" Get yearly mapped class areas

Usage: get_yearly_area.py [options] <input> <output>

"""

from docopt import docopt
import gdal
from osgeo import ogr, osr
import os
import numpy as np
import pandas as pd

def main(raster, years, output):

    """ Main function for getting class area of a raster by year """

    # Open raster
    rast_op = gdal.Open(raster)
    array = rast_op.ReadAsArray()

    # convert to just years

    multiplier = .09 #hectares


    # Deg
    out_deg = pd.DataFrame()
    for year in years:
        out_deg[str(year)] = [0]

    out_name = output + '_strata_deg.csv'
    for year in years:
        count_year = len(array[0:3,:,:][array[0:3,:,:] == year]) * multiplier
        out_deg[str(year)] += count_year

    out_deg.to_csv(out_name)

    # Deg
    out_def = pd.DataFrame()
    for year in years:
        out_def[str(year)] = [0]

    out_name = output + '_strata_def.csv'
    for year in years:
        count_year = len(array[3:6,:,:][array[3:6,:,:] == year]) * multiplier
        out_def[str(year)] += count_year

    out_def.to_csv(out_name)

if __name__ == '__main__':
    args = docopt(__doc__, version='0.6.2')

raster=args['<input>']
output=args['<output>']
years=range(1990,2014)


main(raster, years, output)


