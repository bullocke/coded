#!/usr/bin/env python
# -*- coding: UTF-8 -*-
""" Mosaic images using gdal merge.

Usage: mosaic.py [options] <output> <inpath> <nodata>

  --reverse          reverse input list
"""

import gdal
from docopt import docopt
import glob
import os

if __name__ == '__main__':
    args = docopt(__doc__, version='0.6.2')

output = args['<output>']
path = args['<inpath>']
nodata = int(args['<nodata>'])

file_list = glob.glob(path)

if args['--reverse']:
    file_list.reverse()

files_string = " ".join(file_list)
command = "gdal_merge.py -n {a} -o {b} -of gtiff {c}".format(a=nodata,b=output,c=files_string)

os.system(command)
