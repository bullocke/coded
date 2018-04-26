#!/usr/bin/env python
# -*- coding: UTF-8 -*-
""" Fix edges of strata map

Usage: fix_edges.py <image1> <image2> <output>

"""

import gdal
from docopt import docopt
import numpy as np
from postprocess_utils import save_raster

if __name__ == '__main__':
    args = docopt(__doc__, version='0.6.2')

image1=args['<image1>']
image2=args['<image2>']
output = args['<output>']

im_op1 = gdal.Open(image1)
im_op2 = gdal.Open(image2)


im1_b1 = im_op1.GetRasterBand(1).ReadAsArray()
im1_b2 = im_op1.GetRasterBand(2).ReadAsArray()
im1_b3 = im_op1.GetRasterBand(3).ReadAsArray()

im2_b1 = im_op2.GetRasterBand(1).ReadAsArray()
im2_b2 = im_op2.GetRasterBand(2).ReadAsArray()
im2_b3 = im_op2.GetRasterBand(3).ReadAsArray()


dif = np.logical_and(im1_b1 <= 2, im2_b1 >= 3)

im1_b1[dif] = im2_b1[dif]
im1_b2[dif] = im2_b2[dif]
im1_b3[dif] = im2_b3[dif]

ar_out = np.stack((im1_b1, im1_b2, im1_b3))

save_raster(ar_out, image1, output)

