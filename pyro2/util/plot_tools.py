"""Some basic support routines for configuring the plots during
runtime visualization"""

import math

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid


def setup_axes(myg, num):
    """ create a grid of axes whose layout depends on the aspect ratio of the
    domain """

    L_x = myg.xmax - myg.xmin
    L_y = myg.ymax - myg.ymin

    f = plt.figure(1)

    cbar_title = False

    if L_x > 2*L_y:
        # we want num rows:
        axes = ImageGrid(f, 111,
                        nrows_ncols=(num, 1),
                        share_all=True,
                        cbar_mode="each",
                        cbar_location="top",
                        cbar_pad="10%",
                        cbar_size="25%",
                        axes_pad=(0.25, 0.65),
                        label_mode="L")
        cbar_title = True

    elif L_y > 2*L_x:
        # we want num columns:  rho  |U|  p  e
        axes = ImageGrid(f, 111,
                        nrows_ncols=(1, num),
                        share_all=True,
                        cbar_mode="each",
                        cbar_location="right",
                        cbar_pad="10%",
                        cbar_size="25%",
                        axes_pad=(0.65, 0.25),
                        label_mode="L")

    else:
        # 2-d grid of plots
        ny = int(math.sqrt(num))
        nx = num//ny

        axes = ImageGrid(f, 111,
                        nrows_ncols=(nx, ny),
                        share_all=True,
                        cbar_mode="each",
                        cbar_location="right",
                        cbar_pad="2%",
                        axes_pad=(0.65, 0.25),
                        label_mode="L")

    return f, axes, cbar_title
