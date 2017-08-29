import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid


def setup_axes(myg):
    """ create a grid of axes whose layout depends on the aspect ratio of the
    domain """

    L_x = myg.xmax - myg.xmin
    L_y = myg.ymax - myg.ymin

    f = plt.figure(1)

    cbar_title = False

    if L_x > 2*L_y:
        # we want 4 rows:
        axes = AxesGrid(f, 111,
                        nrows_ncols=(4, 1),
                        share_all=True,
                        cbar_mode="each",
                        cbar_location="top",
                        cbar_pad="10%",
                        cbar_size="25%",
                        axes_pad=(0.25, 0.65),
                        add_all=True, label_mode="L")
        cbar_title = True

    elif L_y > 2*L_x:
        # we want 4 columns:  rho  |U|  p  e
        axes = AxesGrid(f, 111,
                        nrows_ncols=(1, 4),
                        share_all=True,
                        cbar_mode="each",
                        cbar_location="right",
                        cbar_pad="10%",
                        cbar_size="25%",
                        axes_pad=(0.65, 0.25),
                        add_all=True, label_mode="L")

    else:
        # 2x2 grid of plots
        axes = AxesGrid(f, 111,
                        nrows_ncols=(2, 2),
                        share_all=True,
                        cbar_mode="each",
                        cbar_location="right",
                        cbar_pad="2%",
                        axes_pad=(0.65, 0.25),
                        add_all=True, label_mode="L")

    return f, axes, cbar_title
