import matplotlib.pyplot as plt
import numpy as np

logo_text = """
 XXX  X  X X XX  XX  
 X  X X  X XX   X  X 
 X  X  X X X    X  X 
 XXX   XXX X     XX  
 X       X           
 X      X            
"""


class LogoGrid:
    def __init__(self, mask):

        logo_lines = mask.split("\n")
        self.nx = len(logo_lines[1])
        self.ny = len(logo_lines)

        self.xl = np.arange(self.nx)
        self.xr = np.arange(self.nx) + 1.0
        self.x = np.arange(self.nx) + 0.5

        self.yl = np.flip(np.arange(self.ny))
        self.yr = np.flip(np.arange(self.ny) + 1.0)
        self.y = np.flip(np.arange(self.ny) + 0.5)

        self.dx = 1
        self.dy = 1

        self.xmin = self.xl.min()
        self.xmax = self.xr.max()

        self.ymin = self.yl.min()
        self.ymax = self.yr.max()

        #print("xmin, xmax", self.xmin, self.xmax)
        #print("ymin, ymax", self.ymin, self.ymax)

        self.logo = np.zeros((self.nx, self.ny))

        for i in range(1, self.nx):
            for j in range(1, self.ny-1):
                if logo_lines[j][i] == "X":
                    self.logo[i, j] = 1.0

    def draw_grid(self):
        # vertical lines
        for i in range(0, self.nx):
            plt.plot([self.xl[i], self.xl[i]], [self.ymin-0.5*self.dy, self.ymax+0.5*self.dy],
                     color="C0", lw=3)

        plt.plot([self.xr[self.nx-1], self.xr[self.nx-1]],
                 [self.ymin-0.5*self.dy, self.ymax+0.5*self.dy],
                 color="C0", lw=3)

        # horizontal lines
        for j in range(0, self.ny):
            plt.plot([self.xmin-0.5*self.dx, self.xmax+0.5*self.dx], [self.yl[j], self.yl[j]],
                     color="C0", lw=3)

        plt.plot([self.xmin-0.5*self.dx, self.xmax+0.5*self.dx],
                 [self.yr[0], self.yr[0]],
                 color="C0", lw=3)

    def fill_in_logo(self):
        for j in range(self.ny):
            for i in range(self.nx):
                if self.logo[i, j] == 0.0:
                    continue
                plt.fill([self.xl[i], self.xl[i], self.xr[i], self.xr[i], self.xl[i]],
                         [self.yl[j], self.yr[j], self.yr[j], self.yl[j], self.yl[j]],
                         color="k", alpha=0.8)

    def fill_in_background(self):
        xx = np.linspace(self.xmin, self.xmax, 200, endpoint=True)
        yy = np.linspace(self.ymin, self.ymax, 200, endpoint=True)

        x, y = np.meshgrid(xx, yy)

        # a funky function
        ff = self._func(x, y)

        plt.imshow(ff, extent=[self.xmin, self.xmax, self.ymin, self.ymax], alpha=0.25, cmap="plasma")

    def _func(self, x, y):
        return (1 + 0.5*(1.0 + np.tanh(y - 0.5*(self.ymin + self.ymax)))) * np.cos(3*np.pi*x/(self.xmax-self.xmin))


lg = LogoGrid(logo_text)
lg.draw_grid()
lg.fill_in_logo()
lg.fill_in_background()

#plt.subplots_adjust(0,0,1,1,0,0)

ax = plt.gca()
ax.set_axis_off()
ax.set_aspect("equal", "datalim")
ax.xaxis.set_major_locator(plt.NullLocator())
ax.yaxis.set_major_locator(plt.NullLocator())

fig = plt.gcf()
fig.set_size_inches(int(lg.nx+1), int(lg.ny+1))

plt.margins(0.0)
#plt.tight_layout()
plt.savefig("pyro_logo.svg", bbox_inches="tight", pad_inches=0)
