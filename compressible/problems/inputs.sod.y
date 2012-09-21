# simple inputs files for the unsplit CTU hydro scheme

[driver]
max_steps = 200
tmax = 0.25

[io]
basename = sod_y_
tplot = 0.001

[mesh]
nx = 10
ny = 128
xmax = .05
ymax = 1.0
yl_boundary = outflow
yr_boundary = outflow

[sod]
direction = y

dens_left = 1.0
dens_right = 0.125

u_left = 0.0
u_right = 0.0

p_left = 1.0
p_right = 0.1
