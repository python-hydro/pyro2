# simple inputs files for the unsplit CTU hydro scheme

[driver]
max_steps = 200
tmax = 0.25

[io]
basename = sod_x
tplot = 0.001

[mesh]
nx = 128
ny = 10
xmax = 1.0
ymax = .05
xl_boundary = outflow
xr_boundary = outflow

[sod]
direction = x

dens_left = 1.0
dens_right = 0.125

u_left = 0.0
u_right = 0.0

p_left = 1.0
p_right = 0.1
