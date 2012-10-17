# simple inputs files for the unsplit CTU hydro scheme

[driver]
max_steps = 200
tmax = 0.2

[compressible]
limiter = 1

[io]
basename = sod_x_
dt_out = 0.05

[mesh]
nx = 128
ny = 10
xmax = 1.0
ymax = .05
xlboundary = outflow
xrboundary = outflow

[sod]
direction = x

dens_left = 1.0
dens_right = 0.125

u_left = 0.0
u_right = 0.0

p_left = 1.0
p_right = 0.1
