# simple inputs files for the unsplit CTU hydro scheme

[driver]
max_steps = 200
tmax = 0.3

[swe]
limiter = 1
grav = 1.0
riemann = Roe

[io]
basename = dam_x_
dt_out = 0.05

[mesh]
nx = 128
ny = 10
xmax = 1.0
ymax = .05
xlboundary = outflow
xrboundary = outflow

[dam]
direction = x

h_left = 1.0
h_right = 0.125

u_left = 0.0
u_right = 0.0
