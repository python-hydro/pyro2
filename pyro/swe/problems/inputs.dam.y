# simple inputs files for the unsplit CTU hydro scheme

[driver]
max_steps = 200
tmax = 0.3

[io]
basename = dam_y_
dt_out = 0.05

[mesh]
nx = 10
ny = 128
xmax = .05
ymax = 1.0
ylboundary = outflow
yrboundary = outflow

[dam]
direction = y

h_left = 1.0
h_right = 0.125

u_left = 0.0
u_right = 0.0
