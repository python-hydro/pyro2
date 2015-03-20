#!/bin/sh

# advection
./pyro.py --compare_benchmark advection smooth inputs.smooth driver.verbose=0 vis.dovis=0

# compressible
./pyro.py --compare_benchmark compressible quad inputs.quad driver.verbose=0 vis.dovis=0

# diffusion
./pyro.py --compare_benchmark diffusion gaussian inputs.gaussian driver.verbose=0 vis.dovis=0

# incompressible
./pyro.py --compare_benchmark incompressible shear inputs.shear driver.verbose=0 vis.dovis=0

# LM_atm
./pyro.py --compare_benchmark lm_atm bubble inputs.bubble driver.verbose=0 vis.dovis=0


