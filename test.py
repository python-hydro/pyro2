#!/usr/bin/env python

import pyro

opts = "driver.verbose=0 vis.dovis=0".split()

# advection
add_err = pyro.doit("advection", "smooth", "inputs.smooth",
  		    other_commands=opts, comp_bench=True)
 
# compressible
comp_err = pyro.doit("compressible", "quad", "inputs.quad",
  		     other_commands=opts, comp_bench=True)

# diffusion
diff_err = pyro.doit("diffusion", "gaussian", "inputs.gaussian",
  		     other_commands=opts, comp_bench=True)

# incompressible
incomp_err = pyro.doit("incompressible", "shear", "inputs.shear",
  		     other_commands=opts, comp_bench=True)

# LM_atm
lm_err = pyro.doit("lm_atm", "bubble", "inputs.bubble",
  		   other_commands=opts, comp_bench=True)


if not add_err == 0: print "advection problem failed"
if not comp_err == 0: print "compressible problem failed"
if not diff_err == 0: print "diffusion problem failed"
if not incomp_err == 0: print "incompressible problem failed"
if not lm_err == 0: print "lm_atm problem failed"


