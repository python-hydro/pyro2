#!/usr/bin/env python

import pyro

results = {}


opts = "driver.verbose=0 vis.dovis=0".split()

# advection
add_err = pyro.doit("advection", "smooth", "inputs.smooth",
  		    other_commands=opts, comp_bench=True)

results["advection"] = add_err


# compressible
comp_err = pyro.doit("compressible", "quad", "inputs.quad",
  		     other_commands=opts, comp_bench=True)

results["compressible"] = comp_err



# diffusion
diff_err = pyro.doit("diffusion", "gaussian", "inputs.gaussian",
  		     other_commands=opts, comp_bench=True)

results["diffusion"] = diff_err


# incompressible
incomp_err = pyro.doit("incompressible", "shear", "inputs.shear",
  		     other_commands=opts, comp_bench=True)

results["incompressible"] = incomp_err


# LM_atm
lm_err = pyro.doit("lm_atm", "bubble", "inputs.bubble",
  		   other_commands=opts, comp_bench=True)

results["lm_atm"] = lm_err


for s, r in results.iteritems():
    if not r == 0:
        print "{:32} failed".format(s)
    else:
        print "{:32} passed".format(s)







