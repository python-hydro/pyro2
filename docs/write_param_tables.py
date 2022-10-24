import glob
import os

from pyro.util import runparams

for f in glob.iglob("../pyro/**/_defaults", recursive=True):
    rp = runparams.RuntimeParameters()
    rp.load_params(f)

    # strip initial "../pyro/"
    pre, name = os.path.split(os.path.relpath(f, "../pyro/"))
    outfile = "source/{}{}.inc".format(pre, name)
    rp.print_sphinx_tables(outfile=outfile)
