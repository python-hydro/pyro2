import glob
import os

from util import runparams

pfiles = ["../_defaults"]
for path, dirs, files in os.walk("../"):
    for d in dirs:
        for f in glob.iglob(os.path.join(path, d, "_defaults")):
            pfiles.append(f)

for f in pfiles:
    rp = runparams.RuntimeParameters()
    rp.load_params(f)

    pre, name = os.path.split(f)
    outfile = "source/{}{}.inc".format(pre.replace(".", ""), name)
    rp.print_sphinx_tables(outfile=outfile.format(os.path.basename(f)))


