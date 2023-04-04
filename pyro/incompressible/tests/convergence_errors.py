# This should be run from /pyro (where the output files are located)
import glob
import sys

from pyro.analysis.incomp_converge_error import get_errors


def create_file(filename="convergence_errors.txt"):

    # Parse all converge*.h5 outputs
    # store the last file at each resolution
    fnames = glob.glob("converge*.h5")
    res = {f.split("_")[1] for f in fnames}
    res = sorted(res, key=int)

    simfiles = []
    for r in res:
        fnames = glob.glob(f"converge_{r}_*.h5")
        last = max(int(f.split("_")[-1].split(".")[0]) for f in fnames)
        simfiles.append(f"converge_{r}_{last:04d}.h5")

    # Create the file
    with open("incompressible/tests/" + filename, "w") as f:
        f.write("# convergence of incompressible converge problem\n")
        f.write("# (convergence measured with analysis/incomp_converge_error.py)\n")
        f.write("#\n")
        for r, file in zip(res, simfiles):
            errors = get_errors(file)
            f.write(f"{r.rjust(3)} {errors[0]:.14f} {errors[1]:.14f}\n")


if __name__ == "__main__":
    if len(sys.argv) == 2:
        create_file(sys.argv[1])
    else:
        create_file()
