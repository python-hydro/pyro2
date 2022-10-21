import glob
from pathlib import Path

from setuptools import find_packages, setup

# find all of the "_default" files
defaults = []
for path in Path("pyro").rglob("_default*"):
    defaults.append(str(path.relative_to("./pyro")))

# find all of the problem "_pname.defaults" files
# note: pathlib doesn't work here, because of symlinks
for f in glob.glob("pyro/**/_*.defaults", recursive=True):
    defaults.append(f.replace("pyro/", ""))

# find all of the "inputs" files
inputs = []
for f in glob.glob("pyro/**/inputs*", recursive=True):
    if not f.endswith("inputs.auto"):
        inputs.append(f.replace("pyro/", ""))

benchmarks = ["advection/tests/*.h5",
              "advection_fv4/tests/*.h5",
              "advection_nonuniform/tests/*.h5",
              "advection_rk/tests/*.h5",
              "compressible/tests/*.h5",
              "compressible/tests/*.h5",
              "compressible/tests/*.h5",
              "compressible_fv4/tests/*.h5",
              "compressible_rk/tests/*.h5",
              "compressible_sdc/tests/*.h5",
              "diffusion/tests/*.h5",
              "incompressible/tests/*.h5",
              "lm_atm/tests/*.h5",
              "multigrid/tests/*.h5",
              "swe/tests/*.h5"]

setup(name='pyro-hydro',
      description='A python hydrodynamics code for teaching and prototyping',
      long_description=open('README.md', 'r').read(),
      long_description_content_type='text/markdown',
      url='https://github.com/python-hydro/pyro2',
      license='BSD',
      packages=find_packages(),
      entry_points={
          "console_scripts": [
              "pyro_sim.py = pyro.pyro_sim:main",
          ]
      },
      package_data={"pyro": benchmarks + defaults + inputs},
      install_requires=['numpy', 'numba', 'matplotlib', 'h5py'],
      use_scm_version={"version_scheme": "post-release",
                       "write_to": "pyro/_version.py"},
      setup_requires=["setuptools_scm"],
      zip_safe=False)
