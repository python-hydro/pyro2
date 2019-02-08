# this works for python 2 or 3 directly.  To build, do:
#
# python setup.py build_ext
#
# Note the setup.cfg directs the build to be done in-place.
from setuptools import setup, find_packages

requirements_file = open("requirements.txt", "r")
requirements = requirements_file.readlines()

# This call to setup() does all the work
setup(
    name="pyro2",
    version="1.0.0",
    description="A simple python-based tutorial on computational methods for hydrodynamics",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/python-hydro/pyro2",
    author="pyro development team",
    author_email="office@realpython.com",
    license="BSD",
    classifiers=[
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
    packages=find_packages(exclude=["__pycache__", "*.__pycache__", "*.__pycache__.*"]),
    include_package_data=True,
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "pyro=pyro.pyro:__main__",
        ]
    },
)
