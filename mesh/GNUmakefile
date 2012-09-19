all: reconstruction_f.so

reconstruction_f.so: reconstruction_f.f90
	@f2py --fcompiler=gnu95 -c reconstruction_f.f90 -m reconstruction_f >> /dev/null


