all: interface_f.so

interface_f.so: interface_f.f90
	@f2py --fcompiler=gnu95 -c interface_f.f90 -m interface_f >> /dev/null


