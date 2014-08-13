all: incomp_interface_f.so

incomp_interface_f.so: incomp_interface_f.f90
	@f2py --fcompiler=gnu95 -c incomp_interface_f.f90 -m incomp_interface_f >> /dev/null


