all: LM_atm_interface_f.so

LM_atm_interface_f.so: LM_atm_interface_f.f90
	@f2py --fcompiler=gnu95 -c LM_atm_interface_f.f90 -m LM_atm_interface_f >> /dev/null


