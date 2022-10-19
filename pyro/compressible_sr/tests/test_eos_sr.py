import pyro.compressible_sr.eos as eos


def test_eos_consistency():

    dens = 1.0
    eint = 1.0
    gamma = 1.4

    p = eos.pres(gamma, dens, eint)

    dens_eos = eos.dens(gamma, p, eint)

    assert dens == dens_eos

    rhoe_eos = eos.rhoe(gamma, p)

    assert dens*eint == rhoe_eos

    h = eos.h_from_eps(gamma, eint)

    assert (1 + gamma*eint) == h

    rhoh = eos.rhoh_from_rho_p(gamma, dens, p)

    assert dens*h == rhoh
