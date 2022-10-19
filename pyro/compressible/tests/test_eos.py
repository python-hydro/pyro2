import pyro.compressible.eos as eos


def test_eos_consistency():

    dens = 1.0
    eint = 1.0
    gamma = 1.4

    p = eos.pres(gamma, dens, eint)

    dens_eos = eos.dens(gamma, p, eint)

    assert dens == dens_eos

    rhoe_eos = eos.rhoe(gamma, p)

    assert dens*eint == rhoe_eos
