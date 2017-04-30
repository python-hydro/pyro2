import compressible.eos as eos

from nose.tools import assert_equal

def test_eos_consistency():

    dens = 1.0
    eint = 1.0
    gamma = 1.4

    p = eos.pres(gamma, dens, eint)

    dens_eos = eos.dens(gamma, p, eint)

    assert_equal(dens, dens_eos)

    rhoe_eos = eos.rhoe(gamma, p)

    assert_equal(dens*eint, rhoe_eos)
