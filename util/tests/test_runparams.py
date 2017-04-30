from numpy.testing import assert_array_equal
from nose.tools import assert_equal, with_setup

import util.runparams as rp

# test the utilities

def test_is_int():
    assert_equal(rp.is_int("1"), True)
    assert_equal(rp.is_int("1.0"), False)
    assert_equal(rp.is_int("a"), False)

def test_is_float():
    assert_equal(rp.is_float("1.0"), True)
    assert_equal(rp.is_float("1"), True)
    assert_equal(rp.is_float("a"), False)

def test_get_val():
    assert_equal(rp._get_val("1.5"), 1.5)


# test the runtime parameter class

class TestRunParams(object):

    @classmethod
    def setup_class(cls):
        """ this is run once for each class before any tests """
        pass

    @classmethod
    def teardown_class(cls):
        """ this is run once for each class after all tests """
        pass

    def setup(self):
        """ this is run before each test """
        self.rp = rp.RuntimeParameters()
        self.rp.load_params("util/tests/test.ini")

    def teardown(self):
        """ this is run after each test """
        self.rp = None

    def test_get_param(self):
        assert_equal(self.rp.get_param("test.p1"), "x")
        assert_equal(self.rp.get_param("test.p2"), 1.0)
        assert_equal(self.rp.get_param("test2.p1"), "y")
        assert_equal(self.rp.get_param("test3.param"), "this is a test")
        assert_equal(self.rp.get_param("test3.i1"), 1)

    def test_command_line_params(self):

        param_string="test.p1=q test3.i1=2"
        
        self.rp.command_line_params(param_string.split())

        assert_equal(self.rp.get_param("test.p1"), "q")
        assert_equal(self.rp.get_param("test3.i1"), 2)

