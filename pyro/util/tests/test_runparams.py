import os

import pyro.util.runparams as rp

# test the utilities


def test_is_int():
    assert rp.is_int("1")
    assert not rp.is_int("1.0")
    assert not rp.is_int("a")


def test_is_float():
    assert rp.is_float("1.0")
    assert rp.is_float("1")
    assert not rp.is_float("a")


def test_get_val():
    assert rp._get_val("1.5") == 1.5


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

    def setup_method(self):
        """ this is run before each test """
        self.rp = rp.RuntimeParameters()
        tests_dir = os.path.dirname(os.path.realpath(__file__))
        self.rp.load_params(os.path.join(tests_dir, "test.ini"))

    def teardown_method(self):
        """ this is run after each test """
        self.rp = None

    def test_get_param(self):
        assert self.rp.get_param("test.p1") == "x"
        assert self.rp.get_param("test.p2") == 1.0
        assert self.rp.get_param("test2.p1") == "y"
        assert self.rp.get_param("test3.param") == "this is a test"
        assert self.rp.get_param("test3.i1") == 1

    def test_command_line_params(self):

        param_string = "test.p1=q test3.i1=2"

        self.rp.command_line_params(param_string.split())

        assert self.rp.get_param("test.p1") == "q"
        assert self.rp.get_param("test3.i1") == 2

    def test_failure(self):

        try:
            p3 = self.rp.get_param("test.p3")
        except KeyError:
            p3 = -1

        assert p3 == -1
