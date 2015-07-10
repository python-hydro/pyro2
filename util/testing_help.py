from __future__ import print_function

from nose.tools.nontrivial import with_setup

# note, we need to run nosetest with -s to see output

# cool recipe from http://stackoverflow.com/questions/16710061/get-name-of-current-test-in-setup-using-nose
def with_named_setup(setup=None, teardown=None):
    def wrap(f):
        return with_setup(
            lambda: setup(f.__name__) if (setup is not None) else None, 
            lambda: teardown(f.__name__) if (teardown is not None) else None)(f)
    return wrap

def setup_func(f):
    print(f)

def teardown_func(f):
    pass
