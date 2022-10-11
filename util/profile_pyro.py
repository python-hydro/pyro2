"""
A very simple profiling class, to use to determine where
most of the time is spent in a code.  This supports nested
timers and outputs a report at the end.

Warning: At present, no enforcement is done to ensure proper
nesting.

"""


import time


class TimerCollection:
    """A timer collection---this manages the timers and has methods to
    start and stop them.  Nesting of timers is tracked so we can
    pretty print the profiling information.

    To define a timer::

       tc = TimerCollection()
       a = tc.timer('my timer')

    This will add 'my timer' to the list of Timers managed by the
    TimerCollection.  Subsequent calls to timer() will return the same
    Timer object.

    To start the timer::

       a.begin()

    and to end it::

       a.end()

    For best results, the block of code timed should be large enough
    to offset the overhead of the timer class method calls.

    tc.report() prints out a summary of the timing.
    """

    def __init__(self):

        """
        Initialize the collection of timers
        """
        self.timers = []

    def timer(self, name):
        """
        Create a timer with the given name.  If one with that name already
        exists, then we return that timer.

        Parameters
        ----------
        name : str
            Name of the timer

        Returns
        -------
        out : Timer object
            A timer object corresponding to the name.

        """

        # check if any existing timer has this name, if so, return that
        # object
        for t in self.timers:
            if t.name == name:
                return t

        # if we're here, we didn't find one, so create a new one

        # find out how nested we are (the stack count), for pretty printing
        stack_count = 0
        for t in self.timers:
            if t.is_running:
                stack_count += 1

        t_new = Timer(name, stack_count=stack_count)

        self.timers.append(t_new)

        return t_new

    def report(self):
        """
        Generate a timing summary report
        """

        spacing = '   '
        for t in self.timers:
            print(t.stack_count*spacing + t.name + ': ', t.elapsed_time)


class Timer:
    """A single timer -- this simply stores the accumulated time for
    a single named region"""

    def __init__(self, name, stack_count=0):
        """
        Initialize a timer with the given name.

        Parameters
        ----------
        name : str
            The name of the timer
        stack_count : int, optional
            The depth of the timer (i.e. how many timers is this nested
            in).  This is used for printing purposes.

        """
        self.name = name
        self.stack_count = stack_count
        self.is_running = False

        self.start_time = 0
        self.elapsed_time = 0

    def begin(self):
        """
        Start timing
        """
        self.start_time = time.time()
        self.is_running = True

    def end(self):
        """
        Stop timing.  This does not destroy the timer, it simply
        stops it from counting time.
        """
        elapsed_time = time.time() - self.start_time
        self.elapsed_time += elapsed_time
        self.is_running = False


if __name__ == "__main__":
    tc = TimerCollection()

    a = tc.timer('a')
    a.begin()
    time.sleep(10.)
    a.end()

    b = tc.timer('b')
    b.begin()
    time.sleep(5.)

    c = tc.timer('c')
    c.begin()
    time.sleep(10.)
    c.end()

    c = tc.timer('c')
    c.begin()
    time.sleep(10.)
    c.end()

    b.end()

    tc.report()
