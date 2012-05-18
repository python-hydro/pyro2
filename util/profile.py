"""
  A very simple profiling class.  Define some timers and methods
  to start and stop them.  Nesting of timers is tracked so we can
  pretty print the profiling information.

  # define a timer object, labeled 'my timer'
  a = timer('my timer')

  This will add 'my timer' to the list of keys in the 'my timer'
  dictionary.  Subsequent calls to the timer class constructor
  will have no effect.

  # start timing the 'my timer' block of code
  a.begin()
  
  ... do stuff here ...

  # end the timing of the 'my timer' block of code
  a.end()

  for best results, the block of code timed should be large
  enough to offset the overhead of the timer class method
  calls.

  Multiple timers can be instanciated and nested.  The stackCount
  global parameter keeps count of the level of nesting, and the
  timerNesting data structure stores the nesting level for each
  defined timer.

  timeReport() is called at the end to print out a summary of the
  timing.

  At present, no enforcement is done to ensure proper nesting.

"""
  

import time 

timers = {}

# keep basic count of how nested we are in the timers, so we can do some
# pretty printing.
stackCount = 0

timerNesting = {}
timerOrder = []

class timer:



    def __init__ (self, name):
        global timers, stackCount, timerNesting, timerOrder
        
        self.name = name

        keys = timers.keys()

        if name not in keys:
            timers[name] = 0.0
            self.startTime = 0.0
            timerOrder.append(name)
            timerNesting[name] = stackCount
            

    def begin(self):
        global stackCount
        
        self.startTime = time.time()
        stackCount += 1

        
    def end(self):
        global timers, stackCount

        elapsedTime = time.time() - self.startTime
        timers[self.name] += elapsedTime
        
        stackCount -= 1
        

def timeReport():
    global timers, timerOrder, timerNesting

    spacing = '   '
    for key in timerOrder:
        print timerNesting[key]*spacing + key + ': ', timers[key]



if __name__ == "__main__":    
    a = timer('1')
    a.begin()
    time.sleep(10.)
    a.end()
    
    b = timer('2')
    b.begin()
    time.sleep(5.)
    
    c = timer('3')
    c.begin()
    
    time.sleep(20.)
    
    b.end()
    c.end()
    
    timeReport()

