import sys

# inspiration from                                                                          
# http://stackoverflow.com/questions/287871/print-in-terminal-with-colors-using-python      
# which in-turn cites the blender build scripts                                             
class termColors:
    WARNING = '\033[33m'
    SUCCESS = '\033[32m'
    FAIL = '\033[31m'
    BOLD = '\033[1m'
    ENDC = '\033[0m'


def fail(str):
    new_str = termColors.FAIL + str + termColors.ENDC
    print new_str

    # only exit if we are not running in interactive mode.  sys.ps1 is
    # only defined in interactive mode.
    if hasattr(sys, 'ps1'):
        return
    else:
        sys.exit()

def warning(str):
    new_str = termColors.WARNING + str + termColors.ENDC
    print new_str

def success(str):
    new_str = termColors.SUCCESS + str + termColors.ENDC
    print new_str

def bold(str):
    new_str = termColors.BOLD + str + termColors.ENDC
    print new_str

