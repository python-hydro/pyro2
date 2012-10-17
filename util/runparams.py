"""
basic syntax of the parameter file is:

# simple parameter file

[driver]
nsteps = 100         ; comment
max_time = 0.25

[riemann]
tol = 1.e-10
max_iter = 10

[io]
basename = myfile_


The recommended way to use this is for the code to have a master list
of parameters and their defaults (e.g. _defaults), and then the
user can override these defaults at runtime through an inputs file.
These two files have the same format.

The calling sequence would then be:

  runparams.LoadParams("_defaults")
  runparams.LoadParams("inputs")

The parser will determine what datatype the parameter is (string,
integer, float), and store it in a global dictionary (globalParams).
If a parameter that already exists is encountered a second time (e.g.,
there is a default value in _defaults and the user specifies a new
value in inputs), then the second instance replaces the first.

Runtime parameters can then be accessed via any module through the
getParam method:

  tol = runparams.getParam('riemann.tol')

An earlier version of this was based on the Python Cookbook, 4.11, but
we not longer use the ConfigParser module, instead roll our own regex.

If the optional flag noNew=1 is set, then the LoadParams function will
not define any new parameters, but only overwrite existing ones.  This
is useful for reading in an inputs file that overrides previously read
default values.

"""

import string
import re
from util import msg

# we will keep track of the parameters and their comments globally
globalParams = {}
globalParamComments = {}

# for debugging -- keep track of which parameters were actually looked-
# up
usedParams = []

# some utility functions to automagically determine what the data
# types are
def isInt(string):
    """ is the given string an interger? """
    try: int(string)
    except ValueError: return 0
    else: return 1


def isFloat(string):
    """ is the given string a float? """
    try: float(string)
    except ValueError: return 0
    else: return 1


def LoadParams(file, noNew=0):
    """
    reads lines from file and makes dictionary pairs from the data
    to store in globalParams.
    """
    global globalParams

    # check to see whether the file exists
    try: f = open(file, 'r')
    except IOError:
        msg.fail("ERROR: parameter file does not exist: %s" % (file))


    # we could use the ConfigParser, but we actually want to have
    # our configuration files be self-documenting, of the format
    # key = value     ; comment
    sec = re.compile(r'^\[(.*)\]')
    eq = re.compile(r'^([^=#]+)=([^;]+);{0,1}(.*)')

    for line in f.readlines():

        if sec.search(line): 
            lbracket, section, rbracket = sec.split(line)
            section = string.lower(section.strip())
            
        elif eq.search(line):
            left, item, value, comment, right = eq.split(line) 		
            item = string.lower(item.strip())

            # define the key
            key = section + "." + item
            
            # if we have noNew = 1, then we only want to override existing
            # key/values
            if (noNew):
                if (not key in globalParams.keys()):
                    msg.warning("warning, key: %s not defined" % (key))
                    continue

            # check in turn whether this is an interger, float, or string
            if (isInt(value)):
                globalParams[key] = int(value)
            elif (isFloat(value)):
                globalParams[key] = float(value)
            else:
                globalParams[key] = value.strip()

            # if the comment already exists (i.e. from reading in _defaults)
            # and we are just resetting the value of the parameter (i.e.
            # from reading in inputs), then we don't want to destroy the
            # comment
            if comment.strip() == "":
                try:
                    comment = globalParamComments[key]
                except KeyError:
                    comment = ""
                    
            globalParamComments[key] = comment.strip()


def CommandLineParams(cmdStrings):
    """
    finds dictionary pairs from a string that came from the
    commandline.  Stores the parameters in globalParams only if they 
    already exist.
    """
    global globalParams


    # we expect things in the string in the form:
    #  ["sec.opt=value",  "sec.opt=value"]
    # with each opt an element in the list

    for item in cmdStrings:

        # break it apart
        key, value = item.split("=")
            
        # we only want to override existing keys/values
        if (not key in globalParams.keys()):
            msg.warning("warning, key: %s not defined" % (key))
            continue

        # check in turn whether this is an interger, float, or string
        if (isInt(value)):
            globalParams[key] = int(value)
        elif (isFloat(value)):
            globalParams[key] = float(value)
        else:
            globalParams[key] = value.strip()

    
def getParam(key):
    """
    returns the value of the runtime parameter corresponding to the
    input key
    """
    if globalParams == {}:
        msg.warning("WARNING: runtime parameters not yet initialized")
        LoadParams("_defaults")

    # debugging
    if not key in usedParams:
        usedParams.append(key)
        
    if key in globalParams.keys():
        return globalParams[key]
    else:
        msg.fail("ERROR: runtime parameter %s not found" % (key))
        

def printUnusedParams():
    """
    print out the list of parameters that were defined by never used
    """
    for key in globalParams.keys():
        if not key in usedParams:
            msg.warning("parameter %s never used" % (key))
    

def PrintAllParams():
    keys = globalParams.keys()
    keys.sort()

    for key in keys:
        print key, "=", globalParams[key]

    print " "
    

def PrintParamFile():
    keys = globalParams.keys()
    keys.sort()

    try: f = open('inputs.auto', 'w')
    except IOError:
        msg.fail("ERROR: unable to open inputs.auto")


    f.write('# automagically generated parameter file\n')
    
    currentSection = " "

    for key in keys:
        parts = string.split(key, '.')
        section = parts[0]
        option = parts[1]

        if (section != currentSection):
            currentSection = section
            f.write('\n')
            f.write('[' + section + ']\n')

        if (isinstance(globalParams[key], int)):
            value = '%d' % globalParams[key]
        elif (isinstance(globalParams[key], float)):
            value = '%f' % globalParams[key]
        else:
            value = globalParams[key]

        
        if (globalParamComments[key] != ''):
            f.write(option + ' = ' + value + '       ; ' + globalParamComments[key] + '\n')
        else:
            f.write(option + ' = ' + value + '\n')

    f.close()
     
    
if __name__== "__main__":
    LoadParams("inputs.test")
    PrintParamFile()



    



