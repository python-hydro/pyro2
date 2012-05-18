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

"""

import string, sys, re

# we will keep track of the parameters and their comments globally
globalParams = {}
globalParamComments = {}

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


def LoadParams(file):
    """
    returns a dictionary with keys of the form
    <section>.<option> and the corresponding values
    """
    global globalParams

    # check to see whether the file exists
    try: f = open(file, 'r')
    except IOError:
        print 'ERROR: parameter file does not exist: ', file
        sys.exit()
    else:
        f.close()

    # we could use the ConfigParser, but we actually want to have
    # our configuration files be self-documenting, of the format
    # key = value     ; comment
    sec = re.compile(r'^\[(.*)\]')
    eq = re.compile(r'^([^=#]+)=([^;]+);{0,1}(.*)')

    for line in open(file, "r").readlines():

        if sec.search(line): 
            lbracket, section, rbracket = sec.split(line)
            section = string.lower(section.strip())
            
        elif eq.search(line):
            left, item, value, comment, right = eq.split(line) 		
            item = string.lower(item.strip())
            
            # check in turn whether this is an interger, float, or string
            if (isInt(value)):
                globalParams[section + "." + item] = int(value)
            elif (isFloat(value)):
                globalParams[section + "." + item] = float(value)
            else:
                globalParams[section + "." + item] = value.strip()

            # if the comment already exists (i.e. from reading in _defaults)
            # and we are just resetting the value of the parameter (i.e.
            # from reading in inputs), then we don't want to destroy the
            # comment
            if comment.strip() == "":
                try:
                    comment = globalParamComments[section + '.' + item]
                except KeyError:
                    comment = ""
                    
            globalParamComments[section + "." + item] = comment.strip()
    
    
def getParam(key):
    """
    returns the value of the runtime parameter corresponding to the
    input key
    """
    if globalParams == {}:
        print "WARNING: runtime parameters not yet initialized"
        LoadParams("_defaults")
        
    if key in globalParams.keys():
        return globalParams[key]
    else:
        print "ERROR: runtime parameter ", key, " not found"
        

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
        print 'ERROR: unable to open inputs.auto'
        sys.exit()

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

        if (isInt(globalParams[key])):
            value = '%d' % globalParams[key]
        elif (isFloat(globalParams[key])):
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



    



