import numpy as np

# some useful functions to be used as parameter transforms
# the parameter transforms must not be lambdas, otherwise they cannot be pickled and thus they cannot be used in multithreading!
def pow10(x):
    return 10**x

def identity(x):
    return x

def arccos(x):
    return np.arccos(x)
