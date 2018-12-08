import numpy as np

class datapoint:
    """
    Class that defines a datapoint to be fitted by our model.
    
    Init params:
    - nuobs: observing frequency in Hz
    - t: observer time in days (after the jet launch)
    - F: flux density in mJy
    - dF_high, dF_low: high ad low one-sigma errors on the flux density
    
    Init keywords:
    - upper_limit: True if this is an upper limit, in which case the value of F should be the noise level, and dF_high should be the 3 sigma upper limit
    - ABmag: if True, the F and dFs are interpreted as AB magnitudes, and they are thus converted to mJy
    
    Methods:
    - extrap(nu,alpha): extrapolate the flux density to a frequency nu (different from nuobs), by assuming a power-law spectrum Fnu propto nu**alpha with a given slope alpha.
                        Returns another instance of this class with the new nuobs and the extrapolated flux.
    
    """
    
    def __init__(s,nuobs,t,F,dF_high,dF_low,upper_limit=False, ABmag=False):
        s.nuobs = nuobs
        s.t = t
        if ABmag:
            s.F = 10**((8.9-F)/2.5 + 3.)
            s.dF_high = 10**((8.9-(F-dF_low))/2.5 + 3.) - s.F
            s.dF_low = -10**((8.9-(F+dF_high))/2.5 + 3.) + s.F
        else:
            s.F = F
            s.dF_high = dF_high
            s.dF_low = dF_low
        s.upper_limit = upper_limit
    
    def extrap(s,nu,alpha):
        return datapoint(nu,s.t,(nu/s.nuobs)**alpha*s.F,(nu/s.nuobs)**alpha*s.dF_high,(nu/s.nuobs)**alpha*s.dF_low,s.upper_limit)

