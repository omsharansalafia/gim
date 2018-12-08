import numpy as np


class lnlike:
    """
    Class that defines the interface to our loglikelihood function, to be fed to emcee or multinest.
    
    Init params:
    - datapoints: a list of datapoint instances representing the data we want to fit
    - params: a dictionary containing parameter names and their position in the input array, i.e. {'param name':position, ...}, where position is an int.
    - lower_bounds: a dictionary with the lower bounds on parameters, e.g. {'param name': lower_bound}
    - upper_bounds: a dictionary with the upper bounds on parameters, e.g. {'param name': upper_bound}
    - fixed_params: an optional dictionary with the parameters that we want to hold fixed, of the form {'fixed param name': value}
    - lnlike_function: actual likelihood evaluating function, which must be called as func(datapoint_list,param_dict,kwarg_dict)
    - corresponding_params: if given, the names of the params in param_dict passed over to lnlike_function are different from the input parameters, and the correspondences are defined by this dictionary, which must be of the form {'param name':'corresponding param name'}
    - transforms: if given, the parameters in the param_dict passed over to the lnlike_function are transformed through these functions. Otherwise, all entries are set to {'param name': lambda x:x}
    - kwargs: optional dictonary of keyword arguments to be passed over to the likelihood evaluating function
    """
    
    def __init__(s,lnlike_function,datapoints,params,lower_bounds,upper_bounds,fixed_params=None,corresponding_params=None,transforms=None,kwargs=None):
        
        s.lnlike_function = lnlike_function
        s.datapoints = datapoints
        s.params = params
        
        # transform bound dictionaries into arrays
        s.lower_bounds = np.empty(len(params))
        s.upper_bounds = np.empty(len(params))
        for k in params.keys():
            s.lower_bounds[params[k]]=lower_bounds[k]
            s.upper_bounds[params[k]]=upper_bounds[k]        
        
        s.fixed_params = fixed_params
        s.ndims = len(params)
        s.param_names = [n for n in params.keys()]
        s.kwargs = kwargs
        
        if corresponding_params is None:
            s.corresponding_params = {}
            for k in s.param_names:
                s.corresponding_params[k]=k
        else:
            s.corresponding_params = corresponding_params
        
        if transforms is None:
            s.transforms = {}
            for k in s.param_names:
                s.transforms[k] = lambda x:x
        else:
            s.transforms = transforms
                
        
    def __call__(s,x):
        """
        Evaluate the likelihood at point x in the parameter space. 
        x must be a numpy array with the same number of dimensions
        as the (free) parameter space.
        """
        # check bounds
        if np.any(x<s.lower_bounds) or np.any(x>s.upper_bounds):
            return -np.inf
        
        # transform imput parameters into the corresponding output ones
        out_params = {}
        for k in s.param_names:
            out_params[s.corresponding_params[k]]=s.transforms[k](x[s.params[k]])
        
        # put the remaining, fixed parameters into the output dictionary
        if s.fixed_params is not None:
            for k in s.fixed_params.keys():
                out_params[k]=s.fixed_params[k]
        
        # call the likelhood evaluating function
        return s.lnlike_function(s.datapoints,out_params,s.kwargs)
    
    def random_point(s,seed=None):
        """
        Generate a random point in parameter space (uniformly distributed
        between the bounds). Useful to generate starting positions for the
        walkers of an MCMC.
        """
        if seed is None:
            p0 = np.empty(s.ndims)
            for k in s.param_names:
                p0[s.params[k]]=np.random.uniform(s.lower_bounds[s.params[k]],s.upper_bounds[s.params[k]])
            
        else:
            p0 = np.empty(s.ndims)
            for k in s.param_names:
                p0[s.params[k]]=np.random.uniform(s.lower_bounds[s.params[k]],s.upper_bounds[s.params[k]])
        
        return p0
        
