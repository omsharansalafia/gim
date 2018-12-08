import numpy as np
import observations
import gim
import transforms 

# here is an example loglikelihood function (this is the part you need to adapt to your needs, see example below)
def example_loglikelihood_function(datapoints,params,kwargs):
    """
    Compute the chi squared log-likelihood of the given datapoints wrt a model.
    
    Parameters:
    - datapoints: a list of instances of the class *datapoint*, which represent the data to be compared with the model predictions
    - params: a dictionary that must contain the model parameter values
    - kwargs: an additional dictionary with optinoal keyword arguments to this function
    
    Returns:
    - logL: the log-likelihood of the datapoints given the model
    """
    
    # start with a zero loglikelihood
    logL=0.
    
    # do your model preparation here
    # e.g. compute jet or kilonova dynamics
    # stuff here can depend on both the contents of *params*
    # and of *kwargs*
            
    # compute loglikelihood at each datapoint and sum it to L
    for obs in datapoints:
        
        # compute predicted model flux
        # F_model = some formula that depends on *params*, *kwargs*
        # that evaluates the flux density at time obs.t and frequency obs.nuobs
        F_model = 0.
        
        # compute likelihood (Chi-square-like. You can change this if you like)
        if obs.upper_limit:
            if F_model<obs.F:
                continue
            else:
                logL += -0.5*(((F_model-obs.F)/(obs.dF_high/3.))**2+np.log(2*np.pi*(obs.dF_high/3.)**2))
        else:
            if F_model<obs.F:
                logL += -0.5*(((F_model-obs.F)/(obs.dF_low))**2+np.log(2*np.pi*(obs.dF_low)**2))
            else:
                logL += -0.5*(((F_model-obs.F)/(obs.dF_high))**2+np.log(2*np.pi*(obs.dF_high)**2))
    
    if not np.isfinite(logL):
        return -np.inf
    else:
        return logL

# here is my implementation for the Gaussian structured jet as computed with my model. 
# It won't work on your system because you don't have my model installed, but you can 
# look into it to have an idea of how to write your own

# def GJ_loglikelihood(datapoints,params,kwargs):
    # """
    # Compute the chi squared log-likelihood of the given datapoints wrt a Gaussian structured jet afterglow model.
    
    # Parameters:
    # - datapoints: a list of instances of the class *datapoint*, which represents the data to be compared with the model predictions
    # - params: a dictionary that must contain the following keys:
        # - Ec: the jet core isotropic-equivalent energy, in erg
        # - thc: the jet core half-opening angle, in rad
        # - Gc: the jet core Lorentz factor
        # - thg: the angular scale over which the Lorentz factor falls off, in rad
        # - n: the ISM density, in cm^-3
        # - dL: the source luminosity distance, in cm
        # - z: the source redshift
        # - thv: the viewing angle, in rad
        # - ee: the shock microphysics parameter epsilon_e (which controls the fraction of shock energy that goes to electrons)
        # - eB: the shock microphysics parameter epsilon_B (which controls the fraction of shock energy that goes to the magnetic field)
        # - p: the shock microphysics parameter p (which controls the slope of the shock-heated electrons)
    
    # Keywords:
    # - th_res: the latitudinal resolution of the jet grid 
    # - phi_res: the longitudinal resolution of the jet grid
    # - side_expansion: whether to include side expansion in the dynamics
    
    # Returns:
    # - logL: the log-likelihood of the datapoints given the model
    # """
    
    # # extract parameters from the params dictionary (just for ease of coding)
    # Ec  = params['Ec']
    # thc = params['thc']
    # Gc  = params['Gc']
    # thg = params['thg']
    # n   = params['n']
    # dL  = params['dL']
    # z   = params['z']
    # thv = params['thv']
    # ee  = params['ee']
    # eB  = params['eB']
    # p   = params['p']
    
    # # extract kwargs
    # th_res=kwargs['th_res']
    # phi_res=kwargs['phi_res']
    # side_expansion=kwargs['side_expansion']
    
    # # start with a zero loglikelihood
    # logL=0.
    
    # # Gaussian jet structure
    # theta = np.logspace(-4,np.log10(np.pi/2.),1000)
    # Eth = Ec*np.exp(-0.5*(theta/thc)**2)
    # Gth = (Gc-1.)*np.exp(-0.5*(theta/thg)**2)+1.
    
    # # compute jet dynamics
    # if side_expansion:
        # dyn = jade.dynamics.spreading_structured_jet(theta,Eth,Gth,n,th_res,model='sound',us=2.)
    # else:
        # dyn = jade.dynamics.spreading_structured_jet(theta,Eth,Gth,n,th_res,model='sound',us=0.)
    
    # # compute EATS (equal-arrival-time surfaces)
    # eats = jade.eats(dyn,thv)
    
    # # compute loglikelihood at each datapoint and sum it to L
    # for obs in datapoints:
        
        # # compute predicted model flux
        # F_model = eats.synchrotron_flux(obs.t,obs.nuobs,dL,z,ee,eB,p)
        
        # # compute likelihood
        # if obs.upper_limit:
            # if F_model<obs.F:
                # continue
            # else:
                # logL += -0.5*(((F_model-obs.F)/(obs.dF_high/3.))**2+np.log(2*np.pi*(obs.dF_high/3.)**2))
        # else:
            # if F_model<obs.F:
                # logL += -0.5*(((F_model-obs.F)/(obs.dF_low))**2+np.log(2*np.pi*(obs.dF_low)**2))
            # else:
                # logL += -0.5*(((F_model-obs.F)/(obs.dF_high))**2+np.log(2*np.pi*(obs.dF_high)**2))
    
    # if not np.isfinite(logL):
        # return -np.inf
    # else:
        # return logL
        
# ---------------------------------------------------------------------------------------------------------------

# ini this example, we have a switch to select multinest or emcee

# switch multinest / emcee
multinest = False

# ---------------------------- loading event data to be used in the fit -----------------------------------------

# define event distance
dL = 40*3.08e24 # luminosity ditance in cm
z = 0. # redshift

# load the data, and make a list of datapoint instances with the observations we want to fit
datapoints = []

# example: load GW170817 radio data at 3 GHz
infile = "GW170817_radio.dat"
t,F,dF_high,dF_low,nuobsGHz,upper_limit = np.loadtxt(infile,unpack=True,usecols=(1,2,3,4,5,7),delimiter=',')

upper_limit = upper_limit.astype(bool)

# select frequencies we want to fit
my_selection = (nuobsGHz==3.)

# add detections first
for i in np.arange(len(t))[(~upper_limit) & my_selection]:
    datapoints.append(observations.datapoint(1e9*nuobsGHz[i],t[i],F[i],dF_high[i],dF_low[i]))

# then add upper limits (we don't know the noise level for these upper limits, so we just assume it is at 1/3 the upper limit).
for i in np.arange(len(t))[upper_limit & my_selection]:
    datapoints.append(observations.datapoint(1e9*nuobsGHz[i],t[i],F[i]/3.,F[i]/3.,0.,upper_limit=True))

# print the datapoints to make sure everything is right
print("Selected datapoints:")
print("nuobs (Hz), t (d), F (mJy), dF_high (mJy), dF_low (mJy), upper_limit (bool)")
for d in datapoints:
    print("{0:.3g}, {1:.3g}, {2:.3g}, {3:.3g}, {4:.3g}, {5:d}".format(d.nuobs,d.t,d.F,d.dF_high,d.dF_low,d.upper_limit))

print("")

# ---------------------------------- setup the fit -------------------------------------------------------------

# we now need to set up the parameter names and bounds, and the transforms that we want to be applied before passing them over to our
# likelihood function (keep in mind that this effectively associates priors on the parameters).

# define params and bounds (this also defines the parameter space dimensionality)
# the params in this example are those that go into GJ_afterglow_likelhood
params       = {'logEc':0  ,'thc':1   ,'logGc':2 ,'thg':3   ,'logn':4  ,'cos(thv)':5 ,'logeB':6  }
upper_bounds = {'logEc':55.,'thc':1.  ,'logGc':3.,'thg':1.  ,'logn':2. ,'cos(thv)':1.,'logee':0. ,'logeB':0. ,'p':4.}
lower_bounds = {'logEc':45.,'thc':1e-3,'logGc':1.,'thg':1e-3,'logn':-6.,'cos(thv)':0.,'logee':-6.,'logeB':-6.,'p':2.}
fixed_params = {'p':2.16,'dL':40*3.08e24,'z':0.008,'ee':0.1}

# define the transforms to apply to the params before passing them over to the loglikelihood evaluating function
corresp      = {'logEc':'Ec','thc':'thc','logGc':'Gc','thg':'thg','logn':'n','cos(thv)':'thv','logee':'ee','logeB':'eB','p':'p'}
transf       = {'logEc':transforms.pow10,
                'thc':transforms.identity,
                'logGc':transforms.pow10,
                'thg':transforms.identity,
                'logn':transforms.pow10,
                'cos(thv)':transforms.arccos,
                'logee':transforms.pow10,
                'logeB':transforms.pow10,
                'p':transforms.identity}

# kwargs
kwargs = {'th_res':30,'phi_res':30,'side_expansion':False}

# instantiate the lnlike class with these settings
my_lnlike = gim.lnlike(example_loglikelihood_function,datapoints,params,lower_bounds,upper_bounds,fixed_params,corresp,transf,kwargs)

if multinest:
    
    parameter_names = ['' for i in range(my_like.ndims)]
    
    for k in params.keys():
        parameter_names[params[k]]=k
    
    def prior_transform(cube):
        # this assumes uniform priors in these parameters
        return cube * (my_lnlike.lower_bounds - my_lnlike.upper_bounds) + my_lnlike.lower_bounds
    
    from pymultinest.solve import solve
    prefix = 'out_multinest'
    result = solve(LogLikelihood=my_lnlike, Prior=prior_transform, 
        n_dims=len(my_lnlike.upper_bounds), outputfiles_basename=prefix, 
        verbose=True)
    np.savez(prefix + "results.npz", **results)
    
    print()
    print('evidence: %(logZ).1f +- %(logZerr).1f' % result)
    print()
    print('parameter values:')
    for name, col in zip(parameter_names, result['samples'].transpose()):
        print('%15s : %.3f +- %.3f' % (name, col.mean(), col.std()))

    # make marginal plots with corner.py using results.npz['samples']
    # or make marginal plots by running:
    # $ python multinest_marginals.py out_GRB_afterglow_likelihood
    # For that, we need to store the parameter names:
    import json
    with open('%sparams.json' % prefix, 'w') as f:
        json.dump(parameter_names, f, indent=2)

else:
    import emcee

    # initialize sampler
    ndim = my_lnlike.ndims
    nwalkers = 4*ndim
    
    sampler = emcee.EnsembleSampler(nwalkers,ndim,my_lnlike,threads=2)
    
    # the initial positions of the walkers are uniformly distributed within the bounds
    p0s = np.array([my_lnlike.random_point() for j in range(nwalkers)])
    
    import matplotlib.pyplot as plt
    plt.scatter(p0s[:,0],p0s[:,1])
    plt.show()
    
    # burn in
    print('Burn in...')
    pos, prob, state = sampler.run_mcmc(p0s, 1000)
    sampler.reset()
    print('Done.\n Sampling...')
    
    # run sampler
    nsteps = 30000
    print('')
    
    for i, result in enumerate(sampler.sample(pos, iterations=nsteps)):
        if (i+1) % 30 == 0:
            print("Step {0:d}".format(i))
            np.savez("emcee_tempchain.npz",sampler.chain)
            np.savez("emcee_templnprob.npz",sampler.lnprobability)
            
    print('')
    
    # save complete chains
    np.savez("emcee_last_completed_chain.npz",sampler.chain)
    np.savez("emcee_last_completed_lnprob.npz",sampler.lnprobability)

        
        
        
