##  		MCMC functions

import numpy as np
import emcee
import local_variables as loc		# file with the local path pointers and cpu settings
import visibfit_functions as vf


def log_likelihood( pars, galargs, two_comp): 
	'''Galario fit chi2 likelihood function'''
	chi2 = vf.galario_model( pars=pars, galargs=galargs[0], two_comp=two_comp )[1]
	chi2_c = vf.galario_model( pars=pars, galargs=galargs[1], two_comp=two_comp )[1] if len(galargs) > 1 else 0
	return -0.5 * (chi2 + chi2_c)

def log_prior( pars, p_ranges, two_comp): 
	''' prior dist. pars is the array of free parameters, p_ranges their boundaries'''
	if (p_ranges[:, 0] < pars).all() and (pars < p_ranges[:, 1]).all():
		if two_comp == True:
			Rout_constrain = (pars[3]*pars[4] > 1) & (pars[3]*pars[4] < vf.Rmax_model)		# 1" < Rout < Rmax (galario grid)
			Ri_constrain = (pars[3] >= 0.7 *pars[2]) # & (pars[3] < 10 *pars[2])						# 0.7*sigma < Ri  #< 3*sigma
			I_constrain = pars[0] >= pars[1]
			if Ri_constrain & Rout_constrain & I_constrain:	
				return 0.0
			else: return -np.inf
		else:
			return 0.0
	else:	
		return -np.inf

def log_probability( pars, p_ranges, galargs, two_comp):
	logprior = log_prior( pars=pars, p_ranges=p_ranges, two_comp=two_comp)
	if not np.isfinite( logprior):
		return -np.inf
	return logprior + log_likelihood( pars, galargs, two_comp)



def mcmc_run( galargs, p0, p_ranges, nsteps, nwalkers, nthreads, two_comp, backend_fname, thin_f=loc.thin_f, append=False):
	'''
	Launch an MCMC run for the galario fitting. 
	galargs:  	Rmin, dR, nR, nxy, dxy, u, v, Re, Im, w
	p0: 		starting guess parameter vector
	append=True will result in the continuation of previously saved chains
	'''
	ndim = len(p0)
	startpos = None
	bknd_samp = emcee.backends.HDFBackend( backend_fname + '.h5')		# store sampler on file
	if append == False: 
		bknd_samp.reset( nwalkers=nwalkers, ndim=ndim)
		startpos = p0 + 1e-2* np.random.randn( nwalkers, ndim) 	# initialize the walkers with an nD Gaussian ball
	else: print('\n Continuining previous chains from saved backend. \n')
	
	sampler = emcee.EnsembleSampler( nwalkers, ndim, log_probability, args=(p_ranges, galargs, two_comp), 
							# threads=nthreads, 
							backend=bknd_samp, # live_dangerously=False, 
			moves=[ (emcee.moves.DEMove(), 0.7), (emcee.moves.DESnookerMove(), 0.3),], 	# mv1
			# moves=[ (emcee.moves.StretchMove(), 0.5), (emcee.moves.DEMove(), 0.5),], 		# mv2
			# moves = emcee.moves.KDEMove(), 	# mv3	
			)
	# state = sampler.run_mcmc( startpos, 100, progress=progbar, store=False)		# pre-run for hard burn-in
	# new_p0 = np.quantile( state.coords,  0.50, axis=0) + 1e-2* np.random.randn( nwalkers, ndim)
	# sampler.reset()
	sampler.run_mcmc( startpos, nsteps, progress=loc.progbar, store=True, thin=thin_f)			# full production run
	return  # sampler



def regress_mcmc( run_meta: vf.RunParams):
	'''
	Main pipeline for fitting YSO models with galario to a sky model (filename).
	'''
	galargs = [vf.get_galargs( run_meta=run_meta, config_name=config ) for config in run_meta.config_list]		# one or two if SC or CC

	# parameter space domain
	p_ranges_2c = np.array([[7.0, 13],	# Log10( I0disk )	[Log(Jy/sr)]
						[6.5, 13.],		# Log10( IOenvelope)   
						[1e-2, .8],		# sigma i.e. sma [arcsec]
						[1e-2, 1.6],	# Ri [arcsec] (Rmax= 8 / 5 = 1.6, to avoid an envelope cut at high fluxes)
						[5, 1000],		# Rout/Ri [arcsec] fraction of Ri		# [3e-4, 8]
						[1.3, 2.99],	# p_index []
						[-15., 100.],	# inc (deg)
						[-7, 180.],		# PA (deg)
						[-2, 2],		# dRa (arcsec)
						[-2, 2]])		# dDec (arcsec)

	p_ranges_gauss = np.array([[7, 15],	# Log10( I0 )	[Log(Jy/sr)]
						# [0.2, 0.9],	# Hr0
						[1e-5, .8],		# sigma i.e. sma 	[arcsec]
						[-10., 100.],	# inc (deg)
						[-7, 180.],		# PA (deg)
						[-2, 2],		# dRa (arcsec)
						[-2, 2]])		# dDec (arcsec)

	# initial guess for the parameters
	p0_2c = np.array([11, 8.4, 0.2, 0.4, 6, 2.5, 60., 45., 0., 0.]) 	# Log(I0), Log(Ienv), sma, Rin, Rout/Ri, p_idx, (inc, PA, dRA, dDec)
	p0_gauss = np.array([12, 0.2, 80., 45., 0., 0.])				# Log(I0), sma, inc, PA, dRA, dDec
	if run_meta.two_comp:
		p0_mc = p0_2c
		p_rang_mc = p_ranges_2c
	else:
		p0_mc = p0_gauss
		p_rang_mc = p_ranges_gauss

	# execute the MCMC
	backend_name = run_meta.targetpath + run_meta.diskname + '__sampler'	# to save the chains
	mcmc_run( galargs=galargs, p0= p0_mc, p_ranges= p_rang_mc, 
			nsteps=run_meta.nsteps, nwalkers=loc.Nwalkers, nthreads=loc.Ncpu, backend_fname=backend_name, 
			two_comp=run_meta.two_comp, append=False )

	