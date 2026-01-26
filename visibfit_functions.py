# # # Full pipeline to mock obs + galario fit and plots of regression products.

import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import os
from local_variables import *		# file with the local path pointers and cpu settings
from astropy.io import fits
import emcee
import corner
import pandas as pd
import casatasks as ctk
import casatools as cto
# mpl.use('agg')		# quick fix required to import uvplot ...
import uvplot as uvp
import galario.double as gd 
from galario import deg, arcsec
import scipy.ndimage as snd
# from scipy.optimize import curve_fit
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops, regionprops_table
from skimage.morphology import closing, footprints

mm3 = 0.003		# wavelength [metres]
Rmax_model = 8	# [arcsec]	


def compare_gauss_plumm():
	rarr = np.linspace(0, 5, 500)
	sma = 0.5
	gp = GaussianProfile( rarr, 1, sigma=sma)
	ris = sma * np.array([0.5,1,2])
	pidxs = np.array([1.3, 2, 3,])
	n = len(ris) * len(pidxs)
	colors = plt.cm.jet( np.linspace(0,1,n) )	# colouring lines
	i = 0
		
	fig, ax = plt.subplots( figsize=(8, 8))		# diagnostic figure
	ax.plot( rarr, gp, c='k', lw=2, label=f'gaussian, $\sigma$={sma :.1f}')

	for ri in ris:
		for p in pidxs:
			pp = Plummer_envelope( rarr, 1, ri, 5, p)
			ax.plot( rarr, pp, c=colors[i], alpha=0.7, label=f'Ri={ri}, p={p :.2f}')
			ax.set( xscale='linear', yscale='log')
			i +=1

	ax.legend()
	plt.show()


def crop_image( img, centre=None, margins=[100, 100] ):
	'''Select a subimage of margins pixels around the centre (odd size). Squared only'''
	if centre is None:      	# use the middle of the image
		centre = (np.array( img.shape)/2 ).astype(int)
	if margins[0] > min( centre[0], img.shape[0] - centre[0]):
		print( 'margins exceed original image boundary, no crop possible.\n')
		return img
	else:
		return img[ centre[0] - margins[0] : centre[0] + margins[0] +1, centre[1] - margins[1] : centre[1] + margins[1] +1]


def GaussianProfile( R, I0, sigma ):
	'''
	Disk gaussian brighntess profile. R is the projected radius array.
	sigma is the disk semimajor axis. 
	'''
	return I0 * np.exp( - 0.5 * (R / sigma)**2  )


def Plummer_envelope( R, I0, Ri, Rout, p_index ):
	'''
	2D (truncated) Plummer envelope brightness profile. 
	R is the radius array, Ri the transitional radius from rotation to infall,
	p_index the powerlaw exponent sum (m+n)
	'''
	I = I0 * ( 1 + (R/Ri)**2 )**( -(p_index - 1) / 2 )		#Plummer brightness profile
	I[ np.where( R > Rout ) ] = 0		#truncating the profile at r > Rout
	return I


def diskheight_correction( model_image, dxy, sigma, inc, H_r=0.4):
	'''
	Give thickness to galario models with a simple trick.
	'''
	# H_r = 0.35
	H = sigma *2.1436209 * H_r			# [rad], disk height at R90
	pixheight = H * np.sin( inc) / dxy	# [pix], LoS-projected disk height
	if round(pixheight) >= 1:			# only if the projection is visible
		thickened = model_image.copy()
		for i in range(1, round(pixheight) +1):
			s_target = np.roll( model_image, shift=i, axis=1) # snd.shift( model_image, shift=( 0, i ) )  			# shift the model
			thickened = thickened + s_target
		thickened = thickened / (1 + round(pixheight) )		# divide by number of superpositions so total flux is roughly conserved
		# thickened = np.roll( thickened, shift=-round(round(pixheight)/2 + 0.1), axis=1)	# centre closer to original, but not fundamental
		return thickened
	else:
		return model_image


def galario_model( pars, galargs, two_comp=True):
	'''
	Let galario generate a model on the visibilities and return a Chi2 to the data.
	pars: LI0d, (LI0e), sigma, (Ri, Rout/Ri, p_index,) inc, PA, dRA, dDec  , (): if two_comp is True
	galargs:  Rmin, dR, nR, nxy, dxy, u, v, Re, Im, w
	two_comp: whether to use the only a gaussian (False) or gauss + plummer (True)
	returns: target_models (disk + env), chi2, vis_model (target)
	'''
	Rmin, dR, nR, nxy, dxy, u, v, Re, Im, w = galargs
	if two_comp:
		LI0d, LI0e, sigma, Ri, Rout_Ri, p_index, inc, PA, dRA, dDec = pars	
		I0e = 10.**LI0e ; Ri *= arcsec ; Rout = Rout_Ri * Ri 		# Rout *= arcsec
	else: 
		LI0d, sigma, inc, PA, dRA, dDec = pars			# unpack the parameters
	I0d = 10.**LI0d       # convert from log to real space

	# convert everything to radians
	sigma *= arcsec; Rmin *= arcsec; dR *= arcsec
	inc *= deg; PA *= deg ; dRA *= arcsec ; dDec *= arcsec
	R = np.linspace( Rmin, Rmin + dR * nR, nR, endpoint=False)		# radial grid

	I_disk = GaussianProfile( R, I0d, sigma=sigma)				# Brightness profile of YSO
	disk_model = gd.sweep( I_disk, Rmin, dR, nxy, dxy, inc=inc)	# [Jy/pix]
	if two_comp: 
		I_env = Plummer_envelope( R, I0e, Ri, Rout, p_index)	# Brightness profile of the envelope 
		env_model = gd.sweep( I_env, Rmin, dR, nxy, dxy, inc=0) 	
	else: env_model = 0
	# disk_model = diskheight_correction( disk_model, dxy=dxy, sigma=sigma, inc=inc, H_r=Hr0 )		# add disk thickness
	target_model = disk_model + env_model
	# Compute visibilities for the central target that requires rotation and offsets (with PA, inc, dRA, dDec)
	vis_target = gd.sampleImage( target_model, dxy, u, v, PA=PA, dRA=dRA, dDec=dDec, check=False, origin='lower') 

	chi2 = np.sum( w * ((vis_target.real - Re)**2 + (vis_target.imag - Im)**2) )	# chi2 in visibility space (equivalent to galario reduce_chi2()?)
	return (disk_model, env_model), chi2, vis_target


def copy_extra_sources( MSname, nRMS, deconvmod=True ):
	'''
	Create a copy of CASA noisy image for the areas above noise and put everything else (including central target) to zero.
	'''
	table = cto.table()
	table.open( MSname.replace('.ms', '') + '.image' )					# noisy image (for regions selection only)
	noisy_img = table.getcol('map').squeeze().copy( order='F').T 		# copy simanalyze noisy image (convolved)  [Jy/beam]
	beam_dict = table.getkeyword('imageinfo')['restoringbeam']			# a, b and PA of beam
	beam_area = np.pi * beam_dict['major']['value'] * beam_dict['minor']['value'] / (4*np.log(2))	# FWHM ellipse area [arcsec^2/beam]
	img_pixscale = abs( table.getkeyword('coords')['direction0']['cdelt'][0])		# [rad/pix] of noisy image
	beam_to_pix = ( 3600* np.rad2deg( img_pixscale ) )**2  / beam_area				# to convert the flux from [Jy/beam] to [Jy/pix]
	xsrc_img = noisy_img	# deprecated !
	factor = beam_to_pix
	if deconvmod:
		table.open( MSname.replace('.ms', '') + '.model' )		
		deconvolved = table.getcol('map').squeeze().copy( order='F').T			# deconvolved model image of the sky [Jy/pix]
		xsrc_img = deconvolved
		factor = 1		# deconv is already in [Jy/pix]
	table.close()

	## apply threshold to identify the sources on the convolved image
	thresh = nRMS * min_bkg_rms( noisy_img ) 		# min_bkg_rms( noisy_img )
	bw = closing( noisy_img > thresh, footprints.rectangle(3, 3) )
	cleared = clear_border( bw )		# remove artifacts connected to image border
	label_image = label( cleared )		# label image regions
	nimg_masked = np.where( noisy_img > thresh, xsrc_img, 0)		# keep everything above n*RMS 
	
	# little additional check
	# if rms( crop_image( noisy_img, margins=[60,60]) )
	label_image10 = label( clear_border( closing( noisy_img > 10* min_bkg_rms( noisy_img ), footprints.rectangle(3, 3) ) ) )
	print( len( np.unique( label_image)), 'sources detected with nRMS =', nRMS, f'\t({len( np.unique( label_image10))} with 10xRMS)' )
	
	## exclude the central source !
	sources_df = pd.DataFrame( regionprops_table( label_image, properties=('centroid', 'orientation'), ) ).rename( columns={'centroid-0':'y0', 'centroid-1':'x0'} )
	target_idx = ((sources_df[['y0','x0']] - np.array(noisy_img.shape)/2 )**2 ).sum( axis=1).idxmin() + 1	# central source (target)
	nimg_masked[ label_image == target_idx ] = 0		# zero on the main target

	if (nimg_masked > 0).any():	
		fig, ax = plt.subplots( figsize=(5, 5))		# diagnostic figure
		if deconvmod: 
			smooth_r = ( beam_area / np.pi )**0.5 / np.rad2deg( img_pixscale )/3600		# smoothing radius in [pix]
			diag_img = snd.gaussian_filter( nimg_masked, sigma=smooth_r )
			diag_img[ noisy_img < thresh ] = np.nan	; diag_img[ label_image == target_idx ] = np.nan		# just for visualisation
		else:
			diag_img = np.where( noisy_img > thresh, xsrc_img, np.nan)		# use noisy_img just for diagnostic plot
			diag_img[ label_image == target_idx ] = np.nan
		ax.imshow( diag_img, origin='lower', norm=mpl.colors.SymLogNorm( linthresh=thresh ) )
		ax.imshow( np.where(label_image==target_idx, 1, np.nan), origin='lower', cmap='bwr_r')			# mark the target position
		ax.set_axis_off()
		fig.savefig( 'sky_xsrc_map' + MSname.strip('image') + fig_ext, bbox_inches='tight', dpi=300)
		plt.close()
	else: 
		print( '\nNo extra sources found in the image!\n' )
		return (0, img_pixscale)
	# nimg_masked[ nimg_masked <= 1e-50 ] = 1e-50			# remove negative values
	return nimg_masked * factor, img_pixscale		# [Jy/pix], [rad/pix]


def xsrc_to_visib( extra_sources, u, v ):
	'''
	Convert a sky image [Jy/pix] of extra sources to complex visibilities. 
	'''
	yy, xx = np.indices( extra_sources[0].shape ) 
	deltas_pix =  - ( xx - extra_sources[0].shape[1] / 2) +1, yy - extra_sources[0].shape[0] / 2 +1		# (+1 offset due to CASA pixel centring)
	dRA, dDec = deltas_pix[0] * extra_sources[1], deltas_pix[1] * extra_sources[1]		# [rad]
	xsrc_vis = 0 + 0.j
	clean_sources_pos = np.argwhere( abs(extra_sources[0]) > 1e-10 )
	print( 'Converting', len(clean_sources_pos), 'points of extra sources image in visibilities' )
	for p in clean_sources_pos:
		f = extra_sources[0][ p[0], p[1]]		# read the pixel flux value
		Re = np.full_like( u, f )			# add each clean component as constant centred source (phase=0)
		vis_sh = gd.apply_phase_vis( dRA[p[0], p[1]], dDec[p[0], p[1]], u, v, Re + 0.j)		# shift the component to its place in the sky
		xsrc_vis = xsrc_vis + vis_sh
	return xsrc_vis


##  MCMC functions

def log_likelihood( pars, galargs, two_comp): 
	'''Galario fit chi2 likelihood function'''
	chi2 = galario_model( pars=pars, galargs=galargs[0], two_comp=two_comp )[1]
	chi2_c = galario_model( pars=pars, galargs=galargs[1], two_comp=two_comp )[1] if len(galargs) > 1 else 0
	return -0.5 * (chi2 + chi2_c)

def log_prior( pars, p_ranges, two_comp): 
	''' prior dist. pars is the array of free parameters, p_ranges their boundaries'''
	if (p_ranges[:, 0] < pars).all() and (pars < p_ranges[:, 1]).all():
		if two_comp == True:
			Rout_constrain = (pars[3]*pars[4] > 1) & (pars[3]*pars[4] < Rmax_model)		# 1" < Rout < Rmax (galario grid)
			Ri_constrain = (pars[3] >= 0.7 *pars[2]) # & (pars[3] < 10 *pars[2])						# 0.7*sigma < Ri  #< 3*sigma
			if Ri_constrain and Rout_constrain:	
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


def mcmc_run( galargs, p0, p_ranges, nsteps, nwalkers, nthreads, two_comp=False, backend_fname='last_sampler', append=False):
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
	sampler.run_mcmc( startpos, nsteps, progress=progbar, store=True, thin=2)			# full production run
	return sampler


# def mask_chains( samples, thresh=5):
# 	'''
# 	Mask the walker values that are more than thresh sigma away from the median. This removes the steps of a walker before it finds the common minimum.
# 	'''
# 	step_median = np.median( samples, axis=1, keepdims=True)		# median at every step among walkers (evaluate step by step)
# 	walkers_std = np.std( samples, axis=1, keepdims=True)			# stds of walkers spread for given step and param
# 	masked = np.where( np.abs( samples - step_median)  > thresh * walkers_std, np.nan, samples )
# 	if np.isnan( masked ).any():
# 		print( f'Masked {np.count_nonzero( np.isnan(masked))} walker values that are more than {thresh} sigma away from the step median.')
# 		return masked
# 	else:
# 		print( 'No walkers to mask.')
# 		return samples
	
	
def clip_chains( samples, thresh=5):
	'''
	Discard the walkers that are more than thresh sigma away from the median. This removes entire walkers that are stuck throughout the chain.
	'''
	steps_median = np.median( samples, axis=0)		# median of all the steps for each walker
	param_std = np.std( steps_median, axis=0)		# std of parameter posteriors
	clip_idx = np.argwhere( (np.abs( steps_median - np.median( steps_median, axis=0) ) > thresh * param_std ).any( axis=1 ) )	# if exceeds thresh in any param
	if len(clip_idx) > 0:
		print( f'Clipping {len(clip_idx)} walkers that are more than {thresh} sigma away from the median.')
		clipped = np.delete( samples, clip_idx, axis=1 )		# remove the chains of the outlying walkers
		return clipped
	else:
		print( 'No walkers to clip.')
		return samples


def mcmc_plots( samp_bkend, labels, burn_in, walk_clip_thresh=5, figures=True, folder=''):
	'''
	Show the traces of mcmc steps for sampler run and the corner plot.
	'''
	try:
		# print( "Mean acceptance fraction: {0:.3f}".format( np.mean(samp_bkend.acceptance_fraction) ) )
		tau = samp_bkend.get_autocorr_time( discard=int(burn_in), quiet=True)
		print( 'autocorr time: \t', tau)
		new_burn_in = int(2 * np.max(tau))      # discard the burn-in steps based on autocorrelation
		# thinning = int(0.5 * np.min(tau))
	except: 
		print('It was not possible to determine the autocorrelation time tau')
		pass

	samples = samp_bkend.get_chain( discard=int(burn_in) )
	if walk_clip_thresh != None:
		samples = clip_chains( samples, thresh=walk_clip_thresh)		# remove outlying walkers
	flat_samples = samples.reshape( -1, len(labels) )		# discarding the burn-in steps in the first step before chains

	fig, axes = plt.subplots( len(labels), figsize=(8, 8), sharex=True)			# CHAIN traces
	for i in range( len(labels)):
		ax = axes[i]
		ax.plot( samples[:, :, i], "k", alpha=0.3)
		ax.set_xlim(0, len(samples))
		ax.set_ylabel( labels[i])
		ax.yaxis.set_label_coords(-0.1, 0.5)
	axes[-1].set_xlabel("step number")
	fig.savefig( folder + 'chains_steps' + fig_ext, dpi=400)
	if figures: plt.show()
	plt.close()

	cornfig = plt.figure( figsize=(8,8))		# CORNER PLOT
	fig = corner.corner(
		flat_samples, labels=labels, quantiles=[0.16, 0.5, 0.84], # title_quantiles=[0.5],
		show_titles=True, fig=cornfig, 
		label_kwargs={'labelpad':20, 'fontsize':0}, #fontsize=8,
		title_kwargs={"fontsize": 10, 'loc':'left'},	
		)
	cornfig.savefig( folder + 'corner_plot' + fig_ext, bbox_inches='tight')
	if figures: plt.show()
	plt.close()

	# best parameters from the walker step with lowest chi2
	# best_idx = np.unravel_index( samp_bkend.get_log_prob().argmin(), samp_bkend.get_log_prob().shape )
	# best_pars = samp_bkend.get_chain()[best_idx]
	best_pars = np.percentile( flat_samples,  [50, 16, 84], axis=0).T     # best params out of fit + 16% - 18% values !
	return best_pars


def fix_skyflux( data_folder = '/Users/gcolumba/PostDoc_Mac/PostProc/simulations/sky_RT/3mm/' ):
	import glob
	import sys
	from astropy.io import fits
	
	fitslist = sorted( glob.glob( data_folder + '*.fits') )
	if fitslist == []:
		print('NO FILES FOUND, check again the folder path!')
		sys.exit()
	print( len(fitslist), 'files found')

	for fname in fitslist:
		hdul = fits.open( fname )
		hdul.info()
		hdr = hdul[0].header
		sky_image = hdul[0].data 
		fits.writeto( fname, data=sky_image*25, header=hdr, overwrite=True)
	print('\n skyfix completed !')


def cancel_extra_sources( skymodel, nRMS=1, figure=False ):
	'''
	Create a copy of CASA noisy image for the areas above noise and put everything else (including central target) to zero.
	'''
	sky_image = skymodel.copy()
	print( '\n LEVELLING OFF the extra sources with the RMS value. \n')

	## apply threshold to identify the bright sources
	noise_lev = rms( sky_image ) 		# min_bkg_rms( noisy_img )
	thresh = nRMS * noise_lev
	bw = closing( sky_image > thresh, footprints.rectangle(3, 3) )
	cleared = clear_border( bw )		# remove artifacts connected to image border
	label_image = label( cleared )		# label image regions
	sky_masked = np.where( sky_image > thresh, thresh, sky_image)		# keep everything below n*RMS 	# keep thresh or RMS ?
	# sky_masked = np.clip( sky_image, a_max= 3* min_bkg_rms( sky_image), a_min=None)	# this cancels more but creates gradini

	sources_df = pd.DataFrame( regionprops_table( label_image,
		properties=('centroid', 'orientation', 'axis_major_length', 'axis_minor_length', 'equivalent_diameter_area'), ) ).rename(
			columns={'centroid-0':'y0', 'centroid-1':'x0', 'orientation':'PA', 'axis_major_length':'a', 'axis_minor_length':'b', 
				'equivalent_diameter_area':'diam'} )
	
	## keep the central source !
	target_idx = ((sources_df[['y0','x0']] - np.array(sky_image.shape)/2 )**2 ).sum( axis=1).idxmin()	# central source (target)
	miny, minx, maxy, maxx = regionprops( label_image )[target_idx].bbox			# rectangle over central source
	sky_masked[ miny:maxy , minx:maxx] = sky_image[ miny:maxy , minx:maxx ]			# restore the entire target rectangle (envelopes should be safe too)

	if figure:	
		fig, ax = plt.subplots( figsize=(8, 8))		# diagnostic figure
		# thresh = 3* min_bkg_rms( sky_image)		# gradino
		diag_img = np.where( sky_image > thresh, np.nan, sky_image)
		diag_img[ miny:maxy , minx:maxx] = sky_image[ miny:maxy , minx:maxx ]	
		ax.imshow( diag_img[:, ::-1 ], origin='lower', norm=mpl.colors.LogNorm() )	# use noisy_img just for diagnostic plot
		ax.set_axis_off()
		# plt.show()
		fig.savefig( 'levelled_sky' + fig_ext, bbox_inches='tight', dpi=200)
		plt.close()

	return sky_masked


def twoD_Gaussian( img_size, I0, xo, yo, sigma_x, sigma_y, theta, offset ):
	x, y = np.meshgrid( np.arange(0, img_size[0]), np.arange(0, img_size[1]) )
	xo = float(xo)
	yo = float(yo)    
	a = np.cos(theta)**2 /(2*sigma_x**2) + np.sin(theta)**2 /(2*sigma_y**2)
	b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
	c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
	g = offset + I0 * np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2)))
	return g.ravel()


def generate_skymodel( dRA, dDec, a, b, like_filename='disk17_xy_3000um.fits', data_folder='/Users/gcolumba/PostDoc_Mac/PostProc/simulations/3mm/' ):
	'''Create a sky model with a target and some extra sources for mock observations.
	like_filename: a fits file (path) with the sky model to be used as a template.'''
	hdul = fits.open( data_folder + like_filename )
	hdul.info()
	hdr = hdul[0].header
	pixscale = hdr['CDELT1']	# [deg / pix] , copying it from the "real" skymodel
	modelshape = hdul[0].data.shape
	arr = np.zeros( shape= modelshape, dtype=np.float32)		# 4895x4895 pixels, as the 3mm sky models
	
	# create a gaussian target with an offset
	dDec_pix, dRA_pix = dDec / (pixscale*3600), dRA / (pixscale*3600)		# [arcsec] to pixel offset
	PA = np.pi / 9 	#; b = 7 ; a = 12 ; 
	target = twoD_Gaussian( img_size=arr.shape, I0=9e-3, xo=arr.shape[0]/2 + dRA_pix, yo=arr.shape[1]/2 + dDec_pix,
		sigma_x=b, sigma_y=a, theta=-PA, offset=1e-10)		# [Jy/pix]	theta < 0 corresponds to a countclock rotation of the disk
	arr += target.reshape( arr.shape )		# add the target to the array

	# # then add other sources along a vertical line from the target to the top of the image with equal spacing and brightness
	nsources = 4		# number of sources to add
	for i in range( 1, nsources + 1):
		dDec_pix = modelshape[0]/6 / (nsources + 1) * i		# pixel offsets
		extrasource = twoD_Gaussian( img_size=arr.shape, I0=7e-4, xo=arr.shape[0]/2 + 0, yo=arr.shape[1]/2 + dDec_pix,
			sigma_x=5, sigma_y=5, theta=0, offset=0)		# [Jy/pix]
		arr += extrasource.reshape( arr.shape )		# add the target to the array
	
	plt.imshow( arr, origin='lower', norm=mpl.colors.LogNorm(), cmap='gnuplot2')
	plt.colorbar( label='Jy/pix')
	plt.show()
	# save the model to fits
	hdr['CTYPE1'] = 'RA---SIN'		# set the header keywords
	hdr['CTYPE2'] = 'DEC--SIN'
	hdr['INFOMAIN'] = (f'dRA: {dRA :.1f}  [arcsec], dDec: {dDec :.1f},  ' 
					+ f' PA: {np.rad2deg(PA) :.1f} [deg], inc: {np.rad2deg(np.arccos(b/a)) :.1f} ')
	fits.writeto( data_folder + 'disk03_zz_3000um.fits', data=arr, header=hdr, overwrite=True)


def analytic_sensitivity( t):
	'''
	Analytical formula for the expected point-source sensitivity based on exposure time t. Based on ALMA Handbook (except w factor 0.5 as it does not agree with online tool).
	t in seconds, the resulting sensitivity (rms) is in mJy. At 1mm underestimate the rms, but tclean stops anyway.
	'''
	T_sys = 74.262
	A_eff = 113.1 * 0.715	# [m^2]		(0.71 for band 3 and 0.72 for band 1)
	f = 0		# at DEC = -24°
	sig_ps = 2 * 1.380649e-23 * T_sys / ( 0.96*0.88* A_eff * (1 - f) * np.sqrt( 43 * 42 * 2 * 7.5e9 * t) )	# continuum obs with dual pol assumed
	return sig_ps / 1e-29 		# [mJy]


def prepare_sky_model( filename, data_folder, savedir, damp, monosource, nRMS=1, wle=mm3):
	'''
	Open fits file with desired sky brightness model and cut it / damp it before mock-obs.
	'''
	hdul = fits.open( filename )
	hdul.info()
	hdr = hdul[0].header
	pixscale = hdr['CDELT1']	# deg / pix
	sky_image = hdul[0].data #.byteswap().newbyteorder()   	# byteswap is needed for cv2 blurring

	diskname = filename.replace( data_folder, '' ).replace(f'_{round(wle*1e6)}um', '').strip('.fits')  	# each one a separate folder
	os.system( f'mkdir {savedir}{diskname}')
	os.chdir( savedir + diskname )

	main_beam = np.rad2deg( 1.13 * wle / 12	)		# [deg]	,  0.01618 deg to have a FoV as the main beam at 3mm
	pixcut = int( main_beam / pixscale / 2 )				# margin in pixel
	skycut = crop_image( sky_image, margins=[ pixcut, pixcut])[1:, 1:]		# make it even 

	if damp:
		print('\nArtificially damping the sky beyond the MRS\n')
		npix = skycut.shape[0]
		X = Y = np.arange( -npix + npix/2, npix - npix/2 )
		xx, yy = np.meshgrid( X, Y )	# pixel grid for image filter
		max_scale = 5 / 3600		# arcsec / 3600 = [deg]
		filt_sigma = max_scale / pixscale 		# pixel
		gauss_conv = np.exp( - 0.5* (xx**2 + yy**2) / filt_sigma**2 )		# spatial filter to avoid high fourier powers
		skycut = skycut * gauss_conv	# the central target is isolated with a gaussian kernel
		skycut = np.clip( skycut, a_min=1e-8, a_max=None )	# avoid super low values, amin from d17 bkg patch rms=2e-8 Jy/pix

	if monosource:
		skycut = cancel_extra_sources( skycut, nRMS=nRMS, figure=True)

	hdr['CRPIX1'] = hdr['CRPIX2'] = skycut.shape[0] / 2
	hdr['CRVAL1'] = 246.6175 ; hdr['CRVAL2'] = -24.4017		# so that central pixel corresponds to centre of pointing file 1x
	hdr['CTYPE1'] = 'RA---SIN'; hdr['CTYPE2'] = 'DEC--SIN'; 		# uniform it to CASA products
	fits.writeto( 'skycut.fits', np.float32( skycut ), header=hdr, overwrite=True)
	
	plt.imshow(  np.clip( skycut[:, ::-1 ], a_min=1e-8, a_max=None), 		# 1e-8 Jy/pix should be a fair rms low bound
			origin='lower', norm=mpl.colors.LogNorm(), cmap='inferno')
	plt.axis( 'off' )
	plt.savefig( 'sky_model' + fig_ext, bbox_inches='tight', dpi=300)
	plt.close()


def perform_mock_obs( filename, T_exp, data_folder='', savedir='', ptgfile='', damp=False, monosource=True, nRMS=1, vistab_export=True, wle=mm3, config_name=['']):
	'''
	Call CASA simobserve and simanalyze to produce mock observations of filename.
	'''
	prepare_sky_model( filename=filename, data_folder=data_folder, savedir=savedir, damp=damp, monosource=monosource, nRMS=nRMS, wle=wle)
	image_to_process = 'skycut.fits'		# Import fits file
	ctk.importfits( fitsimage=image_to_process , imagename='skymodel.imag', overwrite=True)
	ctk.imhead( imagename='skymodel.imag', mode='put', hdkey='bunit', hdvalue='Jy/pixel')		# Edit image header

	# Generate synthetic visibilities
	diskname = filename.replace( data_folder, '' ).replace(f'_{round(wle*1e6)}um', '').strip('.fits')  	# each one a separate folder
	for i, config in enumerate(config_name):	# list of configurations, iterate on each
		MSname = f'{diskname}.{config}.noisy.ms'
		os.chdir( '../' )
		try:
			ctk.simobserve( project=diskname ,
				skymodel= f'{diskname}/skymodel.imag' ,
				setpointings= False,  
				ptgfile= ptgfile, 
				incenter= f'{299792458.0/wle}Hz' ,		# v = c / lambda
				inwidth = '7.5GHz' ,
				# mapsize=[ '' ] , # ' ' will fully cover (sky) model
				antennalist= config + '.cfg',
				totaltime= f'{T_exp}s' ,
				thermalnoise= 'tsys-atm',
				user_pwv= 0.7 if wle<2e-3 else 5.186,  # 5.186,      # 5.186 @ 3 & 7mm, 0.7 @ 1mm
				overwrite = True,
				graphics= 'file')
			plt.close()

			# Image and analyze the simulated visibilities
			ctk.simanalyze( project=diskname ,
				vis= f'{diskname}/{MSname}', 
				imsize = [0,0] , 	# [0 ,0] means model image will be matched
				cell= '' , 			# empty string means model cell size is to be used
				niter = 10000,
				interactive = False ,
				threshold = f'{analytic_sensitivity(t=T_exp) :.4f}mJy' ,	#  [25 uJy for 10', 10uJy for 1h ...]
				weighting = 'briggs',
				analyze= False,
				graphics= 'file')
			plt.close()
		
		except Exception as e: 
			print( 'Exception:', e)
			print( '\nSimobserve/analyze already performed, proceeding\n')
		
		os.chdir( diskname )
		if monosource==False:
			xRMS_factor = 20 	# need higher RMS for good extraction in compact config
			extra_sources = copy_extra_sources( MSname, nRMS=nRMS + i*xRMS_factor )
			if np.any( extra_sources[0]):
				print('\n  Subtracting EXTRA SOURCES from MOCK-OBS visibilities!  \n')
				casa_table = cto.table()
				casa_table.open( MSname, nomodify=False )
				if not (casa_table.getcol('DATA') == casa_table.getcol('CORRECTED_DATA') ).all():
					print( '\nThis better NOT be the first run for this target!!\n')
				u, v = casa_table.getcol('UVW')[[0,1]] / wle
				vis_extra = xsrc_to_visib( extra_sources=extra_sources, u=u, v=v )
				corr_data = casa_table.getcol('DATA')				# copy original data
				corr_data[:] = corr_data[:] - vis_extra				# subtracted visibilities broadcasted to correct shape
				casa_table.putcol( 'CORRECTED_DATA', corr_data )	# here for uvtable export
				# casa_table.putcol( 'MODEL_DATA', corr_data )		# here for backup ?
				casa_table.flush() ; casa_table.close()

				ctk.tclean(		# image the subtracted data !
					vis= MSname, imagename=f'./xsrc_sub/{config}_rough', datacolumn='corrected', 
					imsize=extra_sources[0].shape, cell = f'{extra_sources[1]}rad',
					phasecenter='ICRS 16h26m28.2s  -24d24m06.12s', #deconvolver='clark', 
					weighting='briggs', niter=50, nsigma=3, threshold=f'{analytic_sensitivity(t=T_exp) :.4f}mJy'  ) 
				
				casa_table.open( f'./xsrc_sub/{config}_rough.image' )		# the one created above, in [Jy/beam]
				img = casa_table.getcol('map').squeeze().copy( order='F') 		# best model img				
				casa_table.close()
				ptitle = 'Target - xsrc (quick clean)' 
				fig, ax = plt.subplots( figsize=(6,6))  
				ci = ax.imshow( img.T , origin='lower', cmap='inferno', norm=mpl.colors.LogNorm( vmin=1e-6, vmax=None, clip=True) )    # transpose to have as sky model
				ax.set( title=ptitle, ) ; ax.axis( 'off' )
				fig.colorbar( ci, ax=ax, label=r'$I_\nu$ [Jy/beam]')
				fig.savefig( ptitle.replace(' ', '_') + config + fig_ext , bbox_inches='tight', dpi=200)
				plt.close()
				
		if vistab_export:				# export the CASA MS to UV table suited for GALARIO
			casa_table = cto.table()
			casa_table.open( MSname ) #  + '.binned' )
			uvp.io.export_uvtable( f'uvtab_C{config.strip("alma.cycle")}.txt', tb=casa_table, vis=MSname, datacolumn='CORRECTED_DATA') 
			casa_table.close()

	return print( '\nMock observation completed !\n')



def rms( arr ):
	return np.sqrt( np.sum( arr**2 ) / len( arr.flatten() ) )


def min_bkg_rms( image):
	'''Find the bkg patch with the lowest rms as a noise, comparing four quadrants around the centre.'''
	npix = image.shape[0]
	bkg_rms = []
	for i in [1,3]:
		for j in [1,3]:
			bkg_patch = crop_image( image, centre=[npix//4 * i, npix//4 * j], margins=[npix//6, npix//6] )
			bkg_rms.append( rms(bkg_patch) )
			# plt.imshow( bkg_patch, origin='lower', norm=mpl.colors.LogNorm(), cmap='gnuplot2')
			# plt.show()
	return min( min(bkg_rms), rms(image) )


def residuals_vis_plot( MSname, model_vis, T_exp, r_robust=0.2):
	'''
	Calculate the residuals between the visibilities of the mock observations and the bestfit model (galario + multisource).
	'''
	casa_table = cto.table()
	casa_table.open( MSname.replace('.ms', '.image') )
	noisy_img = casa_table.getcol('map').squeeze().copy( order='F') 	# cleaned simanalyze simulation image
	pixscale = abs( casa_table.getkeyword('coords')['direction0']['cdelt'][0])		# [rad/pix] of noisy image
	casa_table.close()
	casa_table.open( MSname, nomodify=False )			# data used in regression in MODEL column
	orig_data = casa_table.getcol('CORRECTED_DATA')		# copy original data
	modeldata = orig_data[:].copy()						# inherit the shape structure
	modeldata[:] = model_vis 			 				# copy model visibilities broadcasted to correct shape
	casa_table.putcol( 'CORRECTED_DATA', modeldata )	# add the fitted model to the MS, here just to be imaged
	casa_table.flush() ; casa_table.close()

	ctk.tclean(		# image the best model !
		vis= MSname,
		imagename='./bestmod/best_model',
		datacolumn='corrected',  		# Use the corrected_data where we stored the model visibilities
		imsize= noisy_img.shape,	
		cell = f'{pixscale}rad',
		phasecenter='ICRS 16h26m28.2s  -24d24m06.12s',
		weighting='briggs', robust=r_robust, #deconvolver='clark', 
		niter=10000, nsigma=1, threshold= f'{ 1* analytic_sensitivity(t=T_exp) :.4f}mJy',
		) 
	
	casa_table.open( './bestmod/best_model.image' )		# the one created above, in [Jy/beam]
	best_img = casa_table.getcol('map').squeeze().copy( order='F') 		# best model img				
	casa_table.close()
	ptitle = 'Bestfit model (obs)' 
	fig, ax = plt.subplots( figsize=(6,6))  
	ci = ax.imshow( best_img.T , origin='lower', cmap='inferno', norm=mpl.colors.LogNorm( vmin=min_bkg_rms( noisy_img ), vmax=None, clip=True) )    # transpose to have as sky model
	ax.set( title=ptitle, ) #, xlabel='au', ylabel='au')
	ax.axis( 'off' )
	fig.colorbar( ci, ax=ax, label=r'$I_\nu$ [Jy/beam]')
	# plt.show()
	fig.savefig( ptitle.replace(' ', '_') + fig_ext , bbox_inches='tight', dpi=300)
	plt.close()

	casa_table.open( MSname, nomodify=False )		# now the RESIDUALS
	casa_table.putcol( 'CORRECTED_DATA', orig_data - model_vis  )
	casa_table.flush() ;	casa_table.close()
	
	ctk.tclean(			# image the residuals !
		vis= MSname,
		imagename='./bestmod/residuals',
		datacolumn='corrected',  			# Use the corrected_data where we stored the residual visibilities
		imsize= noisy_img.shape,			# compare with noisy image
		cell = f'{pixscale}rad',
		weighting='briggs', robust=r_robust, #deconvolver='clark', 
		phasecenter='ICRS 16h26m28.2s  -24d24m06.12s',
		niter=10000, nsigma=1, threshold= f'{ 1* analytic_sensitivity(t=T_exp) :.4f}mJy',
		) 

	casa_table.open( './bestmod/residuals.image' )			# the one created above, in [Jy/beam]
	best_res = casa_table.getcol('map').squeeze().copy( order='F') / min_bkg_rms( noisy_img) 		# cleaned residuals img				
	casa_table.close()
	ptitle = 'Bestfit residuals' 
	fig, ax = plt.subplots( figsize=(6,6))  
	ci = ax.imshow( best_res.T, origin='lower', cmap='RdBu_r', norm=mpl.colors.CenteredNorm( vcenter=0) )    # transpose to have as sky model 
	ax.set( title=ptitle, ) #, xlabel='au', ylabel='au')
	ax.axis( 'off' )
	fig.colorbar( ci, ax=ax, label='RMS units')
	# # plt.show()
	fig.savefig( ptitle.replace(' ', '_') + fig_ext , bbox_inches='tight', dpi=300)
	plt.close()

	casa_table.open( MSname, nomodify=False )
	casa_table.putcol( 'CORRECTED_DATA', orig_data )	# restore the original data at its place
	casa_table.flush() ;	casa_table.close()
	np.save( './bestmod/best_residuals', arr=np.float32(best_res) )	# save to file
	return



def resample_image( image, npix_new, old_pixscale, new_pixscale, order=1): 
	'''
	Resample image to a new pixel scale and scale the flux accordingly.
	'''
	pix_ratio = new_pixscale / old_pixscale			# express the size of the final image pixel in the old pixel scale	
	xx, yy = np.meshgrid( np.arange(npix_new) * pix_ratio, np.arange(npix_new)  * pix_ratio)	# create grid for new sampling		
	delta_c = ( image.shape[0] -1) / 2 - ( xx[0, npix_new//2 ] + xx[0, (npix_new-1)//2 ]) / 2 		# centre on original grid, not on new array pixels  c_old - ( xx[0, nxy//2 ] + xx[0, (nxy-1)//2 ]) / 2
	resampled = snd.map_coordinates( image, coordinates=[ yy + delta_c, xx + delta_c], 
										order=order, cval=0, prefilter=True)		# [Jy/beam]
	flux_rescale = pix_ratio**2		# flux rescaling factor
	return resampled * flux_rescale		# [Jy/pix]


def make_uvplots( MSname, bestfit_arr, galargs, two_comp, uvbin_size=10e3, wle=mm3, make_modelimg=True, save_vis=False, Axes=None ):
	'''
	Produce UVplots for all the bestfit solutions. 
	'''
	bestfit = bestfit_arr[:,0].copy()	# only take the best values (no errors)
	inc, PA, dRA, dDec = bestfit[-4:]
	inc *= deg ; PA *= deg ; dRA *= arcsec ; dDec *= arcsec ;		# convert !
	(diskmod, envmod), chi2, vis_mod = galario_model( pars= bestfit, galargs=galargs, two_comp=two_comp )
	Rmin, dR, nR, nxy, dxy, u, v, Re_obs, Im_obs, w = galargs

	if make_modelimg:
		rot_target = snd.rotate( diskmod + envmod, angle=-PA/deg, reshape=False )   	# correct for PA rotation  
		r_s_target = snd.shift( rot_target, shift=( dDec/dxy, -dRA/dxy ) )  			# shift the model to match the mock obs 
		table = cto.table()
		table.open( MSname.replace('.ms', '') + '.image' )		# only for shape and pixscale
		npix = table.getcol('map').squeeze().shape[0]
		pixscale = abs( table.getkeyword('coords')['direction0']['cdelt'][0])	# [rad/pix]
		table.close()

		bestmod_image = resample_image( r_s_target, npix_new=npix, old_pixscale=dxy, new_pixscale=pixscale )		# resample to the sky data pixel scale
		# hdr = fits.Header({'CTYPE1':'RA---SIN', 'CRVAL1': 246.6175, 'CRPIX1': bestmod_image.shape[1]/2, 'CDELT1': np.rad2deg(pixscale), 'CUNIT':'degree', 
						# 'CTYPE2':'DEC--SIN', 'CRVAL2': -24.4017, 'CRPIX2': bestmod_image.shape[0]/2, 'CDELT2': np.rad2deg(pixscale),
						# 'DXY_orig': dxy, 'DR': dR, 'NR': nR, 'Funit':'[Jy/pix]'})
		# fits.writeto( 'best_model.fits', bestmod_image[:, ::-1], overwrite=True, header=hdr)	# save it like skycut
	else: bestmod_image = [[0,0]]
	
	# Nant = 42 # len( np.unique( table.getcol('ANTENNA')))		# number of antennas used (same for all my runs)
	red_chi2 = chi2/(nR - len(bestfit))			#  chi2/(Nant*(Nant-1)/2 - len(bestfit))
	print( '\ngalario Chi^2: ', chi2, '\n reduced chi2: ', red_chi2 ,'\n\n' )
	# np.savetxt( f'bestfit_chi2.txt', bestfit, footer=f'\n{red_chi2 :.3f} \t (reduced chi2) \n{chi2 :.2f} \t (chi2)')
	
	# ptitle = 'Uvplot' + run_name
	if Axes == None: 
		fig, ax = plt.subplots( 1,1, figsize=(3.5,4), layout='tight')	
	else: 
		ax = Axes
	axins = ax.inset_axes( [0,-0.27 , 1, 0.25] )			# create an inset for the imaginary part

	uv = uvp.UVTable( uvtable=[u*wle, v*wle, Re_obs, Im_obs, w], wle=wle, columns=uvp.COLUMNS_V0 )		# observations uv-plot !
	uv.apply_phase( -dRA, -dDec)         # center the source on the phase center
	uv.deproject( inc=inc/deg, PA=PA/deg, inplace=True)
	uv.uvbin( uvbin_size)
	mask = uv.bin_count != 0 # slice(None)
	uvdist = uv.bin_uvdist[mask] / 1000
	data_dict = {'fmt':'o', 'ms':5, 'color':'k', 'linewidth':0, 'capsize':2, 'ecolor':'gray', 'elinewidth':0.5, 'label':'Data', 'alpha':0.8}
	ax.errorbar( x=uvdist, y=uv.bin_re[mask], yerr=uv.bin_re_err[mask], **data_dict)
	axins.errorbar( x=uvdist, y=uv.bin_im[mask], yerr=uv.bin_im_err[mask], **data_dict)
	del uv
	
	uv_mod = uvp.UVTable( uvtable=[u*wle, v*wle, vis_mod.real, vis_mod.imag, w], wle=wle, columns=uvp.COLUMNS_V0 )	# model uv-plot : disk (+ env)
	uv_mod.apply_phase( -dRA, -dDec)    # center the source on the phase center
	uv_mod.deproject( inc=inc/deg, PA=PA/deg, inplace=True)
	# uv_mod.plot( axes=axes, linestyle='-', color='r', alpha=0.9, label='Total model', yerr=False, uvbin_size=uvbin_size)
	uv_mod.uvbin( uvbin_size ) ; mask = uv_mod.bin_count != 0
	uvdist = uv_mod.bin_uvdist[mask] / 1000 
	model_dict = { 'ls':'-', 'color':'r', 'linewidth':1.8, 'label':'Model', 'alpha':0.95}	
	ax.errorbar( uvdist, uv_mod.bin_re[mask], **model_dict)
	axins.errorbar( uvdist, uv_mod.bin_im[mask], **model_dict)
	ax.text( x=0.95, y=0.95, s= fr'$\chi^2_\nu$={red_chi2 :.3f}', ha='right', va='center', transform=ax.transAxes, color='gray', fontsize=9, alpha=1.)
	del uv_mod

	if two_comp:
		os.system( 'rm visib_disk+env.npy' )				# remove it if it exists already
		for i, comp in enumerate([diskmod, envmod]):		# separately plot disk and envelope contributions
			colors, labs, lls = ['tab:blue', 'tab:green'], ['disk','envelope'], ['--',':']
			comp_vis = gd.sampleImage( comp, dxy, u, v, PA=PA, dRA=dRA, dDec=dDec, check=False, origin='lower')	
			uv_mod = uvp.UVTable( uvtable=[u*wle, v*wle, comp_vis.real, comp_vis.imag, w], wle=wle, columns=uvp.COLUMNS_V0 )
			if save_vis: 	
				with open('visib_disk+env.npy', 'ab') as f:		# this requires two separate np.load calls to read back the arrays
					np.save( f, arr=comp_vis )
			uv_mod.apply_phase( -dRA, -dDec)     	# center on the phase center
			uv_mod.deproject( inc=inc/deg, PA=PA/deg, inplace=True)
			uv_mod.uvbin( uvbin_size ) ; mask = uv_mod.bin_count != 0
			uvdist = uv_mod.bin_uvdist[mask] / 1000 
			comp_dict = { 'ls':lls[i], 'color':colors[i], 'lw':1.5, 'label':labs[i], 'alpha':0.92}
			ax.errorbar( uvdist, uv_mod.bin_re[mask],  **comp_dict)
			axins.errorbar( uvdist, uv_mod.bin_im[mask], **comp_dict)
			# uv_mod.plot( axes=(ax, axins), linestyle='--', color=colors[i], alpha=0.9, linewidth='1.5', label=labs[i], yerr=False, uvbin_size=uvbin_size, fontsize=10)
			# ax.yaxis.set_label_coords(-0.1, 0.5) ; axins.yaxis.set_label_coords(-0.1, 0.5)	# get uvplot labels closer

	ax.set( ylabel='Re(V) [Jy]', xscale='log', yscale='log') ; ax.legend( loc='best', bbox_to_anchor=(0, 0, 0.9, 0.9), fontsize=8)
	axins.set( ylabel='Im(V) [Jy]', xscale='log', xlabel='uvdistance [k$\mathrm{\lambda}$]' )
	if ax.get_ylim()[0] < 1e-4: ax.set( ylim=[1e-4, ax.get_ylim()[1]] )		# force lower ylim at 1e-5
	if Axes != None: 
		return ax
	else: 
		fig.savefig( 'uvplot_log' + fig_ext, dpi=200, bbox_inches='tight')
	plt.close()
	return bestmod_image, vis_mod


def pentaplot( diskname, MSname, bestfit_pars, galargs, two_comp, wle, run_name, as_margin=3., rulersize=100):
	'''
	Just plot together in a nice cut four panels about a target: sky model, mock-obs, model mock obs, residuals
	'''
	hdul = fits.open( 'skycut.fits' )			# load sky model 
	pixscale_s = hdul[0].header['CDELT1']		# [deg / pix]
	sky_image = hdul[0].data #.byteswap().newbyteorder() 
	pixcut = int( as_margin / (pixscale_s * 3600) )			# margin in pixel
	skycut = crop_image( sky_image, margins=[ pixcut, pixcut])[:, ::-1 ] * 1000		# [mJy/pix] 
	xc, yc = np.array( skycut.shape ) / 2

	ct = cto.table()
	ct.open( f"{MSname.replace('.ms', '')}.image" )		# cleaned simanalyze simulation image
	noisy_img = ct.getcol('map').squeeze().copy( order='F').T
	pixscale_m = np.rad2deg( abs( ct.getkeyword('coords')['direction0']['cdelt'][0]) ) * 3600		# [arcsec/pix]
	beam_dict = ct.getkeyword('imageinfo')['restoringbeam']						# a, b and PA of beam [", ", deg]
	bmaj = beam_dict['major']['value'] / pixscale_m 
	bmin = beam_dict['minor']['value'] / pixscale_m; PA = beam_dict['positionangle']['value']	

	ct.open( './bestmod/best_model.image' )						# [Jy/beam]
	best_model = ct.getcol('map').squeeze().copy( order='F').T 	# best model clean img
	ct.close()
	best_res = np.load( './bestmod/best_residuals.npy' ).T 				# cleaned residuals img
	rms = min_bkg_rms( noisy_img )
	modlist = [noisy_img, best_res, best_model]
	pixcut_m = int( as_margin / pixscale_m )		# margin in pixel
	for i in range(len(modlist)):
		modlist[i] = crop_image( modlist[i], margins=[ pixcut_m, pixcut_m])

	ptitles = ['Simulated sky', 'Observation', 'Residuals', 'Model']
	units = ['mJy/pix', 'Jy/beam', 'RMS units', 'Jy/beam']
	fig, axes = plt.subplots( 2,2, figsize=(6.5,6.5), layout='constrained' ) #; axes[1,2].set_visible(False)		# hide unused axes
	axs = axes[0:2,0:2].flatten()
	simc = axs[0].imshow( skycut,     origin='lower', norm=mpl.colors.LogNorm(), cmap='inferno')
	obsc = axs[1].imshow( modlist[0], origin='lower', norm=mpl.colors.LogNorm( vmin=rms, clip=True), cmap='inferno')
	resc = axs[2].imshow( modlist[1], origin='lower', norm=mpl.colors.CenteredNorm( vcenter=0), cmap='RdBu_r')
	modc = axs[3].imshow( modlist[2], origin='lower', norm=mpl.colors.LogNorm( vmin=rms, clip=True), cmap='inferno')
	rls_pix = rulersize * 1/140 / (pixscale_s *3600)		# [au * arcsec/au * pix/arcsec]
	axs[0].plot( [xc - 0.5*rls_pix, xc + 0.5*rls_pix], (yc - 0.9*pixcut )*np.array([1,1]), c='w', lw=2, alpha=.9)		# ruler patch
	axs[0].text( xc-0.8*rls_pix , yc-0.9*pixcut, s=f'{rulersize :3.0f} au', color='w', ha='right', va='center', alpha=.8, fontsize=8) 
	axs[0].text( xc+0.8*rls_pix , yc-0.9*pixcut, s=f'{rulersize/140 :0.3f}"', color='w', ha='left', va='center', alpha=.8, fontsize=8)
	axs[1].text( x=0.05, y=0.9, s= f'RMS={rms :1.1e} Jy/beam', ha='left', va='center', transform=axs[1].transAxes, color='gray', fontsize=8, alpha=1)
	
	for i, cb in enumerate([simc, obsc, resc, modc]):
		fig.colorbar( cb, ax=axs[i], shrink=0.8, pad=0.00, label=units[i])
		if i > 0: 
			width_frac = bmaj / modlist[i-1].shape[1] ; height_frac = bmin / modlist[i-1].shape[0]
			beam_patch = mpl.patches.Ellipse( (0.1, 0.1), width=width_frac, height=height_frac, angle=90 + PA, 
						transform=axs[i].transAxes, facecolor='gray', edgecolor='gray', linewidth=1, alpha=1 )
			axs[i].add_patch( beam_patch )
		axs[i].set( title= ptitles[i], aspect='equal')	#, fontsize=9)
		# axs[i].axis('off')
		axs[i].tick_params(axis='both', left=False, top=False, right=False, bottom=False, labelleft=False, labeltop=False, labelright=False, labelbottom=False)

	uvax = fig.add_axes( rect=[1.12, 0.2, 1/2.8, 0.35])		# add an axes for the uvplot
	uvax = make_uvplots( MSname, bestfit_pars, galargs, two_comp, 20e3, wle, make_modelimg=False, Axes=uvax)
	fig.suptitle( diskname + '-' + run_name, fontweight='bold' ) 
	# plt.show()
	fig.savefig( f'pentaplot_{diskname}' + '-' + run_name.replace(' ', '_') + fig_ext , bbox_inches='tight', dpi=300)
	plt.close()


def bestfit_plots( diskname, T_exp, galargs=None, two_comp=True, sampler=None, nRMS=3, burnin=None, walksigma=3, wle=mm3, savedir='', config_name=[]):
	'''
	Produce MCMC plots (chains + corner), UVplot, best model and residual visib images for best solution.
	'''
	os.chdir( savedir + diskname )
	labels_gauss = ['Log($I_0$)', '$\sigma$', '$i$', 'PA', 'dRA', 'dDec']
	labels_2c = [r'Log($I_{0d}$)', r'Log($I_{0e}$)', '$\sigma$', 'R_i', 'R_out/Ri', 'p_idx', '$i$', 'PA', 'dRA', 'dDec']
	labs_mc = labels_2c if two_comp else labels_gauss

	if len(config_name) > 1:
		print( '\nCreating concatenated MS, just for bestfit plots\n')
		MSname = f'{diskname}.concat.noisy.ms'
		if os.path.exists( MSname ):
			print( 'Concat MS already existing!\n')
		else:
			ctk.concat( vis=[ f'{diskname}.{config_name[0]}.noisy.ms', f'{diskname}.{config_name[1]}.noisy.ms'], concatvis= MSname)	
			os.chdir('../')
			print('simanalyzing\n')
			ctk.simanalyze( project=diskname ,
					vis= f'{diskname}/{MSname}', 
					niter = 10000,		# auto cell and imgsize values
					threshold = f'{analytic_sensitivity(t=T_exp) :.4f}mJy' ,	#  [25 uJy for 10', 10uJy for 1h ...]
					weighting = 'briggs', analyze= False, graphics= 'file')
			print( 'printing concat uvtab')
			os.chdir(diskname)
			casa_table = cto.table()
			casa_table.open( MSname ) 
			uvp.io.export_uvtable( f'uvtab_Cconcat.txt', tb=casa_table, vis=MSname, datacolumn='CORRECTED_DATA') 
			casa_table.close()
	else: 
		MSname = f'{diskname}.{config_name[0]}.noisy.ms'

	if sampler is None:
		sampler = emcee.backends.HDFBackend( f'{diskname}__sampler.h5', read_only=True )	# will throw store==True error if diskname is wrong
	nsteps = sampler.get_chain().shape[0]
	if burnin is None:
		burnin = nsteps//(4*2)		# 4 is the thinning factor in the mcmc run
	bestfit = mcmc_plots( sampler, labels=labs_mc, burn_in=burnin, walk_clip_thresh=walksigma, figures=False )
	np.savetxt( f'bestfit_params.txt', bestfit )		# save a (Npar, 3) table with the columns being: best value, 16p, 84p
	# bestfit = np.loadtxt('bestfit_params.txt')
	
	if galargs is None:
		config = 'concat' if len(config_name) > 1 else config_name[0]
		galargs = get_galargs( wle=wle, config_name=config)
	# copy_extra_sources( MSname, nRMS)
	model_image, mod_vis = make_uvplots( MSname, bestfit, galargs, two_comp=two_comp, wle=wle, save_vis=True )
	residuals_vis_plot( MSname, mod_vis, T_exp )
	run_name = f'{round(wle*1e3)}mm_' + os.path.basename( savedir[:-1] ).replace('run_', '').replace('_xsrc', '')
	pentaplot( diskname, MSname, bestfit, galargs, two_comp, wle, run_name)

	# # best model visual check
	plot_img = np.clip( crop_image( model_image, margins=[500, 500]), a_min= 1e-6, a_max=None)		# [:, ::-1]
	plt.imshow( plot_img, origin='lower', norm=mpl.colors.LogNorm(), cmap='inferno')	# slicing to have it mirrored as casa
	plt.title('galario best model')
	plt.axis(False)
	plt.savefig( f'galario_sky-model_bestfit' + fig_ext, bbox_inches='tight', dpi=200)
	plt.close()
	return print( '\n Best-fit plots and images saved.\n')



def get_galargs( wle, config_name):
	'''Read uvtabs and get required parameters for galario.'''
	uvtab_name = f'uvtab_C{config_name.replace("alma.cycle", "")}.txt'
	u, v, Re_obs, Im_obs, w = np.require( np.loadtxt( uvtab_name, unpack=True), requirements='C')
	u /= wle
	v /= wle	# have the baselines in lambda units
	nxy, dxy = gd.get_image_size( u, v, verbose=False) # , PB=1.13*wle/12 )		# number and size of pixel in radians
	# radial grid parameters
	Rmin = 0  			# [arcsec]
	Rmax = Rmax_model	# [arcsec]
	dR = np.rad2deg(dxy) * 3600 / 11   	# [arcsec]
	nR = int( Rmax / dR )			# dR per nr dnon deve superare il raggio massimo del modello,  3*MRS
	return [Rmin, dR, nR, nxy, dxy, u, v, Re_obs, Im_obs, w]


def mcmc_regress( diskname, T_exp, nsteps, two_components=True, Ncpu=None, savedir='', nRMS=10, wle=mm3, config_name=[]):
	'''
	Main pipeline for fitting YSO models with galario to a sky model (filename).
	'''
	os.chdir( savedir + diskname )
	galargs = [get_galargs( wle=wle, config_name=config) for config in config_name]		# one or two if SC or CC

	# parameter space domain
	p_ranges_2c = np.array([[7.8, 13],	# Log10( I0disk )	[Log(Jy/sr)]
						[7.8, 13.],		# Log10( IOenvelope)   
						[1e-2, .8],		# sigma i.e. sma [arcsec]
						[1e-2, 1.6],	# Ri [arcsec] (Rmax= 8 / 5 = 1.6, to avoid an envelope cut at high fluxes)
						[5, 1000],		# Rout/Ri [arcsec] fraction of Ri		# [3e-4, 8]
						[1.3, 2.99],	# p_index []
						[-5., 95.],		# inc (deg)
						[-7, 180.],		# PA (deg)
						[-2, 2],		# dRa (arcsec)
						[-2, 2]])		# dDec (arcsec)

	p_ranges_gauss = np.array([[8, 15],	# Log10( I0 )	[Log(Jy/sr)]
						# [0.2, 0.9],	# Hr0
						[1e-5, .8],		# sigma i.e. sma 	[arcsec]
						[-5., 95.],		# inc (deg)
						[-7, 180.],		# PA (deg)
						[-2, 2],		# dRa (arcsec)
						[-2, 2]])		# dDec (arcsec)

	# initial guess for the parameters
	p0_2c = np.array([11, 8.4, 0.2, 0.3, 6, 2.5, 80., 45., 0., 0.]) 	# Log(I0), Log(Ienv), sma, Rin, Rout/Ri, p_idx, (inc, PA, dRA, dDec)
	p0_gauss = np.array([12, 0.2, 80., 45., 0., 0.])				# Log(I0), sma, inc, PA, dRA, dDec
	if two_components:
		p0_mc = p0_2c
		p_rang_mc = p_ranges_2c
	else:
		p0_mc = p0_gauss
		p_rang_mc = p_ranges_gauss

	# execute the MCMC
	sampled = mcmc_run( galargs=galargs, p0= p0_mc, p_ranges= p_rang_mc, 
			nsteps=nsteps, nwalkers=Nwalkers, nthreads=Ncpu, backend_fname=f'{diskname}__sampler', 
			two_comp=two_components, append=False )
	
	# bestfit_plots( diskname, T_exp, None, two_components, sampled, nRMS, wle=wle, savedir=savedir, config_name=config_name)

