# # # Full pipeline to mock obs + galario fit. Gauss model with additional constant.

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
from scipy.optimize import curve_fit
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops, regionprops_table
from skimage.morphology import closing, footprints
# os.environ["OMP_NUM_THREADS"] = "1"



def crop_image( img, centre=None, margins=[100, 100] ):
	'''Select a subimage of margins pixels around the centre (odd size).'''
	if centre is None:      	# use the middle of the image
		centre = (np.array( img.shape)/2 ).astype(int)
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


def galario_fit( pars, galargs, two_comp=True, extra_sources=[0,0]):
	'''
	Let galario generate a model on the visibilities and return a Chi2 to the data.
	pars: LI0d, (LI0e), sigma, (Ri, p_index,) inc, PA, dRA, dDec  , (): if two_comp is True
	galargs:  Rmin, dR, nR, nxy, dxy, u, v, Re, Im, w
	two_comp: whether to use the only a gaussian (False) or gauss + plummer (True)
	extra_sources: ([model array of non-central sources], pixscale ) 
	returns: target_model (only target), chi2, vis_model (target + extra sources)
	'''
	Rmin, dR, nR, nxy, dxy, u, v, Re, Im, w = galargs
	if two_comp:
		LI0d, LI0e, sigma, Ri, p_index, inc, PA, dRA, dDec = pars
		Ri *= arcsec
		Rout = (Rmin + nR*dR) * arcsec
	else: 
		LI0d, LI0e, sigma, inc, PA, dRA, dDec = pars			# unpack the parameters
		# f = 1
	I0d = 10.**LI0d ; I0e = 10.**LI0e       # convert from log to real space

	# convert everything to radians
	sigma *= arcsec; Rmin *= arcsec; dR *= arcsec
	inc *= deg; PA *= deg ; dRA *= arcsec ; dDec *= arcsec
	R = np.linspace( Rmin, Rmin + dR * nR, nR, endpoint=False)		# radial grid

	I_disk = GaussianProfile( R, I0d, sigma=sigma)		# Brightness profile of YSO
	if two_comp: 
		I_env = Plummer_envelope( R, I0e, Ri, Rout, p_index)	# Brightness profile of the envelope 
		target_model = gd.sweep( I_disk, Rmin, dR, nxy, dxy, inc=inc) + gd.sweep( I_env, Rmin, dR, nxy, dxy, inc=0) 	# [Jy/pix]
	else: 
		# model = I_disk # + a		# a is an additional constant for background
		target_model = gd.sweep( I_disk, Rmin, dR, nxy, dxy, inc=inc) + gd.sweep( np.full_like(I_disk, I0e), Rmin, dR, nxy, dxy, inc=0)	# sphere is isotropic

	# Compute visibilities for the central target that requires rotation and offsets (with PA, inc, dRA, dDec)
	vis_target = gd.sampleImage( target_model, dxy, u, v, PA=PA, dRA=dRA, dDec=dDec, check=False)

	# Compute visibilities for the extra sources (no rotation/offset)
	if np.any( extra_sources[0] ):
		vis_extra = gd.sampleImage( extra_sources[0], extra_sources[1], u, v, PA=0, dRA=0, dDec=0, origin='lower', check=False)	# different pixscale than target model!
		vis_model = vis_target + vis_extra
	else:
		vis_model = vis_target
	
	# Compute chi2 in visibility space (should be equivalent to galario reduce_chi2())
	# chi2 = gd.chi2Image( model, dxy, u, v, Re, Im, w, PA=PA, dRA=dRA, dDec=dDec )
	chi2 = np.sum( w * ((vis_model.real - Re)**2 + (vis_model.imag - Im)**2) )
	# model_image = targ+ + extra_sources_model	# for visualization [Jy/pix]
	return target_model, chi2, vis_model


def copy_extra_sources( diskname, nRMS=1 ):
	'''
	Create a copy of CASA noisy image for the areas above noise and put everything else (including central target) to zero.
	'''
	table = cto.table()
	table.open( f'{diskname}.{config_name}.noisy.image' )		# noisy image with primary beam correction (otherwise damped twice)
	noisy_img = table.getcol('map').squeeze().copy( order='F').T 			# copy simanalyze noisy image (convolved)  [Jy/beam]
	beam_dict = table.getkeyword('imageinfo')['restoringbeam']	# a, b and PA of beam
	img_pixscale = abs( table.getkeyword('coords')['direction0']['cdelt'][0])		# [rad/pix] of noisy image
	table.open( f'{diskname}.{config_name}.noisy.model' )		
	deconvolved = table.getcol('map').squeeze().copy( order='F').T		# deconvolved model image of the sky [Jy/pix]
	table.close()

	## apply threshold to identify the sources on the convolved image
	thresh = nRMS * rms( noisy_img ) 		# min_bkg_rms( noisy_img )
	bw = closing( noisy_img > thresh, footprints.rectangle(3, 3) )
	cleared = clear_border( bw )		# remove artifacts connected to image border
	label_image = label( cleared )		# label image regions
	nimg_masked = np.where( noisy_img > thresh, deconvolved, 0)		# keep everything above n*RMS
	
	sources_df = pd.DataFrame( regionprops_table( label_image,
		properties=('centroid', 'orientation', 'axis_major_length', 'axis_minor_length', 'equivalent_diameter_area'), ) ).rename(
			columns={'centroid-0':'y0', 'centroid-1':'x0', 'orientation':'PA', 'axis_major_length':'a', 'axis_minor_length':'b', 
				'equivalent_diameter_area':'diam'} )
	
	## exclude the central source !
	target_idx = ((sources_df[['y0','x0']] - np.array(noisy_img.shape)/2 )**2 ).sum( axis=1).idxmin()	# central source (target)
	miny, minx, maxy, maxx = regionprops( label_image )[target_idx].bbox		# rectangle over central source
	nimg_masked[ miny:maxy , minx:maxx] = 0			# zeros on the entire target rectangle (envelopes should be safe then)

	if (nimg_masked > 0).any():	
		fig, ax = plt.subplots( figsize=(8, 8))		# diagnostic figure
		diag_img = np.where( noisy_img > thresh, noisy_img, np.nan)
		diag_img[ miny:maxy , minx:maxx] = np.nan
		ax.imshow( diag_img	, origin='lower', norm=mpl.colors.LogNorm() )	# use noisy_img just for diagnostic plot
		ax.set_axis_off()
		fig.savefig( 'multi-source_map' + fig_ext, bbox_inches='tight', dpi=600)
		plt.close()
	else: 
		print( '\nNo extra sources found in the image!\n' )
		return 0, img_pixscale

	# beam_area = np.pi * beam_dict['major']['value'] * beam_dict['minor']['value'] / (4*np.log(2))	# FWHM ellipse area [arcsec^2/beam]
	# beam_to_pix = ( 3600* np.rad2deg( img_pixscale ) )**2  / beam_area		# to convert the flux from [Jy/beam] to [Jy/pix]
	nimg_masked[ nimg_masked <= 1e-12 ] = 1e-12			# remove negative values
	return nimg_masked *1, img_pixscale		# [Jy/pix], [rad/pix]		maybe a DECONVOLUTION would be better here?


def calc_beam_factor( xsrc, gal_mod, dxy):

	xtra = xsrc.copy()
	xtra[ xtra <= 1e-10 ] = np.nan
	
	bestfit = np.loadtxt( 'bestfit_params.txt' )		# load the best fit parameters from the file
	# compute the visibilities of the bestfit model
	inc, PA, dRA, dDec = bestfit[-4:]
	dRA *= arcsec ; dDec *= arcsec ;		# convert to [rad]!

	rot_target = snd.rotate( gal_mod, angle=-PA, reshape=False )   	# correct for PA rotation y 
	galmod_align = snd.shift( rot_target, shift=( -dDec/dxy, -dRA/dxy ) ) 
	galmod_nan = galmod_align.copy()
	galmod_nan[ np.isnan(xtra)] = np.nan
	ratio = galmod_align / xtra
	integ_ratio = np.nansum( galmod_nan ) / np.nansum( xtra)	# sum of the galario model 
	return np.nanmean( ratio )		# mean ratio of the source in the galario model to the CASA noisy image


def locate_multi_sources( diskname ):
	'''
	Determine the number, position and approximate size of multiple sources in the noisy images.
	'''
	os.chdir( diskname )
	noisy_tab = f'{diskname}.{config_name}.noisy.image'
	table = cto.table()
	table.open( noisy_tab)
	noisy_img = table.getcol('map').squeeze().copy().T 			# simanalyze noisy image
	beam_dict = table.getkeyword('imageinfo')['restoringbeam']	# a, b and PA of beam
	img_pixscale = table.getkeyword('coords')['direction0']['cdelt'][0]		# [rad/pix] of noisy image
	table.close()

	# apply threshold to identify the sources
	thresh = 5 * rms( noisy_img ) 		# threshold_otsu( noisy_img )
	bw = closing( noisy_img > thresh, footprints.rectangle(3, 3) )
	cleared = clear_border( bw )		# remove artifacts connected to image border
	label_image = label( cleared )		# label image regions
	# image_label_overlay = label2rgb( label_image, image=noisy_img, bg_label=0 )

	fig, ax = plt.subplots( figsize=(8, 8), tight_layout=True)
	ax.imshow( np.clip( noisy_img, a_min=1e-10, a_max=None ), origin='lower', norm=mpl.colors.LogNorm() )

	for region in regionprops( label_image ):
		if region.area >= 5:				# take regions with large enough areas
			minr, minc, maxr, maxc = region.bbox			# draw rectangle around segmented coins
			rect = mpl.patches.Rectangle( (minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='r', linewidth=1, )
			ax.add_patch(rect)

	ax.set_axis_off()
	fig.savefig( 'multi-source_map' + fig_ext, bbox_inches='tight')
	plt.close()
	os.chdir('..')

	props = pd.DataFrame( regionprops_table( label_image,
			properties=('centroid', 'orientation', 'axis_major_length', 'axis_minor_length', 'equivalent_diameter_area'), ) ).rename(
			 	columns={'centroid-0':'y0', 'centroid-1':'x0', 'orientation':'PA', 'axis_major_length':'a', 'axis_minor_length':'b', 
			  		'equivalent_diameter_area':'diam'} )
	return noisy_img, props, beam_dict, img_pixscale


def twoD_Gaussian( img_size, I0, xo, yo, sigma_x, sigma_y, theta, offset ):
	x, y = np.meshgrid( np.arange(0, img_size[0]), np.arange(0, img_size[1]) )
	xo = float(xo)
	yo = float(yo)    
	a = np.cos(theta)**2 /(2*sigma_x**2) + np.sin(theta)**2 /(2*sigma_y**2)
	b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
	c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
	g = offset + I0 * np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2)))
	return g.ravel()


def fit_extra_sources( diskname, figures=True):
	'''
	Fit 2D gaussians to the non-central sources in the FoV. One source per time in its own image cut. 
	'''
	image, sources_df, beam_info, pix_to_sr = locate_multi_sources( diskname=diskname)		# noisy img and params of the bright regions
	# find the source closest to centre to exclude it from the fitting 
	target_idx = ((sources_df[['y0','x0']] - np.array(image.shape)/2 )**2 ).sum( axis=1).idxmin()	# central source (target)
	sources_tofit = sources_df.drop( target_idx )
	
	param_bounds = ( [1e-5, -500, -500,  0.,   0., -np.pi/2, 0  ], 		# limits on parameters: I0, x0, y0, sig_x, sig_y, PA, offset
					 [1e2, +500, +500,  200,  200, +np.pi/2, 1] )      	# units : 	Jy/beam, pix, pix, pix, pix, rad, Jy/beam
	fitparams = []
	for s in range( len( sources_tofit)):		# iterate on the regions rows
		src = sources_df.iloc[s]
		print( np.rad2deg( src.PA ) ) 
		image_cut = crop_image( image.copy(), centre=src[['y0','x0']].values.astype(int), 
						 margins=[int(1.6* src.diam), int(1.6*src.diam)] )		# cut with 2x the equiv area diameter
		init_guess = [ 1e-3, image_cut.shape[0]/2, image_cut.shape[1]/2, src.a, src.b, src.PA, 1e-5]		# starting guess
		# print( init_guess)
		popt, pcov = curve_fit( twoD_Gaussian, xdata=image_cut.shape, ydata=image_cut.ravel(), p0=init_guess, bounds=param_bounds, absolute_sigma=False )
		fit_stds = np.sqrt(np.diag( pcov ))          # from scipy doc
		print( r'fit: I0= %.3e , x0= %.3f , y0= %.3f, $\sigma_x$= %.3f , $\sigma_y$= %.3f , PA= %.2f, offset= %.2e' % tuple(popt) )
		print( 'Fit 1 sigma errors:', fit_stds ) 
		if figures:
			plt.imshow( image_cut, origin='lower')#, norm=mpl.colors.LogNorm() )
			plt.contour( twoD_Gaussian( image_cut.shape, *popt).reshape(image_cut.shape), cmap='inferno_r')
			plt.show()

		Dy, Dx = popt[[2,1]] - np.array(image_cut.shape)/2		# delta [pixel] from fitted centre to image cut centre
		popt[[2,1]] = Dy, Dx
		beam_area = np.pi * beam_info['major']['value'] * beam_info['minor']['value']	# ellipse area [arcsec^2]
		beam_to_sr = 1 / beam_area		# to convert the flux peak I0 from [Jy/beam] to [Jy/arcsec2]
	
		to_galario = np.array([ beam_to_sr, pix_to_sr, pix_to_sr, pix_to_sr, pix_to_sr, 1, beam_to_sr ])
		fitparams.append( popt * to_galario )
	return np.array(fitparams).reshape( len(sources_tofit), -1)



#  my MCMC functions

def log_likelihood( pars, galargs, two_comp, xsrc): 
	'''Galario fit chi2 likelihood function'''
	chi2 = galario_fit( pars=pars, galargs=galargs, two_comp=two_comp, extra_sources=xsrc )[1]
	return -0.5 * chi2

def log_prior( pars, p_ranges, two_comp): 
	''' prior dist. pars is the array of free parameters, p_ranges their boundaries'''
	if (p_ranges[:, 0] < pars).all() and (pars < p_ranges[:, 1]).all():
		if two_comp == True:
			if (pars[2] <= pars[3]): #  and (pars[3] < pars[4]):		# impose that sigma < Ri ###< Rout
				return 0.0
			else: return -np.inf
		else:
			if (pars[1] <= pars[0]): 				# impose that Idisk > Ienv
				return 0.0
			else: return -np.inf	
	else:	
		return -np.inf

def log_probability( pars, p_ranges, galargs, two_comp, xsrc):
	logprior = log_prior( pars=pars, p_ranges=p_ranges, two_comp=two_comp)
	if not np.isfinite( logprior):
		return -np.inf
	return logprior + log_likelihood( pars, galargs, two_comp, xsrc)


def mcmc_run( galargs, p0, p_ranges, nsteps=2000, nwalkers=40, nthreads=10, two_comp=False, backend_fname='last_sampler', append=False, extra_src=[0,0]):
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
	
	sampler = emcee.EnsembleSampler( nwalkers, ndim, log_probability, args=(p_ranges, galargs, two_comp, extra_src), 
							# threads=nthreads, 
							backend=bknd_samp, # live_dangerously=False, 
			moves=[ (emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2),], 	# mv1
			# moves=[ (emcee.moves.StretchMove(), 0.5), (emcee.moves.DEMove(), 0.5),], 		# mv2
			# moves = emcee.moves.KDEMove(), 	# mv3	
			)
	
	# state = sampler.run_mcmc( startpos, 100, progress=progbar, store=False)		# pre-run for hard burn-in
	# new_p0 = np.quantile( state.coords,  0.50, axis=0) + 1e-2* np.random.randn( nwalkers, ndim)
	# sampler.reset()
	sampler.run_mcmc( startpos, nsteps, progress=progbar, store=True)			# full production run
	return sampler




def clip_chains( samples, thresh=5):
	'''
	Discard the walkers that are more than thresh sigma away from the median.
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


def mcmc_plots( samp_bkend, labels, burn_in, walk_clip_thresh=5, figures=True, folder='./'):
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

	samples = samp_bkend.get_chain()
	flat_samples = samp_bkend.get_chain( discard=int(burn_in),  flat=True)
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
		label_kwargs={'labelpad':20, 'fontsize':0}, fontsize=8,
		title_kwargs={"fontsize": 10, 'loc':'left'},	
		)
	cornfig.savefig( folder + 'corner_plot' + fig_ext, bbox_inches='tight')
	if figures: plt.show()
	plt.close()

	# best parameters from the walker step with lowest chi2
	# best_idx = np.unravel_index( samp_bkend.get_log_prob().argmin(), samp_bkend.get_log_prob().shape )
	# best_pars = samp_bkend.get_chain()[best_idx]
	best_pars = np.percentile( flat_samples,  50, axis=0)     # best params out of fit
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
	sky_masked = np.where( sky_image > thresh, noise_lev, sky_image)		# keep everything below n*RMS
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
		ax.imshow( diag_img, origin='lower', norm=mpl.colors.LogNorm() )	# use noisy_img just for diagnostic plot
		ax.set_axis_off()
		plt.show()
		fig.savefig( 'levelled_sky' + fig_ext, bbox_inches='tight', dpi=200)
		plt.close()

	return sky_masked


def generate_skymodel( like_filename, data_folder='/Users/gcolumba/PostDoc_Mac/PostProc/simulations/3mm/' ):
	'''Create a sky model with a target and some extra sources for mock observations.
	like_filename: a fits file (path) with the sky model to be used as a template.'''
	hdul = fits.open( data_folder + like_filename )
	hdul.info()
	hdr = hdul[0].header
	pixscale = hdr['CDELT1']	# deg / pix
	modelshape = hdul[0].data.shape
	arr = np.zeros( shape= modelshape, dtype=np.float32)		# 4895x4895 pixels, as the 3mm sky models
	
	# create a gaussian target with an offset
	dDec, dRA = -0 / (pixscale*3600), 0 / (pixscale*3600)		# [arcsec] to pixel offset
	target = twoD_Gaussian( img_size=arr.shape, I0=9e-3, xo=arr.shape[0]//2 + dRA, yo=arr.shape[1]//2 + dDec,
		sigma_x=5, sigma_y=15, theta=np.pi/6, offset=1e-10)		# [Jy/pix]
	arr += target.reshape( arr.shape )		# add the target to the array

	# # then add other sources along a vertical line from the target to the top of the image with equal spacing and brightness
	# nsources = 5		# number of sources to add
	# for i in range( 1, nsources + 1):
	# 	dDec = modelshape[0]/6 / (nsources + 1) * i		# pixel offsets
	# 	source = twoD_Gaussian( img_size=arr.shape, I0=5e-4, xo=arr.shape[0]/2 + 0, yo=arr.shape[1]/2 + dDec,
	# 		sigma_x=5, sigma_y=5, theta=0, offset=0)		# [Jy/pix]
	# 	arr += source.reshape( arr.shape )		# add the target to the array
	
	plt.imshow( arr, origin='lower', norm=mpl.colors.LogNorm(), cmap='gnuplot2')
	plt.colorbar( label='Jy/pix')
	plt.show()
	# save the model to fits
	hdr['CTYPE1'] = 'RA---SIN'		# set the header keywords
	hdr['CTYPE2'] = 'DEC--SIN'
	fits.writeto( data_folder + 'disk04test00_xy_3000um.fits', data=arr, header=hdr, overwrite=True)


def analytic_sens( t, a=653.835, b=0.5, c=-0.02):
	'''
	Analytical formula for the expected rms based on exposure time t. Defaults fitted from ALMA calculator.
	t in seconds, the resulting sensitivity (rms) is in uJy.
	'''
	return a * t**(-b) + c


def prepare_sky_model( filename, data_folder, savedir, damp, monosource):
	'''
	Open fits file with desired sky brightness model and cut it / damp it before mock-obs.
	'''
	hdul = fits.open( filename )
	hdul.info()
	hdr = hdul[0].header
	pixscale = hdr['CDELT1']	# deg / pix
	sky_image = hdul[0].data #.byteswap().newbyteorder()   	# byteswap is needed for cv2 blurring

	diskname = filename.replace( data_folder, '' ).replace('_3000um', '').strip('.fits')  	# each one a separate folder
	os.system( f'mkdir {savedir}{diskname}')
	os.chdir( savedir + diskname )

	main_beam = np.rad2deg( 1.13 * wle / 12	)		# [deg]	,  0.01618 deg to have a FoV as the main beam at 3mm
	pixcut = int( main_beam / pixscale / 2 )				# margin in pixel
	skycut = crop_image( sky_image, margins=[ pixcut, pixcut])[:-1, :-1]		# make it even 

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
		skycut = cancel_extra_sources( skycut, nRMS=1)

	hdr['CRPIX1'] = hdr['CRPIX2'] = pixcut
	hdr['CRVAL1'] = 246.6175 ; hdr['CRVAL2'] = -24.4017		# so that central pixel corresponds to centre of pointing file 1x
	hdr['CTYPE1'] = 'RA---SIN'; hdr['CTYPE2'] = 'DEC--SIN'; 		# uniform it to CASA products
	fits.writeto( 'skycut.fits', skycut, header=hdr, overwrite=True)
	
	plt.imshow(  np.clip( skycut[:, ::-1 ], a_min=1e-8, a_max=None), 		# 1e-8 Jy/pix should be a fair rms low bound
			origin='lower', norm=mpl.colors.LogNorm(), cmap='gnuplot2')
	plt.savefig( 'sky_model' + fig_ext)
	plt.close()


def generate_mock_obs( filename, T_exp, data_folder='', savedir='', ptgfile='', damp=False, monosource=True, vistab_export=True):
	'''
	Call CASA simobserve and simanalyze to produce mock observations of filename.
	'''
	prepare_sky_model( filename=filename, data_folder=data_folder, savedir=savedir, damp=damp, monosource=monosource)

	image_to_process = 'skycut.fits'		# Import fits file
	ctk.importfits( fitsimage=image_to_process , imagename='skymodel.imag', overwrite=True)
	ctk.imhead( imagename='skymodel.imag', mode='put', hdkey='bunit', hdvalue='Jy/pixel')		# Edit image header

	# Generate synthetic visibilities
	diskname = filename.replace( data_folder, '' ).replace('_3000um', '').strip('.fits')  	# each one a separate folder
	os.chdir( '../' )
	ctk.simobserve( project=diskname ,
		skymodel= f'{diskname}/skymodel.imag' ,
		# indirection='J2000 16h26m26.39-24d24m30.7' , #also used as pointing center
		setpointings= False,  
		ptgfile= ptgfile, 
		incenter= '100GHz' ,
		inwidth = '7.5GHz' ,
		# mapsize=[ '' ] , # ' ' will fully cover (sky) model
		antennalist= config_name + '.cfg',
		# refdate = '2019/08/15' ,
		totaltime= f'{T_exp}s' ,
		thermalnoise= 'tsys-atm',
		user_pwv= 5.186,
		overwrite = True,
		graphics= 'file')
	plt.close()

	# Image and analyze the simulated visibilities
	ctk.simanalyze( project=diskname ,
		vis= f'{diskname}/{diskname}.{config_name}.noisy.ms' ,
		imsize = [0,0] , 	# [0 ,0] means model image will be matched
		cell= '' , 			# empty string means model cell size is to be used
		niter = 10000,
		interactive = False ,
		threshold = f'{1* analytic_sens(t=T_exp) :.3f}uJy' ,	#  [25 uJy for 10', 10uJy for 1h ...]
		weighting = 'briggs',
		analyze= True,
		graphics= 'file')
	plt.close()

	# export the CASA MS to UV table suited for GALARIO
	os.chdir( diskname )
	if vistab_export:
		vistab_name = f'{diskname}.{config_name}.noisy.ms'		# name of CASA visibility table
		# ctk.split( vis= vistab_name, keepflags=False, outputvis=vistab_name + '.binned', timebin='30s', datacolumn='all')
		casa_table = cto.table()
		casa_table.open( vistab_name ) #  + '.binned' )
		uvp.io.export_uvtable( 'uvtab.txt', tb=casa_table, vis=vistab_name, datacolumn='DATA') # CORRECTED_DATA ? Beware final residuals computation
		casa_table.close()
	print( '\nMock observation completed !\n')


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


def residuals_mock_plot( diskname, T_exp, ptgfile=''):
	'''
	Calculate the residuals between mock observations of the simulated sky model and the bestfit galario model.
	'''
	generate_mock_obs( filename='best_model.fits', T_exp=T_exp, ptgfile=ptgfile, damp=False, vistab_export=False)		# galario model
	os.chdir( '../')

	mod_tab = f'best_model/best_model.{config_name}.noisy.image'		# cleaned simanalyze best model image
	img_tab = f'{diskname}.{config_name}.noisy.image'					# cleaned simanalyze simulation image
	table = cto.table()
	table.open( img_tab )
	img = table.getcol('map').squeeze().copy( order='F') 			# simul
	table.close()
	table.open( mod_tab )
	model_img = table.getcol('map').squeeze().copy( order='F') 	# model
	table.close()
	# dRA, dDec = np.deg2rad( np.loadtxt( 'bestfit_params.txt')[-2:] / 3600 )		# offsets in [rad]
	#pixscale = fits.getval( 'skycut.fits', keyword='CDELT1')	# deg/pix
	# img = snd.shift( img, shift=( np.array([ dDec, dRA])/pixscale)  )		# shift the sky sim to match the centred galario model
	
	# img_pixscale = abs( table.getkeyword('coords')['direction0']['cdelt'][0])		# [rad/pix] of noisy image

	# size = 300
	# obs = crop_image( img, margins=[size, size])
	# model = crop_image( model_img, margins=[size, size])
	# res_n = (obs - model) / rms( img )		# residuals in RMS ratio of the full size img!
	# # lim = np.quantile( res_n, [0.05, 0.95] )
	# np.save( './best_model/' + 'residuals_n', res_n.T )

	# ptitle = 'Residuals'
	# fig, ax = plt.subplots( figsize=(6,6))	
	# ci = ax.imshow( res_n.T , origin='lower', cmap='RdBu_r', norm=mpl.colors.CenteredNorm( vcenter=0 ) )	# transpose to have as sky model
	# ax.set( title=ptitle, ) #, xlabel='au', ylabel='au')
	# ax.axis( 'off' )
	# fig.colorbar( ci, ax=ax, label='RMS units')
	# plt.show()
	# fig.savefig( ptitle + fig_ext , bbox_inches='tight')
	# plt.close()


	res_n = ( img - model_img) / rms( img )	        # residuals in RMS ratio of the full size img!
	# lim = np.quantile( res_n, [0.05, 0.95] )
	# np.save( './best_model/' + 'residuals_n', res_n.T )

	ptitle = 'Residuals_full'
	fig, ax = plt.subplots( figsize=(6,6))  
	ci = ax.imshow( res_n.T , origin='lower', cmap='RdBu_r', norm=mpl.colors.SymLogNorm( linthresh=1 ) )    # transpose to have as sky model    norm=mpl.colors.SymLogNorm( linthresh=0.5 ) )
	ax.set( title=ptitle, ) #, xlabel='au', ylabel='au')
	ax.axis( 'off' )
	fig.colorbar( ci, ax=ax, label='RMS units')
	# plt.show()
	fig.savefig( ptitle + fig_ext , bbox_inches='tight')
	plt.close()


def residuals_vis_plot( diskname, model_vis):
	'''
	Calculate the residuals between the visibilities of the mock observations and the bestfit model (galario + multisource).
	'''
	MSname = f'{diskname}.{config_name}.noisy.ms'		# mock obs MS
	casa_table = cto.table()
	casa_table.open( MSname, nomodify=False )
	modeldata = casa_table.getcol('MODEL_DATA')		# inherit the shape structure
	modeldata[:] = model_vis 		# copy model visibilities broadcasted to correct shape
	casa_table.putcol( 'MODEL_DATA', modeldata )		# add the fitted model to the MS
	casa_table.flush()
	casa_table.open( f'{diskname}.{config_name}.noisy.image' )
	noisy_img = casa_table.getcol('map').squeeze().copy( order='F') 	# cleaned simanalyze simulation image
	pixscale = abs( casa_table.getkeyword('coords')['direction0']['cdelt'][0])		# [rad/pix] of noisy image
	casa_table.close()

	ctk.uvsub( vis= MSname )	# compute the RESIDUALS = DATA - MODEL  and puts them in the corrected_data column
	ctk.tclean(		# image the residuals !
		vis=MSname,
		imagename='fitting_residuals',
		datacolumn='corrected',  	# Use the residuals
		imsize=[1728,1728],			# lo dice lui boh
		cell=f'{np.rad2deg(pixscale)*3600}arcsec',		# basically the pixscale
		weighting='briggs',
		niter=0, )                # No CLEANing, just make the residuals image
	
	casa_table.open( 'fitting_residuals.image' )		# the one created above, in [Jy/beam]
	res_img = casa_table.getcol('map').squeeze().copy( order='F') 		# residuals				
	casa_table.close()
	res_n = res_img / rms( noisy_img )	        # residuals in RMS ratio of the full size img!
	# lim = np.quantile( res_n, [0.05, 0.95] )
	# np.save( './best_model/' + 'residuals_n', res_n.T )

	ptitle = 'Residuals_log'
	fig, ax = plt.subplots( figsize=(6,6))  
	ci = ax.imshow( res_n.T , origin='lower', cmap='RdBu_r', norm=mpl.colors.SymLogNorm( linthresh=1 , vmax= 10, vmin=-10) )    # transpose to have as sky model    norm=mpl.colors.SymLogNorm( linthresh=0.5 ) )
	ax.set( title=ptitle, ) #, xlabel='au', ylabel='au')
	ax.axis( 'off' )
	fig.colorbar( ci, ax=ax, label='RMS units')
	# plt.show()
	fig.savefig( ptitle + fig_ext , bbox_inches='tight', dpi=300)

	ptitle = 'Residuals_lin'
	# fig, ax = plt.subplots( figsize=(6,6))  
	ci = ax.imshow( res_n.T , origin='lower', cmap='RdBu_r', vmin=None, vmax=None )    # transpose to have as sky model    norm=mpl.colors.SymLogNorm( linthresh=0.5 ) )
	ax.set( title=ptitle, ) #, xlabel='au', ylabel='au')
	# ax.axis( 'off' )
	fig.colorbar( ci, ax=ax, label='RMS units')
	# # plt.show()
	fig.savefig( ptitle + fig_ext , bbox_inches='tight', dpi=300)
	plt.close()


	ctk.tclean(		# image the best model !
		vis=MSname,
		imagename='best_model',
		datacolumn='data',  	# Use the residuals
		imsize=[1728,1728],			# lo dice lui boh
		cell=f'{np.rad2deg(pixscale)*3600}arcsec',		# basically the pixscale
		weighting='briggs',
		niter=0, )                # No CLEANing, just make the residuals image
	
	casa_table.open( 'best_model.image' )		# the one created above, in [Jy/beam]
	best_img = casa_table.getcol('map').squeeze().copy( order='F') 		# best model img				
	casa_table.close()
	ptitle = 'Bestfit_model' 
	fig, ax = plt.subplots( figsize=(6,6))  
	ci = ax.imshow( best_img.T , origin='lower', cmap='gnuplot2', norm=mpl.colors.LogNorm( vmin=rms( noisy_img ), vmax=None, clip=True) )    # transpose to have as sky model    norm=mpl.colors.SymLogNorm( linthresh=0.5 ) )
	ax.set( title=ptitle, ) #, xlabel='au', ylabel='au')
	ax.axis( 'off' )
	fig.colorbar( ci, ax=ax, label=r'$I_\nu$ [Jy/beam]')
	# plt.show()
	fig.savefig( ptitle + fig_ext , bbox_inches='tight', dpi=300)
	plt.close()

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


def make_uvplots( diskname, bestfit_arr, galargs, two_comp, uvbin_size=30e3, extra_sources=[0,0] ):
	''' Produce UVplots for all the bestfit solutions. '''
	# uvbin_size = 30e3     # uv-distance bin, units: wle

	if bestfit_arr.ndim < 2:
		bestfit_arr = np.expand_dims( bestfit_arr, axis=0)
	
	for b in range( bestfit_arr.shape[0] ):		# iterate on the given b best solutions
		bestfit = bestfit_arr[b,:].copy()
		# compute the visibilities of the bestfit model
		inc, PA, dRA, dDec = bestfit[-4:]
		inc *= deg ; PA *= deg ; dRA *= arcsec ; dDec *= arcsec ;		# convert !
		target_model, chi2, vis_model = galario_fit( pars= bestfit, galargs=galargs, two_comp=two_comp, extra_sources=extra_sources )
		Rmin, dR, nR, nxy, dxy, u, v, Re_obs, Im_obs, w = galargs
		amod = 0.02
		mod_lab = None

		if b==0:	# do this only for the best one of all
			rot_target = snd.rotate( target_model, angle=-PA/deg, reshape=False )   	# correct for PA rotation  
			r_s_target = snd.shift( rot_target, shift=( dDec/dxy, -dRA/dxy ) )  		# shift the model to match the mock obs 
			if np.any( extra_sources[0] ):
				npix, pixscale = extra_sources[0].shape[0], extra_sources[1]
			else:		# these two should coincide anyway
				table = cto.table()
				table.open( f'{diskname}.{config_name}.noisy.image.pbcor' )		# only for shape and pixscale
				npix = table.getcol('map').squeeze().shape[0]
				pixscale = abs( table.getkeyword('coords')['direction0']['cdelt'][0])	# [rad/pix]
				table.close()
			
			bestmod_image = resample_image( r_s_target, npix_new=npix, old_pixscale=dxy, new_pixscale=pixscale ) + extra_sources[0]		# resample to the sky data pixel scale
			hdr = fits.Header({'CTYPE1':'RA---SIN', 'CRVAL1': 246.6175, 'CRPIX1': bestmod_image.shape[1]//2, 'CDELT1': np.rad2deg(pixscale), 'CUNIT':'degree', 
					  		'CTYPE2':'DEC--SIN', 'CRVAL2': -24.4017, 'CRPIX2': bestmod_image.shape[0]//2, 'CDELT2': np.rad2deg(pixscale),
					  		'DXY_orig': dxy, 'DR': dR, 'NR': nR})
			fits.writeto( 'best_model.fits', bestmod_image[:, ::-1], overwrite=True, header=hdr)	# save it like skycut
			
			# observations uv-plot
			uv = uvp.UVTable( uvtable=[u*wle, v*wle, Re_obs, Im_obs, w], wle=wle, columns=uvp.COLUMNS_V0 )
			uv.apply_phase( -dRA, -dDec)         # center the source on the phase center
			uv.deproject( inc=0, PA=0, inplace=False)
			axes = uv.plot( label='Data', linestyle='.', color='k', yerr=True, uvbin_size=uvbin_size )

			red_chi2 = chi2/(nR - len(bestfit))
			print( '\ngalario Chi^2: ', chi2, '\n reduced chi2: ', red_chi2 ,'\n\n' )
			np.savetxt( f'bestfit_chi2.txt', bestfit, footer=f'\n{red_chi2 :.3f} \t (reduced chi2) \n{chi2 :.2f} \t (chi2)')
			amod = 1.
			mod_lab = 'Best model'

		# model uv-plot
		uv_mod = uvp.UVTable( uvtable=[u*wle, v*wle, vis_model.real, vis_model.imag, w], wle=wle, columns=uvp.COLUMNS_V0 )
		uv_mod.apply_phase( -dRA, -dDec)     # center the source on the phase center
		uv_mod.deproject( inc=0, PA=0, inplace=False)
		uv_mod.plot( axes=axes, linestyle='-', color='r', alpha=amod, label=mod_lab, yerr=False, uvbin_size=uvbin_size)

	axes[0].figure.savefig( 'uvplot' + fig_ext)
	plt.close()
	return bestmod_image, vis_model


def bestfit_plots( diskname, galargs, two_comp, extra_sources):
	'''Produce UVplots for best solutions and a skymodel of bestfit'''
	# mpl.use('macosx')
	bestfit = np.loadtxt( 'bestfit_params.txt' )
	model_image, mod_vis = make_uvplots( diskname, bestfit, galargs=galargs, two_comp=two_comp, extra_sources=extra_sources)		# make the UV plots and save the best model image
	residuals_vis_plot( diskname, mod_vis )

	# # best model visual check
	plot_img = np.clip( crop_image( model_image, margins=[500, 500]), a_min= 1e-10, a_max=None)		# [:, ::-1]
	plt.imshow( plot_img, origin='lower', norm=mpl.colors.LogNorm(), cmap='gnuplot2')	# slicing to have it mirrored as casa
	plt.title('galario best model')
	#plt.show()
	plt.savefig( f'galario_sky-model_bestfit.png')
	plt.close()
	# plot_img = np.clip( model_image, a_min= 1e-10, a_max=None)		# [:, ::-1]
	# plt.imshow( plot_img, origin='lower', norm=mpl.colors.LogNorm(), cmap='gnuplot2')
	# plt.title('galario best model')
	# #plt.show()
	# plt.savefig( f'galario_sky-model_bestfit__whole.png')
	plt.close()
	return



def get_galargs():
	u, v, Re_obs, Im_obs, w = np.require( np.loadtxt( f'uvtab.txt', unpack=True), requirements='C')
	u /= wle
	v /= wle	# have the baselines in lambda units
	nxy, dxy = gd.get_image_size( u, v, verbose=True) # , PB=1.13*wle/12 )		# number and size of pixel in radians

	# radial grid parameters
	Rmin = 0  	# arcsec
	Rmax = 6	# arcsec
	dR = np.rad2deg(dxy) * 3600 / 11   	# arcsec
	nR = int( Rmax / dR )			# dR per nr dnon deve superare il raggio massimo del modello,  3*MRS
	return [Rmin, dR, nR, nxy, dxy, u, v, Re_obs, Im_obs, w]


def mcmc_regress( diskname, nsteps=200, two_components=True, Ncpu=9, savedir='', monosource=True):
	'''
	Main pipeline for fitting YSO models with galario to a sky model (filename).
	'''
	# projectname = filename.strip( savedir ).strip('_3000um.fits')  	# each one a separate folder
	os.chdir( savedir + diskname )
	galargs = get_galargs() 
	extra_sources = copy_extra_sources( diskname ) if monosource==False else (0,0)		# deal with multiplicity in FoV

	# parameter space domain
	p_ranges_2 = np.array([[8., 15],		# Log10( I0disk )	[Log(Jy/sr)]
						[2., 12.],		# Log10( IOenvelope)   /// f (lum fraction partition) []
						[1e-5, .7],		# sigma i.e. sma [arcsec]
						[1e-3, 6],		# Ri [arcsec]
						# [0.5, 10],		# Rout [arcsec]
						[1, 5],			# p_index []
						[-5., 95.],		# inc (deg)
						[-7, 180.],		# PA (deg)
						[-2, 2],		# dRa (arcsec)
						[-2, 2]])		# dDec (arcsec)

	p_ranges_gauss = np.array([[8.5, 15],	# Log10( I0 )	[Log(Jy/sr)]
						[-5, 11],		# Log(a) const 		[Log(Jy/sr)]
						[1e-5, .8],		# sigma i.e. sma 	[arcsec]
						[-5., 95.],		# inc (deg)
						[-7, 180.],		# PA (deg)
						[-2, 2],		# dRa (arcsec)
						[-2, 2]])		# dDec (arcsec)

	# initial guess for the parameters
	p0_2 = np.array([12, 7., 0.2, 2.1, 3, 80., 45., 0., 0.]) 	# Log(I0), Log(Ienv), sma, Rin, p_idx, (inc, PA, dRA, dDec)
	p0_gauss = np.array([12, 5., 0.2, 80., 45., 0., 0.])		# Log(I0), Log(a), sma, inc, PA, dRA, dDec
	labels_gauss = ['Log($I_0$)', 'Log(Ie)', '$\sigma$', '$i$', 'PA', 'dRA', 'dDec']
	# labels_2 = ['Log($I_0$)', 'f', '$\sigma$', 'Ri', 'Rout', 'p_idx', '$i$', 'PA', 'dRA', 'dDec']
	labels_2 = [r'Log($I_{0d}$)', r'Log($I_{0e}$)', '$\sigma$', 'R_i', 'p_idx', '$i$', 'PA', 'dRA', 'dDec']
	if two_components:
		p0_mc = p0_2
		p_rang_mc = p_ranges_2
		labs_mc = labels_2
	else:
		p0_mc = p0_gauss
		p_rang_mc = p_ranges_gauss
		labs_mc = labels_gauss

	# execute the MCMC
	sampled = mcmc_run( galargs=galargs, p0= p0_mc, p_ranges= p_rang_mc, 
			nsteps=nsteps, nwalkers=Nwalkers, nthreads=Ncpu, backend_fname='last_sampler', 
			two_comp=two_components, append=False, extra_src=extra_sources )

	# sampled = emcee.backends.HDFBackend( 'last_sampler.h5' )
	bestfit = mcmc_plots( sampled, labels=labs_mc, burn_in= nsteps//3, figures=False )
	np.savetxt( f'bestfit_params.txt', bestfit )

	bestfit_plots( diskname, galargs=galargs, two_comp=two_components, extra_sources=extra_sources)		# make the UV plots and save the best model image
	# os.chdir( '../')


# def main_fit( filename, T_exp=5000, nsteps=200, two_components=True, Ncpu=9, data_folder='', savedir='', ptgfile='', damp=False):
# 	'''
# 	Main pipeline for fitting YSO models with galario to a sky model (filename).
# 	'''

# 	generate_mock_obs( filename, T_exp=T_exp, damp=damp, 
# 				   data_folder=data_folder, savedir=savedir, ptgfile=ptgfile)
	
# 	mcmc_fit()
# residuals_mock_plot( diskname=diskname, T_exp=)








