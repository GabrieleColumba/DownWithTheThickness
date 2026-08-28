###     Module for plotting the bestfit models resulting from the MCMC regression

import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import os
import casatools as cto
import casatasks as ctk
from astropy.io import fits
import scipy.ndimage as snd
import emcee, corner
import galario.double as gd
from galario import deg, arcsec
# mpl.use('agg')		# quick fix required to import uvplot ...
import uvplot as uvp
from run_params import RunParams
import local_variables as loc		# file with the local path pointers and cpu settings
import visibfit_functions as vf
from mockobs import analytic_sensitivity, do_simanalyze

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


def angle_best_median( fl_samples, ang_idx, niter=10):
	'''
	For angular quantities that can be cyclic (PA), check if shifting the domain endpoints provides a better best value (median).
	niter: Descrizione
	'''
	delta_shift = 180 / niter	# [deg]
	for i in ang_idx:
		count = []
		angles = fl_samples[:, i].copy()	# select only the angle parameters
		percs = np.percentile( angles,  [50, 16, 84] ).T 
		if percs[2] - percs[1] < 22:			# uncertainty smaller than a significant fraction of the whole range
			print( '\nMarginalisation already accurate, skipping the angular median check.')
		else:
			problem = angles[ angles < 0] 
			angles = np.delete( angles, angles<0 )			# fix for the values below 0 that would wrongly increase sampling near 175°
			angles = np.append( angles, np.random.random( len(problem)) *180 )
			for n in range(niter):
				angles_s = np.where( angles < n*delta_shift,  angles + 180, angles)		# move them to the end of the range
				med = np.nanmedian( angles_s)
				hist = np.histogram( angles_s, bins=18)
				count.append( hist[0][ np.argmin( abs( hist[1] - med)) ] )		# check hist counts near median
			
			n_best = np.argmax( count )
			angles_b = np.where( angles < n_best*delta_shift,  angles + 180, angles)	
			fl_samples[:, i] = angles_b		# update orig samples with the adjusted interval
			print( 'PA values recentered with a domain shift of [deg]', n_best*delta_shift)
	return fl_samples

	
def clip_chains( samples, thresh):
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
	fig.savefig( folder + 'chains_steps' + loc.fig_ext, dpi=300)
	if figures: plt.show()
	plt.close()

	flat_samples = angle_best_median( flat_samples, ang_idx=[-3])		# i=-3 is the PA index for both 1c and 2c runs

	cornfig = plt.figure( figsize=(8,8))		# CORNER PLOT
	fig = corner.corner(
		flat_samples, labels=labels, quantiles=[0.16, 0.5, 0.84], # title_quantiles=[0.5],
		show_titles=True, fig=cornfig, 
		label_kwargs={'labelpad':20, 'fontsize':0}, #fontsize=8,
		title_kwargs={"fontsize": 10, 'loc':'left'},	
		)
	cornfig.savefig( folder + 'corner_plot' + loc.fig_ext, bbox_inches='tight')
	if figures: plt.show()
	plt.close()

	# best parameters from the walker step with lowest chi2
	# best_idx = np.unravel_index( samp_bkend.get_log_prob().argmin(), samp_bkend.get_log_prob().shape )
	# best_pars = samp_bkend.get_chain()[best_idx]
	best_pars = np.percentile( flat_samples,  [50, 16, 84], axis=0).T     # best params out of fit + 16% - 18% values !
	return best_pars


def triplot( run_meta: RunParams, as_margin=5., rulersize=1000):
	'''
	Just plot together in a nice cut three panels about a target: sky model, mock-obs, mockobs-xsrc. Inspired by pentaplot().
	'''
	config_name = run_meta.config_list[0]		# pick the highest resolution config 
	MSname = run_meta.get_MS_path( config_name=config_name )	 
	tpath = run_meta.targetpath
	if os.path.exists( tpath + 'xsrc_sub/') == False:
		print( run_meta.diskname, 'had no extra sources to show')
		return
	else:
		hdul = fits.open( tpath + 'skycut.fits' )			# load sky model 
		pixscale_s = hdul[0].header['CDELT1']		# [deg / pix]
		sky_image = hdul[0].data #.byteswap().newbyteorder() 
		pixcut = int( as_margin / (pixscale_s * 3600) )			# margin in pixel
		skycut = vf.crop_image( sky_image, margins=[ pixcut, pixcut])[:, ::-1 ] * 1000		# [mJy/pix] 
		xc, yc = np.array( skycut.shape ) / 2

		ct = cto.table()
		ct.open( f"{MSname.replace('.ms', '')}.image" )		# cleaned simanalyze simulation image
		noisy_img = ct.getcol('map').squeeze().copy( order='F').T * 1000				# [mJy/pix]
		pixscale_m = np.rad2deg( abs( ct.getkeyword('coords')['direction0']['cdelt'][0]) ) * 3600		# [arcsec/pix]
		beam_dict = ct.getkeyword('imageinfo')['restoringbeam']						# a, b and PA of beam [", ", deg]
		bmaj = beam_dict['major']['value'] / pixscale_m 
		bmin = beam_dict['minor']['value'] / pixscale_m; PA = beam_dict['positionangle']['value']	
		
		ct.open( tpath + f'xsrc_sub/{config_name}_rough.image' )		# after xsrc subtraction, in [Jy/beam]
		xsrc_sub = ct.getcol('map').squeeze().copy( order='F').T * 1000		# [mJy/pix]
		ct.close()

		rms = vf.min_bkg_rms( noisy_img )
		modlist = [noisy_img, xsrc_sub]
		pixcut_m = int( as_margin / pixscale_m )		# margin in pixel
		for i in range(len(modlist)):
			modlist[i] = vf.crop_image( modlist[i], margins=[ pixcut_m, pixcut_m])

		ptitles = ['Sky model', 'Mock observation', 'Obs - extra sources subtracted']
		units = [r'$I_\nu$ [mJy/pix]', r'$I_\nu$ [mJy/beam]', r'$I_\nu$ [mJy/beam]']
		fig, axes = plt.subplots( 1,3, figsize=(12,5) ) #; axes[1,2].set_visible(False)		# hide unused axes
		axs = axes.flatten()
		skyc = axs[0].imshow( skycut,     origin='lower', norm=mpl.colors.LogNorm(), cmap='viridis')
		obsc = axs[1].imshow( modlist[0], origin='lower', norm=mpl.colors.LogNorm( vmin=rms, clip=True), cmap='inferno')
		xsrc = axs[2].imshow( modlist[1], origin='lower', norm=mpl.colors.LogNorm( vmin=rms, clip=True), cmap='inferno')
		rls_pix = rulersize * 1/140 / (pixscale_s *3600)		# [au * arcsec/au * pix/arcsec]
		axs[0].plot( [xc - 0.5*rls_pix, xc + 0.5*rls_pix], (yc - 0.9*pixcut )*np.array([1,1]), c='w', lw=2, alpha=.9)		# ruler patch
		axs[0].text( xc-0.8*rls_pix , yc-0.9*pixcut, s=f'{rulersize :3.0f} au', color='w', ha='right', va='center', alpha=.8, fontsize=8) 
		axs[0].text( xc+0.8*rls_pix , yc-0.9*pixcut, s=f'{rulersize/140 :0.2f}"', color='w', ha='left', va='center', alpha=.8, fontsize=8)
		axs[1].text( x=0.5, y=0.05, s= f'bkg RMS={rms :1.1e} mJy/beam', ha='center', va='center', transform=axs[1].transAxes, color='w', fontsize=7, alpha=0.65)
		
		for i, cb in enumerate([skyc, obsc, xsrc]):
			fig.colorbar( cb, ax=axs[i], shrink=0.605, pad=0.00, label=units[i])
			if i > 0: 		# add beam size patch
				width_frac = bmaj / modlist[i-1].shape[1] ; height_frac = bmin / modlist[i-1].shape[0]
				beam_patch = mpl.patches.Ellipse( (0.1, 0.1), width=width_frac, height=height_frac, angle=90 + PA, 
							transform=axs[i].transAxes, facecolor='w', edgecolor='w', linewidth=0, alpha=1 )
				axs[i].add_patch( beam_patch )
			axs[i].set( title= ptitles[i], aspect='equal')	#, fontsize=9)
			# axs[i].axis('off')
			axs[i].tick_params(axis='both', left=False, top=False, right=False, bottom=False, labelleft=False, labeltop=False, labelright=False, labelbottom=False)

		#fig.suptitle( diskname + '-' + run_name, fontweight='bold' ) 
		# plt.show()
		[fig.savefig( tpath + f'Triplot_{run_meta.diskname}' + '-' + run_meta.run_name.replace(' ', '_') + fig_ext ,
					bbox_inches='tight', dpi=400) for fig_ext in ('.png', '.pdf')]
		plt.close()


def residuals_vis_plot( run_meta: RunParams, model_vis, r_robust=0.2):
	'''
	Calculate the residuals between the visibilities of the mock observations and the bestfit model (galario + multisource).
	'''
	T_exp = run_meta.Texp
	MSname = run_meta.get_MS_path( config_name=run_meta.config_list )
	tpath = run_meta.targetpath
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
		imagename= tpath + 'bestmod/best_model',
		datacolumn='corrected',  		# Use the corrected_data where we stored the model visibilities
		imsize= noisy_img.shape,	
		cell = f'{pixscale}rad',
		phasecenter='ICRS 16h26m28.2s  -24d24m06.12s',
		weighting='briggs', robust=r_robust, #deconvolver='clark', 
		niter=10000, nsigma=1, threshold= f'{ 1* analytic_sensitivity(t=T_exp) :.4f}mJy',
		) 
	
	casa_table.open( tpath + 'bestmod/best_model.image' )		# the one created above, in [Jy/beam]
	best_img = casa_table.getcol('map').squeeze().copy( order='F') 		# best model img				
	casa_table.close()
	ptitle = 'Bestfit model (obs)' 
	fig, ax = plt.subplots( figsize=(6,6))  
	ci = ax.imshow( best_img.T , origin='lower', cmap='inferno', norm=mpl.colors.LogNorm( vmin=vf.min_bkg_rms( noisy_img ), vmax=None, clip=True) )    # transpose to have as sky model
	ax.set( title=ptitle, ) #, xlabel='au', ylabel='au')
	ax.axis( 'off' )
	fig.colorbar( ci, cax= ax.inset_axes([1, 0, .05, 1]), ax=ax, label=r'$I_\nu$ [Jy/beam]')
	# plt.show()
	fig.savefig( tpath + ptitle.replace(' ', '_') + loc.fig_ext , bbox_inches='tight', dpi=300)
	plt.close()

	casa_table.open( MSname, nomodify=False )		# now the RESIDUALS
	casa_table.putcol( 'CORRECTED_DATA', orig_data - model_vis  )
	casa_table.flush() ;	casa_table.close()
	
	ctk.tclean(			# image the residuals !
		vis= MSname,
		imagename= tpath + 'bestmod/residuals',
		datacolumn='corrected',  			# Use the corrected_data where we stored the residual visibilities
		imsize= noisy_img.shape,			# compare with noisy image
		cell = f'{pixscale}rad',
		weighting='briggs', robust=r_robust, #deconvolver='clark', 
		phasecenter='ICRS 16h26m28.2s  -24d24m06.12s',
		niter=10000, nsigma=1, threshold= f'{ 1* analytic_sensitivity(t=T_exp) :.4f}mJy',
		) 

	casa_table.open( tpath + 'bestmod/residuals.image' )			# the one created above, in [Jy/beam]
	best_res = casa_table.getcol('map').squeeze().copy( order='F') / vf.min_bkg_rms( noisy_img) 		# cleaned residuals img				
	casa_table.close()
	ptitle = 'Bestfit residuals' 
	fig, ax = plt.subplots( figsize=(6,6))  
	ci = ax.imshow( best_res.T, origin='lower', cmap='RdBu_r', norm=mpl.colors.CenteredNorm( vcenter=0) )    # transpose to have as sky model 
	ax.set( title=ptitle, ) #, xlabel='au', ylabel='au')
	ax.axis( 'off' )
	fig.colorbar( ci, cax= ax.inset_axes([1, 0, .05, 1]), ax=ax, label='RMS units')
	# # plt.show()
	fig.savefig( tpath + ptitle.replace(' ', '_') + loc.fig_ext , bbox_inches='tight', dpi=300)
	plt.close()

	casa_table.open( MSname, nomodify=False )
	casa_table.putcol( 'CORRECTED_DATA', orig_data )	# restore the original data at its place
	casa_table.flush() ;	casa_table.close()
	np.save( tpath + 'bestmod/best_residuals', arr=np.float32(best_res) )	# save to file
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


def make_uvplots( run_meta: RunParams, bestfit_arr, galargs, uvbin_size=50e3, logbins=False, make_modelimg=True, save_vis=False, Axes=None ):
	'''
	Produce UVplots for all the bestfit solutions. 
	'''
	bestfit = bestfit_arr[:,0].copy()	# only take the best values (no errors)
	inc, PA, dRA, dDec = bestfit[-4:]
	inc *= deg ; PA *= deg ; dRA *= arcsec ; dDec *= arcsec ;		# convert !
	(diskmod, envmod), chi2, vis_mod = vf.galario_model( pars= bestfit, galargs=galargs, two_comp=run_meta.two_comp )
	Rmin, dR, nR, nxy, dxy, u, v, Re_obs, Im_obs, w = galargs
	wle = run_meta.wle

	if make_modelimg:
		rot_target = snd.rotate( diskmod + envmod, angle=-PA/deg, reshape=False )   # correct for PA rotation  
		r_s_target = snd.shift( rot_target, shift=( dDec/dxy, -dRA/dxy ) )  		# shift the model to match the mock obs 
		tabname = run_meta.get_MS_path( run_meta.config_list ).replace('.ms', '') + '.image'
		table = cto.table()		;		table.open( tabname )	
		npix = table.getcol('map').squeeze().shape[0]								# only for shape and pixscale
		pixscale = abs( table.getkeyword('coords')['direction0']['cdelt'][0])		# [rad/pix]
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
	#uv.deproject( inc=inc/deg, PA=PA/deg, inplace=True)
	uv.uvbin( uvbin_size, logbins=logbins)
	uvdist = uv.bin_uvdist / 1000		# [klam] 
	uvdist[ np.isclose( uvdist, 0) ] = np.nan
	data_dict = {'fmt':'o', 'ms':5, 'color':'k', 'linewidth':0, 'capsize':2, 'ecolor':'gray', 'elinewidth':0.5, 'label':'Data', 'alpha':0.8}
	ax.errorbar( x=uvdist, y=uv.bin_re, yerr=uv.bin_re_err, **data_dict)
	axins.errorbar( x=uvdist, y=uv.bin_im, yerr=uv.bin_im_err, **data_dict)
	del uv
	
	uv_mod = uvp.UVTable( uvtable=[u*wle, v*wle, vis_mod.real, vis_mod.imag, w], wle=wle, columns=uvp.COLUMNS_V0 )	# model uv-plot : disk (+ env)
	uv_mod.apply_phase( -dRA, -dDec)    # center the source on the phase center
	# uv_mod.deproject( inc=inc/deg, PA=PA/deg, inplace=True)
	uv_mod.uvbin( uvbin_size, logbins=logbins ) ; #mask = uv_mod.bin_count != 0
	model_dict = { 'ls':'-', 'color':'r', 'linewidth':1.8, 'label':'Model', 'alpha':0.95}	
	ax.errorbar( uvdist, uv_mod.bin_re, **model_dict)
	axins.errorbar( uvdist, uv_mod.bin_im, **model_dict)
	ax.text( x=0.95, y=0.95, s= fr'$\chi^2_\nu$={red_chi2 :.3f}', ha='right', va='center', transform=ax.transAxes, color='gray', fontsize=9, alpha=1.)
	del uv_mod

	if run_meta.two_comp:
		colors, labs, lls = ['tab:blue', 'tab:green'], ['disk','envelope'], ['--',':']
		for i, comp in enumerate([diskmod, envmod]):		# separately plot disk and envelope contributions
			comp_vis = gd.sampleImage( comp, dxy, u, v, PA=PA, dRA=dRA, dDec=dDec, check=False, origin='lower')	
			uv_mod = uvp.UVTable( uvtable=[u*wle, v*wle, comp_vis.real, comp_vis.imag, w], wle=wle, columns=uvp.COLUMNS_V0 )
			uv_mod.apply_phase( -dRA, -dDec)     	# center on the phase center
			# uv_mod.deproject( inc=inc/deg, PA=PA/deg, inplace=True)
			uv_mod.uvbin( uvbin_size, logbins=logbins ) ; #mask = np.isnan( uvdist ) | (uv_mod.bin_count != 0 )
			if save_vis: 	
				with open( run_meta.targetpath + 'visib_disk+env.npy', 'ab') as f:		# this requires two separate np.load calls to read back the arrays
					np.save( f, arr=comp_vis ) 		# uv_mod.bin_re + 1.j*uv_mod.bin_im  to save the binned instead of the comp_vis full

			comp_dict = { 'ls':lls[i], 'color':colors[i], 'lw':1.5, 'label':labs[i], 'alpha':0.92}
			ax.errorbar( uvdist, uv_mod.bin_re,  **comp_dict)
			axins.errorbar( uvdist, uv_mod.bin_im, **comp_dict)

	ax.set( ylabel='Re(V) [Jy]', xscale='log', yscale='log') ; ax.legend( loc='best', bbox_to_anchor=(0, 0, 0.9, 0.9), fontsize=8)
	axins.set( ylabel='Im(V) [Jy]', xscale='log', xlabel='uvdistance [k$\mathrm{\lambda}$]' )
	if ax.get_ylim()[0] < 1e-4: ax.set( ylim=[1e-4, ax.get_ylim()[1]] )		# force lower ylim at 1e-5
	if Axes != None: 
		return ax
	else: 
		fig.savefig( run_meta.targetpath + 'uvplot_log' + loc.fig_ext, dpi=200, bbox_inches='tight')
	plt.close()
	return bestmod_image, vis_mod


def pentaplot( run_meta: RunParams, bestfit_pars, galargs, as_margin=3., rulersize=100):
	'''
	Just plot together in a nice cut four panels about a target: sky model, mock-obs, model mock obs, residuals
	'''
	MSname = run_meta.get_MS_path( run_meta.config_list )
	tpath = run_meta.targetpath
	hdul = fits.open( tpath + 'skycut.fits' )		# load sky model 
	pixscale_s = hdul[0].header['CDELT1']			# [deg / pix]
	sky_image = hdul[0].data #.byteswap().newbyteorder() 
	pixcut = int( as_margin / (pixscale_s * 3600) )			# margin in pixel
	skycut = vf.crop_image( sky_image, margins=[ pixcut, pixcut])[:, ::-1 ] * 1000		# [mJy/pix] 
	xc, yc = np.array( skycut.shape ) / 2

	ct = cto.table()	; 	tabname = MSname.replace('.ms', '') + '.image'
	ct.open( tabname )		# cleaned simanalyze simulation image
	noisy_img = ct.getcol('map').squeeze().copy( order='F').T
	pixscale_m = np.rad2deg( abs( ct.getkeyword('coords')['direction0']['cdelt'][0]) ) * 3600		# [arcsec/pix]
	beam_dict = ct.getkeyword('imageinfo')['restoringbeam']						# a, b and PA of beam [", ", deg]
	bmaj = beam_dict['major']['value'] / pixscale_m 
	bmin = beam_dict['minor']['value'] / pixscale_m; PA = beam_dict['positionangle']['value']	

	ct.open( tpath + 'bestmod/best_model.image' )						# [Jy/beam]
	best_model = ct.getcol('map').squeeze().copy( order='F').T 	# best model clean img
	ct.close()
	best_res = np.load( tpath + 'bestmod/best_residuals.npy' ).T 				# cleaned residuals img
	rms = vf.min_bkg_rms( noisy_img )
	modlist = [noisy_img, best_res, best_model]
	pixcut_m = int( as_margin / pixscale_m )		# margin in pixel
	for i in range(len(modlist)):
		modlist[i] = vf.crop_image( modlist[i], margins=[ pixcut_m, pixcut_m])

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
		if i > 0: 		# add beam size patch
			width_frac = bmaj / modlist[i-1].shape[1] ; height_frac = bmin / modlist[i-1].shape[0]
			beam_patch = mpl.patches.Ellipse( (0.1, 0.1), width=width_frac, height=height_frac, angle=90 + PA, 
						transform=axs[i].transAxes, facecolor='gray', edgecolor='gray', linewidth=1, alpha=1 )
			axs[i].add_patch( beam_patch )
		axs[i].set( title= ptitles[i], aspect='equal')	#, fontsize=9)
		# axs[i].axis('off')
		axs[i].tick_params(axis='both', left=False, top=False, right=False, bottom=False, labelleft=False, labeltop=False, labelright=False, labelbottom=False)

	uvax = fig.add_axes( rect=[1.12, 0.2, 1/2.8, 0.35])		# add an axes for the uvplot
	uvax = make_uvplots( run_meta, bestfit_pars, galargs, 50e3, True, make_modelimg=False, save_vis=False, Axes=uvax)
	figtitle = run_meta.diskname + '-' + run_meta.run_name
	fig.suptitle( figtitle, fontweight='bold' ) 
	# plt.show()
	fig.savefig( tpath + f'pentaplot_' + figtitle.replace(' ', '_') + loc.fig_ext , bbox_inches='tight', dpi=300)
	plt.close()



def bestfit_plots( run_meta: RunParams, galargs=None, sampler=None, burnin=None, walksigma=3 ):
	'''
	Produce MCMC plots (chains + corner), UVplot, best model and residual visib images for best solution.
	'''
	diskname = run_meta.diskname
	print( '\n\n Creating the bestfit plots for ' + diskname + ' ...\n')
	config_list = run_meta.config_list
	targetpath = run_meta.targetpath
	labels_gauss = ['Log($I_0$)', '$\sigma$', '$i$', 'PA', 'dRA', 'dDec']
	labels_2c = [r'Log($I_{0d}$)', r'Log($I_{0e}$)', '$\sigma$', 'R_i', 'R_out/Ri', 'p_idx', '$i$', 'PA', 'dRA', 'dDec']
	labs_mc = labels_2c if run_meta.two_comp else labels_gauss

	MSname = run_meta.get_MS_path( config_list )		# "concat" or single config MS
	if os.path.exists( MSname ):
		print( 'MS found!\n', MSname )
	elif run_meta.compact_conf==True and os.path.exists( MSname )==False:
		print( '\nCreating concatenated MS, just for bestfit plots\n')
		ctk.concat( vis=[ run_meta.get_MS_path( c ) for c in config_list], concatvis= MSname)	
		do_simanalyze( MSname=MSname, run_meta=run_meta, export_vis=True )
	else: 
		raise FileNotFoundError( '\nWARNING:\n' + MSname + ' not found, cannot create bestfit plots.\n\n')

	if sampler is None:				# MCMC plots
		sampler = emcee.backends.HDFBackend( targetpath + f'{diskname}__sampler.h5', read_only=True )	# will throw store==True error if diskname is wrong
	nsteps = sampler.get_chain().shape[0]
	if burnin is None:
		burnin = nsteps//(loc.thin_f*2)		# same thinning factor as the mcmc run
	bestfit = mcmc_plots( sampler, labels=labs_mc, burn_in=burnin, walk_clip_thresh=walksigma, figures=False, folder=targetpath )
	np.savetxt( targetpath + f'bestfit_params.txt', bestfit )		# save a (Npar, 3) table with the columns being: best value, 16p, 84p
	# bestfit = np.loadtxt('bestfit_params.txt')
	
	if galargs is None:				# visib plots
		config = 'concat' if len(config_list) > 1 else config_list[0]
		galargs = vf.get_galargs( run_meta=run_meta, config_name=config )
	# copy_extra_sources( MSname, nRMS)
	if os.path.exists( targetpath + 'visib_disk+env.npy' ):
		os.remove( targetpath + 'visib_disk+env.npy' )				# remove it if it exists already
	model_image, mod_vis = make_uvplots( run_meta, bestfit, galargs, uvbin_size=20e3, logbins=False, save_vis=True )
	residuals_vis_plot( run_meta, mod_vis )
	pentaplot( run_meta, bestfit, galargs)

	#						# best model visual check
	plot_img = np.clip( vf.crop_image( model_image, margins=[500, 500]), a_min= 1e-6, a_max=None)		# [:, ::-1]
	fig, ax = plt.subplots( figsize=(6,6))
	ax.imshow( plot_img, origin='lower', norm=mpl.colors.LogNorm(), cmap='inferno')	# slicing to have it mirrored as casa
	ax.set_title('galario best model')
	ax.axis(False)
	fig.savefig( targetpath + f'galario_sky-model_bestfit' + loc.fig_ext, bbox_inches='tight', dpi=200)
	plt.close()
	return print( '\n Best-fit plots and images saved.\n')


