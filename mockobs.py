###     Module for performing the mock observations with CASA functions 

import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib as mpl
import casatools as cto
import casatasks as ctk
from astropy.io import fits
import scipy.ndimage as snd
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops, regionprops_table
from skimage.morphology import closing, footprints
import galario.double as gd
# mpl.use('agg')		# quick fix required to import uvplot ...
import uvplot as uvp
from run_params import RunParams
import local_variables as loc		# file with the local path pointers and cpu settings
import visibfit_functions as vf


def copy_extra_sources( MSname, nRMS, deconvmod=True ):
	'''
	Create a copy of CASA noisy image for the areas above noise and put everything else (including central target) to zero.
	'''
	targetpath = os.path.dirname( MSname ) + '/'
	table = cto.table()
	table.open( MSname.replace('.ms', '.image' ))						# noisy image (for regions selection only)
	noisy_img = table.getcol('map').squeeze().copy( order='F').T 		# copy simanalyze noisy image (convolved)  [Jy/beam]
	beam_dict = table.getkeyword('imageinfo')['restoringbeam']			# a, b and PA of beam
	beam_area = np.pi * beam_dict['major']['value'] * beam_dict['minor']['value'] / (4*np.log(2))	# FWHM ellipse area [arcsec^2/beam]
	img_pixscale = abs( table.getkeyword('coords')['direction0']['cdelt'][0])		# [rad/pix] of noisy image
	beam_to_pix = ( 3600* np.rad2deg( img_pixscale ) )**2  / beam_area				# to convert the flux from [Jy/beam] to [Jy/pix]
	xsrc_img = noisy_img	# deprecated !
	bkg_rms = vf.min_bkg_rms( noisy_img)
	factor = beam_to_pix
	if deconvmod:
		table.open( MSname.replace('.ms', '.model' ))		
		deconvolved = table.getcol('map').squeeze().copy( order='F').T			# deconvolved model image of the sky [Jy/pix]
		xsrc_img = deconvolved
		bkg_rms = vf.rms(deconvolved)
		factor = 1		# deconv is already in [Jy/pix]
	table.close()

	## apply threshold to identify the sources on the convolved image
	thresh = nRMS * vf.min_bkg_rms( noisy_img) 		# min_bkg_rms( noisy_img )
	bw = closing( noisy_img > thresh, footprints.rectangle(3, 3) )
	cleared = clear_border( bw )		# remove artifacts connected to image border
	label_image = label( cleared )		# label image regions
	nimg_masked = np.where( noisy_img > thresh, xsrc_img, 0)		# keep everything above n*RMS 

	# little additional check
	label_image10 = label( clear_border( closing( noisy_img > 10* vf.min_bkg_rms( noisy_img ), footprints.rectangle(3, 3) ) ) )
	print( len( np.unique( label_image)) -1, 'sources detected with nRMS =', nRMS, f'\t({len( np.unique( label_image10))-1} with 10xRMS)' )
	
	if len( np.unique( label_image)) < 3:	# 0 in label_image is bkg
		print( '\nNo extra sources found in the image!\n' )
		return (0, img_pixscale)
	
	else:
		## exclude the central source !
		sources_df = pd.DataFrame( regionprops_table( label_image, properties=('centroid', 'orientation'), ) ).rename( columns={'centroid-0':'y0', 'centroid-1':'x0'} )
		target_idx = ((sources_df[['y0','x0']] - np.array(noisy_img.shape)/2 )**2 ).sum( axis=1).idxmin() + 1	# central source (target)
		nimg_masked[ label_image == target_idx ] = 0		# zero on the main target

		fig, ax = plt.subplots( figsize=(5, 5))		# diagnostic figure
		if deconvmod: 
			smooth_r = ( beam_area / np.pi )**0.5 / np.rad2deg( img_pixscale )/3600		# smoothing radius in [pix]
			diag_img = snd.gaussian_filter( nimg_masked, sigma=smooth_r )
			diag_img[ noisy_img < thresh ] = np.nan			# just for visualisation
		else:
			diag_img = np.where( noisy_img > thresh, xsrc_img, np.nan)		# use noisy_img just for diagnostic plot
		diag_img[ label_image == target_idx ] = np.nan
		ax.imshow( diag_img, origin='lower', norm=mpl.colors.SymLogNorm( linthresh=thresh ) )
		ax.imshow( np.where(label_image==target_idx, 1, np.nan), origin='lower', cmap='bwr_r')			# mark the target position
		ax.set_axis_off()
		fig.savefig( targetpath + 'sky_xsrc_map_' + os.path.basename( MSname )[10:].replace('.ms', '') + loc.fig_ext, bbox_inches='tight', dpi=300)
		plt.close()

		return nimg_masked * factor, img_pixscale, bkg_rms	# [Jy/pix], [rad/pix], [Jy/pix]


def xsrc_to_visib( extra_sources, u, v ):
	'''
	Convert a sky image [Jy/pix] of extra sources to complex visibilities. Only convert the signal above given nRMS threshold.
	'''
	RMS_thresh = extra_sources[2]		# background level
	yy, xx = np.indices( extra_sources[0].shape ) 
	deltas_pix =  - ( xx - extra_sources[0].shape[1] / 2) +1, yy - extra_sources[0].shape[0] / 2 +1		# (+1 offset due to CASA pixel centring)
	dRA, dDec = deltas_pix[0] * extra_sources[1], deltas_pix[1] * extra_sources[1]		# [rad]
	xsrc_vis = 0 + 0.j
	clean_sources_pos = np.argwhere( abs(extra_sources[0]) > RMS_thresh )
	print( 'Converting', len(clean_sources_pos), 'points of extra sources image in visibilities' )
	for p in clean_sources_pos:
		f = extra_sources[0][ p[0], p[1]] - RMS_thresh		# read the pixel flux value and subtract the nRMS background
		Re = np.full_like( u, f )							# add each clean component as constant centred source (phase=0)
		vis_sh = gd.apply_phase_vis( dRA[p[0], p[1]], dDec[p[0], p[1]], u, v, Re + 0.j)		# shift the component to its place in the sky
		xsrc_vis = xsrc_vis + vis_sh
	return xsrc_vis


def do_simanalyze( MSname, run_meta: RunParams, export_vis):
	'''
	Launch a CASA simanalyze task to make the clean image and export the uv table for CONCAT MS.
	TODO: make this more general.
	'''
	T_exp = run_meta.Texp
	projdir = run_meta.diskname
	prev_cwd = os.getcwd()
	try:
		os.chdir( run_meta.savedir )
		print('\n', '...simanalyzing...')  
		ctk.simanalyze( project=projdir ,
			vis= f'{projdir}/{os.path.basename(MSname)}', 				# simanalyze needs the relative path...
			niter = 100,		# auto cell and imgsize values
			threshold = f'{analytic_sensitivity(t=T_exp) :.4f}mJy' ,	#  [25 uJy for 10', 10uJy for 1h ...]
			weighting = 'briggs', analyze=False, graphics= 'file')
		plt.close()
	finally:
		print( 'simanalyze done') ;	os.chdir( prev_cwd )

	if export_vis:
		print( 'printing concat uvtab')
		casa_table = cto.table()
		casa_table.open( MSname ) 
		uvp.io.export_uvtable( run_meta.targetpath + 'uvtab_Cconcat.txt', tb=casa_table, vis=MSname, datacolumn='CORRECTED_DATA') 
		casa_table.close()
	return


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


def prepare_sky_model( run_meta: RunParams):
	'''
	Open fits file with desired sky brightness model and cut it / damp it before mock-obs.
	'''
	hdul = fits.open( run_meta.fitspath )
	hdul.info()
	hdr = hdul[0].header
	pixscale = hdr['CDELT1']	# deg / pix
	sky_image = hdul[0].data #.byteswap().newbyteorder()   	# byteswap is needed for cv2 blurring

	targetpath = run_meta.targetpath
	os.makedirs( targetpath, exist_ok=True)

	main_beam = np.rad2deg( 1.13 * run_meta.wle / 12	)		# [deg]	,  0.01618 deg to have a FoV as the main beam at 3mm
	pixcut = int( main_beam / pixscale / 2 )				# margin in pixel
	skycut = vf.crop_image( sky_image, margins=[ pixcut, pixcut])[1:, 1:]		# make it even 

	if run_meta.damp:
		print('\nArtificially damping the sky beyond the MRS\n')
		npix = skycut.shape[0]
		X = Y = np.arange( -npix + npix/2, npix - npix/2 )
		xx, yy = np.meshgrid( X, Y )	# pixel grid for image filter
		max_scale = 5 / 3600		# arcsec / 3600 = [deg]
		filt_sigma = max_scale / pixscale 		# pixel
		gauss_conv = np.exp( - 0.5* (xx**2 + yy**2) / filt_sigma**2 )		# spatial filter to avoid high fourier powers
		skycut = skycut * gauss_conv	# the central target is isolated with a gaussian kernel
		skycut = np.clip( skycut, a_min=1e-8, a_max=None )	# avoid super low values, amin from d17 bkg patch rms=2e-8 Jy/pix

	if run_meta.monosrc:
		skycut = _cancel_extra_sources( skycut, nRMS=run_meta.nRMS, figure=True, figpath=targetpath)

	hdr['CRPIX1'] = hdr['CRPIX2'] = skycut.shape[0] / 2
	hdr['CRVAL1'] = 246.6175 ; hdr['CRVAL2'] = -24.4017		# so that central pixel corresponds to centre of pointing file 1x
	hdr['CTYPE1'] = 'RA---SIN'; hdr['CTYPE2'] = 'DEC--SIN'; 		# uniform it to CASA products
	fits.writeto( targetpath + 'skycut.fits', np.float32( skycut ), header=hdr, overwrite=True)
	
	ptitle = 'Sky model'
	fig, ax = plt.subplots( figsize=(6, 5.5) )  
	ci = ax.imshow( skycut[:, ::-1 ]*1000, origin='lower', cmap='inferno', norm=mpl.colors.LogNorm( vmin=max(1e-4, skycut.min()*1000), vmax=None, clip=True) ) 
	ax.set( title=ptitle, ) ; ax.axis( 'off' )
	fig.colorbar( ci, cax= ax.inset_axes([1, 0, .05, 1]),  ax=ax, label=r'$I_\nu$ [mJy/pix]')
	fig.savefig( targetpath + ptitle.replace(' ', '_') + loc.fig_ext , bbox_inches='tight', dpi=300)
	plt.close()
	return 


def perform_mock_obs( run_meta: RunParams, ptgfile='', vistab_export=True):
	'''
	Call CASA simobserve and simanalyze to produce mock observations (visibilities) of filename.
	'''
	diskname = run_meta.diskname
	wle = run_meta.wle
	T_exp = run_meta.Texp
	targetpath = run_meta.targetpath

	prepare_sky_model( run_meta=run_meta)				# create the skycut.fits file
	image_to_process = targetpath + 'skycut.fits'		# Import fits file
	image_name = targetpath + 'skymodel.imag'
	ctk.importfits( fitsimage=image_to_process , imagename=image_name, overwrite=True)
	ctk.imhead( imagename=image_name, mode='put', hdkey='bunit', hdvalue='Jy/pixel')	# Edit image header

	prev_cwd = os.getcwd()
	try:
		os.chdir( run_meta.savedir )		# CASA simobserve/simanalyze only work with relative paths.
		for i, config in enumerate( run_meta.config_list):	# iterate on each configuration
			MSname = run_meta.get_MS_path( config_name=config ) 
			try:
				ctk.simobserve( project=diskname ,
					skymodel=f'{diskname}/{os.path.basename(image_name)}' ,
					setpointings= False,  
					ptgfile= ptgfile, 
					incenter= f'{299792458.0/wle}Hz' ,		# v = c / lambda
					inwidth = '7.5GHz' ,
					# mapsize=[ '' ] , # ' ' will fully cover (sky) model
					antennalist= config + '.cfg',
					totaltime= f'{T_exp}s' ,
					thermalnoise= 'tsys-atm',
					user_pwv= 0.7 if wle<2e-3 else 5.186,  # 5.186,      # 5.186 @ 3 & 7mm, 0.7 @ 1mm
					overwrite = False, 
					graphics= 'file')
				plt.close()

				# Image and analyze the simulated visibilities
				ctk.simanalyze( project=diskname ,
					vis= f'{diskname}/{os.path.basename(MSname)}', 
					imsize = [0,0] , 	# [0 ,0] means model image will be matched
					cell= '' , 			# empty string means model cell size is to be used
					niter = 1000,
					interactive = False ,
					threshold = f'{analytic_sensitivity(t=T_exp) :.4f}mJy' ,	#  [25 uJy for 10', 10uJy for 1h ...]
					weighting = 'briggs',
					analyze= False,
					graphics= 'file')
				plt.close()
			
			except Exception as e: 
				print( 'Exception:', e)
				print( '\nSimobserve/analyze already performed, proceeding\n')

			if run_meta.monosrc==False:
				xRMS_factor = 20 	# need higher RMS for good extraction in compact config
				extra_sources = copy_extra_sources( MSname, nRMS=run_meta.nRMS + i*xRMS_factor )
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
						vis= MSname, imagename= targetpath + f'xsrc_sub/{config}_rough', datacolumn='corrected', 
						imsize=extra_sources[0].shape, cell = f'{extra_sources[1]}rad',
						phasecenter='ICRS 16h26m28.2s  -24d24m06.12s', #deconvolver='clark', 
						weighting='briggs', niter=50, nsigma=1, threshold=f'{analytic_sensitivity(t=T_exp) :.4f}mJy'  ) 
					
					casa_table.open( targetpath + f'xsrc_sub/{config}_rough.image' )		# the one created above, in [Jy/beam]
					img = casa_table.getcol('map').squeeze().copy( order='F') 		# best model img				
					casa_table.close()
					ptitle = 'Obs - extra sources subtracted'	# 'Target - xsrc (quick clean)' 
					fig, ax = plt.subplots( figsize=(6,5.5))  
					ci = ax.imshow( img.T , origin='lower', cmap='inferno', norm=mpl.colors.LogNorm( vmin=1e-6, vmax=None, clip=True) )    # transpose to have as sky model
					ax.set( title=ptitle, ) ; ax.axis( 'off' )
					fig.colorbar( ci, ax=ax, cax= ax.inset_axes([1, 0, .05, 1]), label=r'$I_\nu$ [Jy/beam]')
					fig.savefig( targetpath + ptitle.replace(' ', '_') + config + loc.fig_ext , bbox_inches='tight', dpi=200)
					plt.close()
					
			if vistab_export:				# export the CASA MS to UV table suited for GALARIO
				casa_table = cto.table()
				casa_table.open( MSname ) #  + '.binned' )
				uvp.io.export_uvtable( targetpath + f'uvtab_C{config.strip("alma.cycle")}.txt', tb=casa_table, vis=MSname, datacolumn='CORRECTED_DATA') 
				casa_table.close()
	finally:
		os.chdir( prev_cwd )

	return print( '\nMock observation completed !\n')





def _fix_skyflux( data_folder = '/Users/gcolumba/PostDoc_Mac/PostProc/simulations/sky_RT/3mm/' ):
	import glob
	import sys
	
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


def _cancel_extra_sources( skymodel, nRMS=1, figure=False, figpath='' ):
	'''
	Create a copy of CASA noisy image for the areas above noise and put everything else (including central target) to zero.
	'''
	sky_image = skymodel.copy()
	print( '\n LEVELLING OFF the extra sources with the RMS value. \n')

	## apply threshold to identify the bright sources
	noise_lev = vf.rms( sky_image ) 		# min_bkg_rms( noisy_img )
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
		fig.savefig( figpath + 'levelled_sky' + loc.fig_ext, bbox_inches='tight', dpi=200)
		plt.close()

	return sky_masked


def _twoD_Gaussian( img_size, I0, xo, yo, sigma_x, sigma_y, theta, offset ):
	x, y = np.meshgrid( np.arange(0, img_size[0]), np.arange(0, img_size[1]) )
	xo = float(xo)
	yo = float(yo)    
	a = np.cos(theta)**2 /(2*sigma_x**2) + np.sin(theta)**2 /(2*sigma_y**2)
	b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
	c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
	g = offset + I0 * np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2)))
	return g.ravel()


def _generate_skymodel( dRA, dDec, a, b, like_filename='disk17_xy_3000um.fits', data_folder='/Users/gcolumba/PostDoc_Mac/PostProc/simulations/3mm/' ):
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
	target = _twoD_Gaussian( img_size=arr.shape, I0=9e-3, xo=arr.shape[0]/2 + dRA_pix, yo=arr.shape[1]/2 + dDec_pix,
		sigma_x=b, sigma_y=a, theta=-PA, offset=1e-10)		# [Jy/pix]	theta < 0 corresponds to a countclock rotation of the disk
	arr += target.reshape( arr.shape )		# add the target to the array

	# # then add other sources along a vertical line from the target to the top of the image with equal spacing and brightness
	nsources = 4		# number of sources to add
	for i in range( 1, nsources + 1):
		dDec_pix = modelshape[0]/6 / (nsources + 1) * i		# pixel offsets
		extrasource = _twoD_Gaussian( img_size=arr.shape, I0=7e-4, xo=arr.shape[0]/2 + 0, yo=arr.shape[1]/2 + dDec_pix,
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
