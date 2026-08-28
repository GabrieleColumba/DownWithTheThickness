# # # Full pipeline to mock obs + galario fit and plots of regression products.

import numpy as np
from matplotlib import pyplot as plt
from run_params import RunParams
import galario.double as gd
from galario import deg, arcsec

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


def get_galargs( run_meta: RunParams, config_name):
	'''
	Read uvtabs and get required parameters for galario.
	'''
	uvtab_name = run_meta.targetpath + f'uvtab_C{config_name.replace("alma.cycle", "")}.txt'
	u, v, Re_obs, Im_obs, w = np.require( np.loadtxt( uvtab_name, unpack=True), requirements='C')
	u /= run_meta.wle
	v /= run_meta.wle	# have the baselines in lambda units
	nxy, dxy = gd.get_image_size( u, v, verbose=False) # , PB=1.13*wle/12 )		# number and size of pixel in radians
	# radial grid parameters
	Rmin = 0  			# [arcsec]
	Rmax = Rmax_model	# [arcsec]
	dR = np.rad2deg(dxy) * 3600 / 11   	# [arcsec]
	nR = int( Rmax / dR )			# dR per nr dnon deve superare il raggio massimo del modello,  3*MRS
	return [Rmin, dR, nR, nxy, dxy, u, v, Re_obs, Im_obs, w]



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


