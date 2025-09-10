### ANALYSE THE RESULTS OF MOCKOBS FITTING AND THE RETRIEVED DISK PARAMETERS  
# # usage example: >>> python DownWithTheThickness/analysis.py 3000 -Texp 3600 -2c -monosrc
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import os
import glob
import sys
import argparse
import pandas as pd
# import scipy.integrate as si
import astropy.units as u
from astropy import constants as const
import casatools as cto
import emcee, corner
from scipy.optimize import curve_fit
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops, regionprops_table
from skimage.morphology import closing, footprints
from local_variables import *
plt.rcParams.update({ 'font.size':11, 'legend.fontsize':9, 'figure.dpi':200})

OKlist = np.array([17, 20, 30, 42, 43, 50, 52, 53, 57, 65, 67, 72, 78, 79, 82, 83]) 	# disk numbers with sim info available (70, 29 no bc binary, 63 75 no bc no info in truths)
tungslist = np.array([17, 20, 30, 42, 50, 52, 53, 57, 65, 67, 70, 78, 79]) 		# disk numbers fitted in Tung+24
pixscale = 9.92063492063492e-6      	# deg
sr_to_pix = np.deg2rad( pixscale )**2   # convert Jy/sr to Jy/pix
dist = 140 *u.pc  # parsec
au_to_rad = 1 / dist.to_value(u.au)
au_to_as = 1 / dist.to_value(u.au) * 180 / np.pi * 3600		# from au to arcsec


def gauss_flux_tot( I0, sigma, Rmax):
	'''
	Compute the total flux from a gaussian disk with peak brightness I0 integrating radially up to Rmax. 
	inputs  I0: Jy/sr       sigma, Rmax: rad
	returns F_v: [Jy]
	'''
	return 2*np.pi* I0 * sigma**2 * (1 - np.exp( -0.5 * (Rmax/sigma)**2 ) )


def accuracy_ratio( obs_val, sim_val ):
	ratio = obs_val / sim_val
	ratio[ ratio < 1] = 1 / ratio
	return ratio


def planck_bbody( v, T):
	'''v frequency in Hz and T temperature in K. Bv in cgs units. '''
	return 2 * const.h.cgs.value * v**3 / const.c.cgs.value**2 / ( np.exp( const.h.cgs.value * v / (const.k_B.cgs.value * T) ) - 1 )


def kappa_empir( v_obs, beta):
	'''Empirical emission coefficient kappa [cm^2 / g]. v_obs in [Hz]'''
	kappa = 10 * ( v_obs / 9.857e11 )**beta
	return kappa


def temp_profile_Tung( lum, r):
	'''Average T(r) profile from Tung24 fit. lum=Lint+Lacc in [Lsun] and r in [au]. '''
	return lum**0.25 * ( r / 35 )**(-0.52) * 71   # Kelvin


def thick_flux( v, d, r_max, l_star):
	'''
	Compute theoretical flux in case of fully thick disk (rough integration).
	v [Hz], d [pc], r_max [rad], l_star [Lsun].
	returns: F_v in [Jy]
	'''
	r_arr = np.linspace( 4e-10, r_max, 1000) * d			# radial integration grid [pc]
	dr = r_arr[1] - r_arr[0]
	B_v = planck_bbody( v, T= temp_profile_Tung( lum=l_star, r=r_arr.to_value(u.au) ) )
	F_v = 1 / d.cgs.value**2 * np.sum( B_v * 2 * np.pi * r_arr.cgs.value * dr.cgs.value )	# [cgs: erg/s/cm2/Hz]
	return F_v * 1e23		# [Jy]


def alma_resolution( wle, config_name):
	'''Return the FWHM resolution [arcsec] given the lambda [m] and the config.'''
	L80_dict = {'6':1172.5, '7':1673.1, '8':3527.3 , '9':6482.6}	# 80 percentile baselines lenght [m]
	C_number = config_name[-1]		# take the config number
	theta_res = 0.574 * wle / L80_dict[ C_number ]		# [rad]
	return theta_res * 180 / np.pi * 3600	# [arcsec]


def plot_opacity():
	# let's take the opacity used by Tung to check the dust mass
	opac_df = pd.read_table( '/Users/gcolumba/PostDoc_Mac/PostProc/simulations/dustkapscatmat_optool.inp', 
							skiprows=44, nrows=150, delimiter='\s+', engine='python', header=None, names=['lam', 'kabs', 'ksca', 'g'])

	v_obs = const.c.cgs.value / (opac_df.lam.values *1e-4) 		# Hz
	k_v_15 = kappa_empir( v_obs, beta=1.5)
	k_v_1 = kappa_empir( v_obs, beta=1.0)

	fig, ax = plt.subplots()
	ax.plot( opac_df.lam *1e-3, opac_df.kabs, label='K_abs (opTool)', c='k', ls='--')		# lambda from um to mm
	# ax.plot( opac_df.lam *1e-3, opac_df.ksca, label='K_sca', alpha=0.2)
	ax.plot( opac_df.lam *1e-3, k_v_15, label=r'k($\beta=1.5$)', c='b', ls='-.')
	ax.plot( opac_df.lam *1e-3, k_v_1, label=r'k($\beta=1.0$)', c='r')
	ax.vlines( x=[0.89, 3, 7], ymin=1e-2, ymax=1e5, colors='gray', alpha=0.6, linestyles=':', linewidths=1)
	ax.set( xscale='log', yscale='log', xlabel='$\lambda$ [mm]', ylabel='$\kappa$ [cm2 / g]', xlim=[1e-4, 20], ylim=[1e-2, 1e5])
	ax.legend( loc='lower left')
	# ax.grid( True, axis='both', alpha=0.5, linestyle=':')
	plt.show()


# def plot_Fv_compare_mod( df, v_obs, kappa, rdata='sim', Tbb=122):

# 	F_thick_sim = [ thick_flux( v_obs, dist, r_max=df.R_sim.iloc[i] *au_to_rad, l_star=df.L_tot.iloc[i]) for i in range( len(df.R_sim)) ]
# 	thin_flux_sim = df.M_sim/100 * const.M_sun.cgs.value * 0.54 * planck_bbody( v_obs, T=Tbb) / dist.cgs.value**2  *1e23		# [Jy] 
# 	thin_flux_sim2 = df.M_sim/100 * const.M_sun.cgs.value * 0.54 * planck_bbody( v_obs, 
# 								T= temp_profile_Tung(df.L_tot, r=df.R_sim) ) / dist.cgs.value**2  *1e23		# [Jy] 
	
# 	Rdata = df.R_obs if rdata=='obs' else df.R_sim		# use either just for plotting purposes
# 	Fobs = df.F_obs * (1 - 0.7 * np.sin( np.deg2rad( df.i_sim)) )		# reduce obs flux by an amount prop to inc

# 	ptitle = 'Flux_thickness_MOD'
# 	fig, ax = plt.subplots( figsize=(7,4), tight_layout=True)
# 	# ax.scatter( x=df.R_obs, y=df.F_thick, marker='s', c='k', label='Thick flux', alpha=0.8 )	# thick fluxes
# 	# ax.scatter( x=df.R_sim, y=thin_flux_sim, marker='x', c='gray', label='Thin flux from Msim (T=122K)', alpha=0.7)
# 	ax.scatter( x=df.R_sim, y=F_thick_sim, marker='s', c='k', label='Thick flux from Rsim', alpha=0.7)
# 	ax.scatter( x=df.R_sim, y=thin_flux_sim, marker='v', c='r', label='Thin flux from Msim (T Tung)', alpha=0.8)
# 	ax.scatter( x=Rdata, y=Fobs, marker='o', c='g', label='Observed flux', alpha=0.7 )		# observed fluxes	# r OBS or SIM ??
# 	ax.set( xlabel= fr'$ R_\mathrm{{{rdata}}} $ [au]', ylabel= r'$ F_{\nu} $ [Jy]', xscale='log', yscale='log', title=ptitle )
# 	# ax.grid( True, axis='x', alpha=0.5, linestyle=':')
# 	ax.legend( loc='lower right')
# 	fig.savefig( ptitle + f'_k{kappa :.3f}' + '.png' , bbox_inches='tight')
# 	plt.show()


def plot_Fv_compare( df, v_obs, k_sim, rdata='sim', Tavg=122, run_name=''):

	F_thick_sim = [ thick_flux( v_obs, dist, r_max=df.R_sim.iloc[i] *au_to_rad, l_star=df.L_tot.iloc[i]) for i in range( len(df.R_sim)) ]
	thin_flux_sim = df.M_sim/100 * const.M_sun.cgs.value * k_sim * planck_bbody( v_obs, T=Tavg) / dist.cgs.value**2  *1e23		# [Jy] 
	# thin_flux_sim2 = df.M_sim/100 * const.M_sun.cgs.value * k_sim * planck_bbody( v_obs, 
	# 							T= temp_profile_Tung(df.L_tot, r=df.R_sim) ) / dist.cgs.value**2  *1e23		# [Jy] 
	
	Rdata = df.R_obs if rdata=='obs' else df.R_sim		# use either just for plotting purposes

	ptitle = 'Flux thickness' + run_name
	fig, ax = plt.subplots( figsize=(6,4), tight_layout=True)
	# ax.scatter( x=df.R_obs, y=df.F_thick, marker='s', c='k', label='Thick flux', alpha=0.8 )	# thick fluxes
	# ax.scatter( x=df.R_sim, y=thin_flux_sim, marker='x', c='gray', label='Thin flux from Msim (T=122K)', alpha=0.7)
	ax.scatter( x=df.R_sim, y=F_thick_sim, 	 marker='s', c='k', label='Thick flux from Rsim', alpha=0.7)
	ax.scatter( x=df.R_sim, y=thin_flux_sim, marker='v', c='r', label=f'Thin flux from Msim (T={Tavg :1.0f} K)', alpha=0.8)
	ax.scatter( x= Rdata, 	y= df.F_obs, 	 marker='o', c='g', label='Fitted flux', alpha=0.7 )		# "observed" fluxes from galario fitting
	ax.scatter( x= Rdata, 	y= df.Fv_count,  marker='o', c='b', label='Counts flux', alpha=0.3 )		# "observed" fluxes from direct counts
	ax.set( xlabel= fr'$ R_\mathrm{{{rdata}}} $ [au]', ylabel= r'$ F_{\nu} $ [Jy]', xscale='log', yscale='log', title=ptitle )
	# ax.grid( True, axis='x', alpha=0.5, linestyle=':')
	ax.legend( ) # loc='lower right'
	[ fig.savefig( ptitle + f'_k{k_sim :.3f}_T{Tavg :1.0f}K' + fig_ext , bbox_inches='tight') for fig_ext in ('.png', '.pdf') ]
	plt.show()
	

def thick_sim_inspo( df, k_sim, v_obs):
	'''directly from Tungs_truth.dat'''
	F_thick_sim = [ thick_flux( v_obs, dist, r_max=df.R_disk.iloc[i] *au_to_rad, l_star=df.L_acc.iloc[i] + df.L_int.iloc[i]) for i in range( len(df.R_disk)) ]
	thin_flux_sim = df.M_disk/100 * const.M_sun.cgs.value * k_sim * planck_bbody( v_obs, T=122) / dist.cgs.value**2  *1e23	
	
	ptitle = 'Simulation thin vs thick spread'
	fig, ax = plt.subplots( figsize=(7,4), tight_layout=True)
	ax.scatter( x=df.R_disk, y= F_thick_sim, marker='s', c='k', label='Thick flux from Rsim', alpha=0.7)
	ax.scatter( x=df.R_disk, y= thin_flux_sim, marker='v', c='r', label='Thin flux from Msim (T=122K)', alpha=0.7)
	ax.set( xlabel= r'$ R_\mathrm{obs} $ [au]', ylabel= r'$ F_{\nu} $ [Jy]', xscale='log', yscale='log', title=ptitle )
	# ax.grid( True, axis='x', alpha=0.4, linestyle=':')
	ax.legend( loc='lower right')
	[ fig.savefig( ptitle + f'_k{k_sim :.3f}' + fig_ext , bbox_inches='tight') for fig_ext in ('.png', '.pdf') ]
	plt.show()
	

def plot_mass_compare( df, run_name, Tavg):
	'''Compared retrieved mass from obs to simul mass of disks. '''
	M_ratio = accuracy_ratio( df.M_obs, df.M_sim/100 )

	ptitle = 'Mass comparison' + run_name
	fig, ax = plt.subplots( figsize=(5,5), tight_layout=True)
	ax.axline( xy1=(0.0001, 0.0001), slope=1, ls='--', c='gray' )		# y=x identity
	ax.scatter( x=df.M_sim/100, y=df.M_obs, marker='o', c='r', alpha=0.7)		# observed fluxes
	ax.text( x=0.01, y=0.85, s= f'mean accuracy: {np.mean( M_ratio) :1.1f}x',
		ha='left', va='center', transform=ax.transAxes, color='gray', fontsize=10, alpha=0.8)
	ax.set( xlabel= r'$ M_\mathrm{sim} $ [M$_{\odot}$]', ylabel=r'$ M_\mathrm{obs} $ [M$_{\odot}$]' , xscale='log', yscale='log', title=ptitle )
	# ax.grid( True, axis='x', alpha=0.5, linestyle=':')
	[ fig.savefig( ptitle + f'_T{Tavg :1.0f}K' + fig_ext , bbox_inches='tight') for fig_ext in ('.png', '.pdf') ]
	plt.show()


def plot_radius_compare( df, res_limit, run_name):
	'''Assuming R_obs is R_90, in [au]. '''
	r_ratio_90 = accuracy_ratio( df.R_obs, df.R_sim )
	r_ratio_95 = accuracy_ratio( df.R_obs*1.14, df.R_sim )
	R_reslim = res_limit * 2.1436 / np.sqrt(8 * np.log(2))		# resolution limit in terms of R_90 radii, to compare apples with apples
	
	ptitle = 'Radius comparison' + run_name
	fig, ax = plt.subplots( figsize=(5,5), tight_layout=True)
	ax.fill_between( [0.01, R_reslim, 10], y1=[10, 10, R_reslim], y2=0.01, step='pre', facecolor='gray', alpha=0.16, label=r'$\theta_\mathrm{res}$' )
	ax.axline( xy1=(0.5, 0.5), slope=1, ls='--', c='gray', alpha=0.8 )		# y=x identity
	ax.scatter( x=df.R_sim *au_to_as, y=df.R_obs/1.42 *au_to_as, marker='o', c='r', label='$R_{68\%}$', alpha=0.2)
	ax.scatter( x=df.R_sim *au_to_as, y=df.R_obs *1   *au_to_as, marker='o', c='g', label='$R_{90\%}$', alpha=0.8)		# observed radii
	ax.scatter( x=df.R_sim *au_to_as, y=df.R_obs*1.14 *au_to_as, marker='o', c='b', label='$R_{95\%}$', alpha=0.2)
	ax.text( x=0.01, y=0.7, s=(f'median accuracy $R_{{90\%}}$: {np.median( r_ratio_90) :1.1f}x \nmean accuracy $R_{{90\%}}$: {np.mean( r_ratio_90) :1.1f}x'
		f'\nmean accuracy $R_{{95\%}}$: {np.mean( r_ratio_95) :1.1f}x'),
		ha='left', va='center', transform=ax.transAxes, color='k', fontsize=10, alpha=0.8)
	ax.set( xlabel= r'$ R_\mathrm{sim} $ [arcsec]', ylabel=r'$ R_\mathrm{obs} $ [arcsec]' , xscale='log', yscale='log',
		 title=ptitle, xlim=[0.03,2.5], ylim=[0.03, 2.5], aspect='equal' )
	# ax.grid( True, axis='x', alpha=0.5, linestyle=':')
	ax.legend()
	[ fig.savefig( ptitle + fig_ext , bbox_inches='tight') for fig_ext in ('.png', '.pdf') ]
	plt.show()


def plot_inc_compare( df, run_name):
	'''Assuming inc in [deg]. '''
	inc = df.i_sim.copy() 
	inc[ inc>= 90] = inc - 90
	ptitle = 'Inclination comparison' + run_name
	fig, ax = plt.subplots( figsize=(5,5), tight_layout=True)
	ax.axline( xy1=(1, 1), slope=1, ls='--', c='gray' )		# y=x identity
	ax.scatter( x=inc, y=df.i_obs, marker='o', c='orange', alpha=0.8)
	ax.set( xlabel= r'$ i_\mathrm{sim} $ [deg]', ylabel=r'$ i_\mathrm{obs} $ [deg]', title=ptitle )
	# ax.grid( True, axis='x', alpha=0.5, linestyle=':')
	[fig.savefig( ptitle + fig_ext , bbox_inches='tight') for fig_ext in ('.png', '.pdf') ]
	plt.show()


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


def count_flux_sources( diskname, nRMS=5, config_name='', results_dir='' ):
	'''
	Create a copy of CASA noisy image for the areas above noise and put everything else (including central target) to zero.
	'''
	os.chdir( results_dir + diskname )
	table = cto.table()
	table.open( f'{diskname}.{config_name}.noisy.image' )		# noisy image 
	noisy_img = table.getcol('map').squeeze().copy( order='F').T 			# copy simanalyze noisy image (convolved)  [Jy/beam]
	beam_dict = table.getkeyword('imageinfo')['restoringbeam']	# a, b and PA of beam
	img_pixscale = abs( table.getkeyword('coords')['direction0']['cdelt'][0])		# [rad/pix] of noisy image
	table.close()

	## apply threshold to identify the sources on the convolved image
	thresh = nRMS * rms( noisy_img ) 		# min_bkg_rms( noisy_img )
	bw = closing( noisy_img > thresh, footprints.rectangle(3, 3) )
	cleared = clear_border( bw )		# remove artifacts connected to image border
	label_image = label( cleared )		# label image regions
	nimg_masked = np.where( noisy_img > thresh, noisy_img, 0.)		# keep everything above n*RMS
	
	sources_df = pd.DataFrame( regionprops_table( label_image,
		properties=('centroid', 'orientation', 'axis_major_length', 'axis_minor_length', 'equivalent_diameter_area'), ) ).rename(
			columns={'centroid-0':'y0', 'centroid-1':'x0', 'orientation':'PA', 'axis_major_length':'a', 'axis_minor_length':'b', 
				'equivalent_diameter_area':'diam'} )
	
	## keep only the central source !
	target_idx = ((sources_df[['y0','x0']] - np.array(noisy_img.shape)/2 )**2 ).sum( axis=1).idxmin()	# central source (target)
	miny, minx, maxy, maxx = regionprops( label_image )[target_idx].bbox		# rectangle over central source
	target_img = nimg_masked[ miny:maxy , minx:maxx]		# [Jy/beam]

	beam_area = np.pi * beam_dict['major']['value'] * beam_dict['minor']['value'] / (4*np.log(2))	# FWHM ellipse area [arcsec^2/beam]
	beam_to_pix = ( 3600* np.rad2deg( img_pixscale ) )**2  / beam_area		# to convert the flux from [Jy/beam] to [Jy/pix]
	F_v = np.sum( target_img * beam_to_pix )		# integrated flux

	# if (nimg_masked > 0).any():	
	# 	fig, ax = plt.subplots( figsize=(8, 8))		# diagnostic figure
	# 	diag_img = np.where( noisy_img > thresh, noisy_img, np.nan)
	# 	diag_img[ miny:maxy , minx:maxx] = np.nan
	# 	ax.imshow( target_img, origin='lower', norm=mpl.colors.LogNorm() )	# use noisy_img just for diagnostic plot
	# 	ax.set_axis_off()
	# 	# fig.savefig( 'multi-source_map' + fig_ext, bbox_inches='tight', dpi=600)
	# 	plt.close()
	# else: 
	# 	print( '\nNo extra sources found in the image!\n' )
	# 	return 0, img_pixscale
	return F_v		# [Jy] integrated flux observed


def produce_truths_df():
	'''From Tungs data export a dataframe with the simulation truths of my interest. '''
	import h5py
	catalog = 'disk_01440_rmax_500_f_2_rho_3.8346e-15_thermal_False.h5'
	hf = h5py.File( catalog, 'r')
	disks = {}
	for k in hf.attrs.keys():	#Extract the disk quantities
		disks[k] = hf.attrs[k]

	Rsim = []; Msim = []; Lint = []; Lacc = []; dTemp1 = []; dTemp2 = []; multip =[];
	angs_x = []; angs_y = []; angs_z = []; hr = []; Mstar = []; age =[];
	ids = disks['list_of_disks']

	for i_d in ids:
		prefix = 'disk_' + str(i_d).zfill(5)
		directions = disks[prefix + '_direction'] 	# coordinates (x, y, z) of the normal vector of the disk
		angs_x.append( np.arccos( directions[0]) *180/np.pi)
		angs_y.append( np.arccos( directions[1]) *180/np.pi)    
		angs_z.append( np.arccos( directions[2]) *180/np.pi)    
		Rsim.append( disks[prefix + '_radius'] )
		Msim.append( disks[prefix + '_mass'] )			# disk mass
		Lint.append( disks[prefix + '_star_lum'] )
		Lacc.append( disks[prefix + '_star_acclum'] )
		dTemp1.append( disks[prefix + '_Temp_mid'] )	# mid, mavg o simple ?
		dTemp2.append( disks[prefix + '_Temp_mavg'] )
		multip.append( disks[prefix + '_multiplicity'])
		hr.append( disks[prefix + '_hoverr'] )			# scale height?
		Mstar.append( disks[prefix + '_sink_mass'] )	# star mass
		age.append( disks[prefix + '_sink_age'] )

	dfT = pd.DataFrame( np.array([Msim, Rsim, Lint, Lacc, dTemp1, dTemp2, multip, hr, Mstar, age, angs_x, angs_y, angs_z]).T, 
		columns=['M_disk', 'R_disk', 'L_int', 'L_acc', 'Tmid_disk', 'Tmavg_disk', 'multiplicity', 'hr', 'M_star', 'age', 'i_yz', 'i_xz', 'i_xy'], index=ids)
	dfT.to_csv( 'Tungs_truths.dat', sep='\t')		# saving it to file for reuse


def OLD_produce_truths_df():
	TT_sim = np.load( 'disks_Tung.pkl', allow_pickle=True, encoding='bytes')
	# U_sim = np.load( '../simulations/disks_nmhd.pkl', allow_pickle=True)
	tungslist = np.array([17, 20, 30, 42, 50, 52, 53, 57, 65, 67, 70, 78, 79]) 		# disk numbers fitted in Tung+24

	# # comparison between Ugo and Tung dicts: they match (quite)
	# for n in tungslist:		# systems fitted by Tung
	# 	print( '\n', n)
	# 	print( ' ugo: \t', U_sim[f'{n}']['mass'][0] )
	# 	print( 'tung: \t', TT_sim[b'disk_mass'][n-1])

	# 	print( '\n ugo: \t', U_sim[f'{n}']['radius'][0] )
	# 	print( 'tung: \t', TT_sim[b'disk_radius'][n-1])

	# 	print( '\n ugo: \t', U_sim[f'{n}']['star_acclum'][0] )
	# 	print( 'tung: \t', TT_sim[b'star_acclum'][n-1])
	
	# for n in U_sim.keys():
	# 	print( n, U_sim[f'{n}']['multiplicity'][0] )
	# plt.plot( TT_sim[b'star_lum'] , c='b', alpha=0.7)
	# plt.plot( TT_sim[b'star_acclum'], c='r', alpha=0.7)
	# plt.plot( TT_sim[b'sink_mass'] , c='y' , alpha=0.7)
	# plt.yscale( 'log')
	# plt.show()

	masses = TT_sim[b'disk_mass'][tungslist-1]
	radii = TT_sim[b'disk_radius'][tungslist-1]
	L_int = TT_sim[b'star_lum'][tungslist-1]
	L_acc = TT_sim[b'star_acclum'][tungslist-1]
	dtemp = TT_sim[b'disk_temp'][tungslist-1]

	dfT = pd.DataFrame( np.array([masses, radii, L_int, L_acc, dtemp]).T, 
					columns=['M_disk', 'R_disk', 'L_int', 'L_acc', 'T_disk'], index=tungslist+1)
	dfT.to_csv( 'Tungs_truths.dat', sep='\t')		# saving it to file for reuse


def mass_annuli_calc( v_obs, LI0_d, sma, R_obs, kappa, Ltot ):
	r_grid = np.logspace(-5, np.log10(R_obs), 400)
	Mtot = 0
	for i in range(len(r_grid) - 1):
		dFv = gauss_flux_tot( 10**LI0_d, sma, r_grid[i+1]) - gauss_flux_tot( 10**LI0_d, sma, r_grid[i])     # annulus flux density of DISK [Jy]
		dM = ( dist.cgs.value )**2 / kappa * dFv * 1e-23 / planck_bbody( v_obs, T=temp_profile_Tung( lum=Ltot, r=(r_grid[i+1] + r_grid[i])/2 ) )
		Mtot = Mtot + dM
	return Mtot / const.M_sun.cgs.value


def main_analysis( wle, results_dir, config_name, run_name, T_avg=122, figures=True):
	'''
	Analyse the bestfit parameters of the whole sample and the derived quantities, comparing them to the simulation truths.   
	'''
	v_obs = 299792458.0/wle			# [Hz]		# 100 *1e9   obs frequency
	k_sim = 0.54 if round(wle*1e3)==3 else 0.138	# opTool original opacity for the simulation truths
	k_obs = k_sim # kappa_empir( v_obs, beta=1.5)	# 1.5 good for both 3mm and 7mm (not 0.9mm) # for the OBS # [cm2 / g]
	# T_avg = 122		# K

	disklist = sorted( glob.glob( results_dir + 'disk*') )
	if disklist == []:    
		print('NO FILES FOUND, check again the folder path!')
		sys.exit()
	print( len(disklist), 'files found')
	truths_df = pd.read_csv( truth_path, sep='\t', index_col=0 ) #.loc[OKlist]	# load my simulation truths file
	paramlist = []

	for fpath in disklist:
		os.chdir( fpath )
		diskname = fpath.strip( results_dir ).strip('disk')
		disk_n = int(diskname.strip( '_yzx'))
		
		if disk_n in OKlist:
			try:
				# read the bestfit params from file for I0 and sma
				pars = np.loadtxt( 'bestfit_params.txt')	# galario fits I0 in Jy/sr units
				LI0_d = pars[0]                   			# disk peak intensity 	Log[Jy/sr]
				LI0_env = pars[1]							# envelope peak intensity	Log[Jy/sr]
				Ri = np.deg2rad( pars[3] /3600)				# inner env radius	[arcsec --> rad]
				sma = np.deg2rad( pars[2] /3600)     		# gauss disk sigma	[arcsec --> rad]
				i_obs = pars[-4]							# disk inclination [deg]
				p_idx = pars[4]		# TODO : adjust when using for new 2c method !!!

				R_68 = sma * np.sqrt( -2 * np.log(1-0.68))      # 68% radius  [rad]
				R_90 = R_68 * 1.42                              # 90% radius
				R_95 = R_68 * 1.62
				R_obs = R_90     # as Tung 

				F_v = gauss_flux_tot( 10**LI0_d, sma, 1*R_obs)      # observed flux density of DISK [Jy]
				Fv_count = count_flux_sources( 'disk'+diskname, nRMS=5, config_name=config_name, results_dir=results_dir )
				
				# theoretical fully thick disk flux
				l_star = truths_df.loc[ disk_n ][['L_acc', 'L_int']].sum()		# L_acc + L_int [Lsun]
				F_v_thicc = thick_flux( v_obs, dist, R_obs, l_star=l_star)
				# M_obs = F_v *1e-23 * ( dist.cgs.value )**2 / (k_obs * planck_bbody( v_obs, T_avg) )  / const.M_sun.cgs.value	# [Msun]
				M_obs = mass_annuli_calc( v_obs, LI0_d, sma, R_obs, k_obs, l_star)

			except: 
				print('No bestfit params found for ', diskname)
				M_obs = R_obs = F_v = F_v_thicc = Fv_count = i_obs = Ri = p_idx = LI0_d = LI0_env = np.nan
			
			finally:
				M_sim = truths_df.loc[ disk_n ]['M_disk']		# [Msun] total disk mass from simulations
				epsilon = (M_obs *100 - M_sim) / M_sim			# obs - truth normalised discrepancy (factor 100 dust-to-gas)
				R_sim = truths_df.loc[ disk_n ]['R_disk']		# [au]
				R_obs = (R_obs * dist).to_value( u.au )			# rad to [au]
				Ri = (Ri * dist).to_value( u.au )				# rad to [au]
				i_sim = truths_df.loc[ disk_n ][ 'i' + diskname.strip( str(disk_n) ) ]
				L_tot = truths_df.loc[ disk_n ][['L_acc', 'L_int']].sum()		# L_acc + L_int [Lsun]
				F_sim_thin = M_sim/100 * const.M_sun.cgs.value * k_sim * planck_bbody( v_obs, T=122) / dist.cgs.value**2  *1e23	

				paramlist.append( [diskname, R_obs, R_sim, Ri, p_idx, M_obs, M_sim, epsilon, LI0_d, LI0_env, F_v, Fv_count, F_v_thicc, i_obs, i_sim, L_tot, F_sim_thin] )

	res_df = pd.DataFrame( paramlist, 
				   columns=['source', 'R_obs', 'R_sim', 'Ri', 'p_idx', 'M_obs', 'M_sim', 'epsilon_M', 'LI0_d', 'LI0_env',
				 'F_obs', 'Fv_count', 'F_thick', 'i_obs', 'i_sim', 'L_tot', 'Fsim_thin']
				   ).set_index('source')
	
	os.chdir( results_dir )
	res_df.to_csv( f'analysis_results-{run_name}.txt', sep='\t') #, float_format='%.2e')
	# res_df = pd.read_csv( f'analysis_results-3600s.txt', sep='\t', index_col='source')	# to load it
	
	if figures:
		# plot_opacity()
		plot_Fv_compare( res_df, v_obs, k_sim, Tavg=T_avg, rdata='sim', run_name=run_name )
		# plot_inc_compare( res_df, run_name )
		plot_mass_compare( res_df, run_name, T_avg )
		theta = alma_resolution( wle=wle, config_name=config_name)
		# plot_radius_compare( res_df, theta, run_name )
		# thick_sim_inspo( truths_df, k_sim, v_obs)

		# truths_total = pd.read_csv( truth_path, sep='\t', index_col=0 )
		# thick_sim_inspo( truths_total, k_sim, v_obs)
	return res_df



def crop_image( img, centre=None, margins=[100, 100] ):
	'''Select a subimage of margins pixels around the centre (odd size).'''
	if centre is None:      	# use the middle of the image
		centre = (np.array( img.shape)/2 ).astype(int)
	return img[ centre[0] - margins[0] : centre[0] + margins[0] +1, centre[1] - margins[1] : centre[1] + margins[1] +1]

def circular_region( arr, radius, centre=None):
	'''
	Apply a circular mask to arr. Radius in pixel.'''
	ydim, xdim = arr.shape
	if centre is None:      # use the middle of the image
		centre = ( int(xdim/2), int(ydim/2) )

	yy, xx = np.ogrid[:ydim, :xdim]     # broadcasts to a full grid
	d = np.sqrt( (xx - centre[0])**2 + (yy - centre[1])**2 )
	return d < radius


def peak_beam_avg( image, table):
	'''
	Compute the average in one beam of the cleaned image around the source peak.
	'''
	pixscale = np.rad2deg( abs( table.getkeyword('coords') ['direction0']['cdelt'][0] ) ) * 3600		# arcsec/pix
	a = table.getkeyword('imageinfo') ['restoringbeam']['major']['value']	# Beam sma, arcsec
	b = table.getkeyword('imageinfo') ['restoringbeam']['minor']['value']	# beam minor axis, arcsec
	r_beam = (a + b) * .5 / pixscale		# avg beam radius, in pixels
	peak_idx = np.unravel_index( np.argmax( crop_image(image, margins=[50,50]) ), shape=(101,101) )
	delta_centre = np.array(peak_idx ) - [50,50]	# offsets of the photocentre
	peak_centre = np.array( image.shape ) / 2 + np.roll( delta_centre, 1) 	# roll to put correct x and y offset in full image
	beam_avg = np.nanmean( image[circular_region( image, r_beam, peak_centre)] )
	return beam_avg


def plot_SNR( df, run_name ):

	fig, ax = plt.subplots( figsize=(6,4), tight_layout=True)
	dff = df.reset_index()
	dff.plot( xticks=dff.index, rot=90, logy=True, ax=ax, marker='o')
	ax.set_xticklabels( df.index)
	# ax.axhline( y=[0.002], color='gray', ls=':')
	ax.axhline( y=10, color='gray', ls='--')
	ax.axhline( y=100, color='gray', ls='-')
	ax.grid( True, axis='x', alpha=0.5, linestyle=':')
	[fig.savefig( 'SNR_plot' + run_name + fig_ext, bbox_inches='tight') for fig_ext in ('.png', '.pdf') ]
	plt.show()



def assess_SNR( Texp, wle, results_dir, config_name, run_name ):
	'''
	Evaluate the SNR of the cleaned image across the entire sample in results_dir. 
	'''
	fitslist = sorted( glob.glob( results_dir + 'disk*') )
	if fitslist == []:    
		print('NO FILES FOUND, check again the folder path!')
		sys.exit()
	# print('Files in the list:\n')
	print( len(fitslist), 'files found')

	SNRs = []

	for fname in fitslist:

		projectname = fname.replace( results_dir, '' ).replace( f'_{round(wle*1e6)}um', '').strip('.fits')  	# each one a separate folder
		img_tab = f'{results_dir}{projectname}/{projectname}.{config_name}.noisy.image'		# cleaned simanalyze image
		table = cto.table()
		table.open( img_tab )
		img = table.getcol('map').squeeze().copy() 

		peak = np.max( crop_image(img, margins=[35,35]) )	# find the peak flux in a region around the centre
		peak_beam = peak_beam_avg( img, table=table)

		noise = min_bkg_rms( img )		# the minimum rms from bkg patches
		# noise = rms( img )					# the rms of the entire image including target source

		snr = peak_beam / noise
		SNRs.append( [projectname.strip( 'disk' ), snr, peak, peak_beam, noise] )
		table.close()

	df = pd.DataFrame( SNRs, columns=['source', 'SNR', 'max peak', 'beam peak', 'noise']).set_index('source')
	os.chdir( results_dir )
	df.to_csv( f'Peak-beam_SNR_{Texp}s.txt', sep='\t') #, float_format='%.2e')
	plot_SNR( df, run_name)


def M_emp_relation( data, a, alpha, beta, gamma):
	'''
	Empiric formula to obtain observed disk mass [Msun] from total luminosity [Lsun], observed radius [au] and retrieved flux [Jy]. '''
	L_tot, R_obs, F_v = data.copy()
	M_disk = a * L_tot**alpha * R_obs**beta * F_v**gamma
	return M_disk

def M_emp_relation_log( data, La, alpha, beta, gamma):
	'''
	Empiric LOG formula to obtain observed disk mass [Msun] from total luminosity [Lsun], observed radius [au] and retrieved flux [Jy]. '''
	LL_tot, LR_obs, LF_v = data.copy()
	LM_disk = La + LL_tot*alpha + LR_obs*beta + LF_v*gamma
	return LM_disk


def fit_Mobs( results_dir, run_name, logfit=True):
	'''
	Regress the Mobs relation with a simple curve fit. 
	'''
	os.chdir( results_dir )
	df = pd.read_csv( f'analysis_results-{run_name}.txt', sep='\t')		# import the results dataframe
	df.drop(df[df['source'] == '29_xz'].index, inplace=True)
	df.dropna( inplace=True )
	xdata = [ df.L_tot.values, df.R_obs.values, df.F_obs.values ] 		# put multivariate data into 1D arrays [Lsun, au, Jy]
	
	param_bounds = np.array( [[1e-10, -5, -5, -6 ], 		# limits on parameters: a, alpha, beta, # gamma
							[1e10 , +5, +5, +6 ]] )      	
	init_guess = [ 1e-3, 0.1, 0.5, 0.04]		# starting guess
	fitfunc = M_emp_relation
	ydata = df.M_sim.values/100
	if logfit:
		xdata = np.log10( xdata ) ; ydata = np.log10( ydata )
		param_bounds[:,0] = np.log10( param_bounds[:,0] ) ; init_guess[0] = np.log10( init_guess[0] )
		fitfunc = M_emp_relation_log
	
	popt, pcov = curve_fit( fitfunc, xdata=xdata, ydata=ydata, p0=init_guess, bounds=param_bounds, absolute_sigma=True )
	popt[0] = 10**popt[0] if logfit else popt[0]
	fit_stds = np.sqrt(np.diag( pcov ))          # from scipy doc
	print( r'fit:\n a = %.3e, $\alpha $ = %.3f , $\beta $= %.3f, $\gamma $= %.3f ' % tuple(popt) )	#  
	print( 'Fit 1 sigma errors:', fit_stds ) 

	ptitle = 'M_disk empirical fit' 
	fig, axs = plt.subplots( 1,3, figsize=(6,3), sharey=True, constrained_layout=True)
	fig.suptitle( ptitle )
	xlabs = ['L_tot', 'R_obs', 'F_v']
	xdata = [ df.L_tot.values, df.R_obs.values, df.F_obs.values ] 		# return to linear values just for plotting 
	for i in range(3): 
		axs[i].scatter( xdata[i], df.M_sim.values/100 )
		xdata_sorted_i = [ df.L_tot.values[np.argsort(xdata[i])], df.R_obs.values[np.argsort(xdata[i])], df.F_obs.values[np.argsort(xdata[i])] ]
		# axs[i].plot( np.sort(xdata[i]), M_obs_relation( xdata_sorted_i, *popt), c='r')
		if logfit: 
			xdata_mean = np.full_like( xdata_sorted_i, fill_value=10**np.mean( np.log10(xdata_sorted_i), axis=1).reshape(3,1) )
		else: 
			xdata_mean = np.full_like( xdata_sorted_i, fill_value=np.mean( xdata_sorted_i, axis=1).reshape(3,1) )
		xdata_i = xdata_mean.copy(); xdata_i[i] = xdata_sorted_i[i]		# i var left free and others fixed at their (log) mean
		axs[i].plot( xdata_sorted_i[i], M_emp_relation( xdata_i, *popt), c='r')
		axs[i].set( xlabel=xlabs[i], xscale='log', yscale='log')
	axs[0].set( ylabel='M_disk')
	llab = '_log' if logfit else ''
	[ fig.savefig( ptitle + llab + fig_ext , bbox_inches='tight') for fig_ext in ('.png', '.pdf') ]
	plt.show()
	return popt



# def log_likelihood(theta, x, y, yerr):
#     m, b, log_f = theta
#     model = m * x + b
#     sigma2 = yerr**2 + model**2 * np.exp(2 * log_f)
#     return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2))
# def log_likelihood( pars, galargs, two_comp, xsrc): 
# 	'''Galario fit chi2 likelihood function'''
# 	chi2 = galario_fit( pars=pars, galargs=galargs, two_comp=two_comp, extra_sources=xsrc )[1]
# 	return -0.5 * chi2

# def log_prior( pars, p_ranges, two_comp): 
# 	''' prior dist. pars is the array of free parameters, p_ranges their boundaries'''
# 	if (p_ranges[:, 0] < pars).all() and (pars < p_ranges[:, 1]).all():
# 		if two_comp == True:
# 			# if ( 2* pars[3] < pars[4]):		# impose that 2 Ri < Rout  (pars[2] <= pars[3]): 
# 			return 0.0
# 			# else: return -np.inf
# 		else:
# 			if (pars[1] <= pars[0]): 				# impose that Idisk > Ienv
# 				return 0.0
# 			else: return -np.inf	
# 	else:	
# 		return -np.inf

# def log_probability( pars, p_ranges, galargs, two_comp, xsrc):
# 	logprior = log_prior( pars=pars, p_ranges=p_ranges, two_comp=two_comp)
# 	if not np.isfinite( logprior):
# 		return -np.inf
# 	return logprior + log_likelihood( pars, galargs, two_comp, xsrc)


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
	# fig.savefig( folder + 'chains_steps' + fig_ext, dpi=400)
	if figures: plt.show()
	plt.close()

	cornfig = plt.figure( figsize=(8,8))		# CORNER PLOT
	fig = corner.corner(
		flat_samples, labels=labels, quantiles=[0.16, 0.5, 0.84], # title_quantiles=[0.5],
		show_titles=True, fig=cornfig, 
		label_kwargs={'labelpad':20, 'fontsize':0}, #fontsize=8,
		title_kwargs={"fontsize": 10, 'loc':'left'},	
		)
	# cornfig.savefig( folder + 'corner_plot' + fig_ext, bbox_inches='tight')
	if figures: plt.show()
	plt.close()

	best_pars = np.percentile( flat_samples,  50, axis=0)     # best params out of fit
	return best_pars


def inspect_plots( two_comp=True, sampler=None, burnin=None, walksigma=4, results_dir=''):
	'''
	inspect MCMC plots (chains + corner).
	'''
	labels_gauss = ['Log($I_0$)', 'Log(Ie)', '$\sigma$', '$i$', 'PA', 'dRA', 'dDec']
	labels_2c = [r'Log($I_{0d}$)', r'Log($I_{0e}$)', '$\sigma$', 'R_i', 'p_idx', '$i$', 'PA', 'dRA', 'dDec']
	labs_mc = labels_2c if two_comp else labels_gauss

	disklist = sorted( glob.glob( results_dir + 'disk*') )
	if disklist == []:    
		print('NO FILES FOUND, check again the folder path!')
		sys.exit()
	print( len(disklist), 'files found')

	for fpath in disklist:
		os.chdir( fpath )
		diskname = fpath.replace( results_dir, '' )
		print( '\nInspecting:  ', diskname)
		# disk_n = int(diskname.strip( '_yzx'))
		# os.chdir( diskname )

		if sampler is None:
			sampler = emcee.backends.HDFBackend( f'{diskname}__sampler.h5', read_only=True )	# will throw store==True error if diskname is wrong
		nsteps = sampler.get_chain().shape[0]
		if burnin is None:
			burnin = nsteps//3
		bestfit = mcmc_plots( sampler, labels=labs_mc, burn_in=burnin, walk_clip_thresh=walksigma, figures=True )



if __name__=='__main__':

	parser = argparse.ArgumentParser()		# parsing the name of the disk file to read
	parser.add_argument('RT_wavel', type=int, help='obs wavelength (3000 or 7000 [um]) (default: 3000)')
	parser.add_argument('-config', type=str, help='ALMA antenna configuration (default: 11.7)')
	parser.add_argument('-Texp', type=int, default=3600, help='exposure time (default: 3600s)')
	parser.add_argument('-2c', action='store_true', help='use two-component model (default: False)')
	parser.add_argument('-monosrc', action='store_true', help='do NOT use multi-source fit and clip the extra sources (default: False)')
	parser.add_argument('-Tavg', type=int, default=122, help='average temperature of disks for flux-mass conversion (default: 122K)')
	args = vars( parser.parse_args() )

	model_comps = '2c' if args['2c'] else 'g+'
	xsrc_flag = 'mono' if args['monosrc'] else 'xsrc'
	wle = float(args["RT_wavel"]) *1e-6		# [m]	assuming wle is exact as names
	folder_wle = f'{round(wle*1e3)}mm/'
	savedir = savedir_prefix + folder_wle + f'run_{args["Texp"]}s_{model_comps}_{xsrc_flag}/' 		# results directory name
	config_name = 'alma.cycle' + args['config']
	run_suffix = f' - { folder_wle.strip("/") }  {args["Texp"]}s  {model_comps}'

	for t in np.logspace( 2, 2.7, 5):
		main_analysis( wle=wle, results_dir=savedir, config_name=config_name, run_name=run_suffix, T_avg=t, figures=True )

	assess_SNR( Texp=args['Texp'], wle=wle, results_dir=savedir, config_name=config_name, run_name=run_suffix )
	main_analysis( T_avg=args['Tavg'], wle=wle, results_dir=savedir, config_name=config_name, run_name=run_suffix, figures=True )
	fit_Mobs( results_dir=savedir, run_name=run_suffix, logfit=True)
	# inspect_plots( two_comp=args['2c'], results_dir=savedir )


# Texp = 10800
# model_comps = '2c'
# xsrc_flag = 'mono'
# wle = 0.003
# folder_wle = f'{round(wle*1e3)}mm/'
# savedir = savedir_prefix + folder_wle + f'run_{Texp}s_{model_comps}_{xsrc_flag}/'
# run_suffix = f' - { folder_wle.strip("/") }  {Texp}s  {model_comps}'
# config_name = 'alma.cycle11.7' 

# # df = main_analysis( wle=wle, results_dir=savedir, config_name=config_name, run_name=run_suffix, T_avg=150, figures=True )

# for t in np.logspace( 2, 3, 6):
# 	main_analysis( wle=wle, results_dir=savedir, config_name=config_name, run_name=run_suffix, T_avg=t, figures=True )





