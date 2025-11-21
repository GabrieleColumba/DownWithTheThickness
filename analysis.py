### ANALYSE THE RESULTS OF MOCKOBS FITTING AND THE RETRIEVED DISK PARAMETERS  
# # usage example: >>> python DownWithTheThickness/analysis.py 3000 -Texp 3600 -2c -monosrc
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import os
import glob
# import sys
import argparse
import pandas as pd
# import scipy.integrate as si
import astropy.units as u
from astropy.io import fits
from astropy import constants as const
import casatools as cto
# import emcee, corner
from scipy.optimize import curve_fit
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops, regionprops_table
from skimage.morphology import closing, footprints
from local_variables import *
from visibfit_functions import crop_image
plt.rcParams.update({ 'font.size':10, 'legend.fontsize':8, 'figure.dpi':200})

tungslist = np.array([17, 20, 30, 42, 50, 52, 53, 57, 65, 67, 70, 78, 79]) 		# disk numbers fitted in Tung+24, wb 43 ?? not shown
OKlist = 	np.array([17, 20, 30, 42, 50, 53, 57, 65, 67, 72, 78, 79, 82, 83, 43]) 	# (70, 78, 29, 52 no bc binary, 43 misterious and thick, 63 75 no bc no info in truths)
prettylist =np.array([17, 20, 30, 42, 50, 53, 57, 65, 67, 72, 79, 83])
# pixscale = 9.92063492063492e-6      	# deg
# sr_to_pix = np.deg2rad( pixscale )**2   # convert Jy/sr to Jy/pix
dist = 140 *u.pc  # parsec
au_to_rad = 1 / dist.to_value(u.au)
au_to_as = 1 / dist.to_value(u.au) * 180 / np.pi * 3600		# from au to arcsec


def gauss_flux_integral( I0, sigma, Rmax):
	'''
	Compute the total flux from a gaussian disk with peak brightness I0 integrating radially up to Rmax. 
	inputs  I0: [Jy/sr]       sigma, Rmax: [rad]
	returns F_v: [Jy]
	'''
	return 2*np.pi* I0 * sigma**2 * (1 - np.exp( -0.5 * (Rmax/sigma)**2 ) )


def plummer_integral( I0, Ri, p_idx, Rmax):
	'''
	Compute the total flux from a Plummer envelope with peak brightness I0 integrating radially up to Rmax. 
	inputs  I0: [Jy/sr]       Ri, Rmax: [rad]    p_idx: [adim] (p + q exponent)
	returns F_v: [Jy]
	'''
	p = ( p_idx - 1 ) / 2
	return np.pi* I0 * Ri**2 / (1-p) * (  ( 1 + (Rmax/Ri)**2 )**(1-p) -1 )


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

def temp_profile_envs( lum, r, q=0.4):
	'''Average T(r) profile for envelopes from Maury+2019. lum=Lint+Lacc in [Lsun] and r in [au]. '''
	return (lum/1e5)**(q/2) * ( r / 13400 )**(-q) * 60   # Kelvin


def disk_avg_T( lum, r):
	'''Average T weighted over disk surface, from from Tung24 T law. lum[Lsun] and r (tot disk) in [au]. '''
	return lum**0.25 * ( r)**(-0.52) * 15.1   # Kelvin


def thick_flux( v, d, r_max, l_star):
	'''
	Compute theoretical flux in case of fully thick disk (rough integration).
	v [Hz], d [pc], r_max [rad], l_star [Lsun].
	returns: F_v in [Jy]
	'''
	r_arr = np.linspace( 4e-10, r_max, 1000) * d			# radial integration grid [pc]
	dr = r_arr[1] - r_arr[0]
	B_v = planck_bbody( v, T= temp_profile_Tung( lum=l_star, r=r_arr.to_value(u.au) ) )
	F_v = 1 / d.cgs.value**2 * np.sum( B_v * 2 * np.pi * r_arr.cgs.value * dr.cgs.value, axis=0)	# [cgs: erg/s/cm2/Hz]
	return F_v * 1e23		# [Jy]


def alma_resolution( wle, config_name):
	'''Return the FWHM resolution [arcsec] given the lambda [m] and the config.'''
	L80_dict = {'4':369.2,'6':1172.5, '7':1673.1, '8':3527.3 , '9':6482.6}	# 80 percentile baselines lenght [m]
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


def scatter_with_errors( ax, x, y, x_lo=None, x_up=None, y_lo=None, y_up=None,
						fmt='o', facecolor='C0', edge_darken=0.9, ecolor=None,
						capsize=3, marker_alpha=0.8, err_alpha=0.25, label=None, **kwargs):
	'''
	Draw scatter points with asymmetric errorbars.
	- x, y : 1D arrays
	- x_lo, x_up, y_lo, y_up : arrays or None (absolute values, not deltas)
	'''
	def make_err( arr, lower, upper):		# build asymmetric error arrays for matplotlib (shape (2, N))
		if lower is None and upper is None:
			return None
		if lower is None: lower = arr
		if upper is None: upper = arr
		neg = arr - lower		# error must be positive deltas
		pos = upper - arr
		return np.vstack([neg, pos])
	
	x = np.asarray(x); y = np.asarray(y)
	xerr = None
	yerr = make_err( y, y_lo, y_up)

	if ecolor is None:	# compute darker edge color if not provided
		base_rgb = np.array(mpl.colors.to_rgb(facecolor))
		ecolor = tuple(np.clip(base_rgb * edge_darken, 0, 1))

	# # plot markers with facecolor and darker edge with same alpha
	# markerline = ax.errorbar( x, y, xerr=xerr, yerr=yerr, fmt=fmt, markerfacecolor=facecolor, markeredgecolor=facecolor,
	# 						ecolor=ecolor, elinewidth=1, capsize=capsize, alpha=alpha, label=label, **kwargs)
	
	# draw errorbar container with overall alpha=1 (we'll set parts individually)
	plotline, caplines, barlines = ax.errorbar( x, y, xerr=xerr, yerr=yerr, fmt=fmt, markerfacecolor=facecolor, markeredgecolor=facecolor,
					ecolor=ecolor, elinewidth=1, capsize=capsize, alpha=marker_alpha, label=label, **kwargs)

	[bar.set_alpha(err_alpha) for bar in barlines]
	[cap.set_alpha( err_alpha) for cap in caplines]
	return # markerline


def ratio_histogram( var1, var2, run_name, histcolor='tab:green'):
	'''
	Plot a histogram of the ratio between var1/var2 and write the mean and std of the distribution.
	'''
	ratio = var1 / var2		# generally obs/sim
	mean_r = np.nanmean( ratio)
	base_rgb = np.array(mpl.colors.to_rgb(histcolor))
	darken_factor = 0.8			# compute a slightly darker edge color automatically
	edge_rgb = tuple(np.clip(base_rgb * darken_factor, 0, 1))
	style = {'edgecolor': edge_rgb, 'linewidth': 1.5, 'zorder':2}

	ptitle =  f'{var1.name}_{var2.name} ratio' + run_name
	fig, ax = plt.subplots( figsize =(4,4), tight_layout=True )
	# fig.suptitle( ptitle )
	q16, median_r, q84 = np.nanquantile( ratio, [0.16, 0.5, 0.84])
	hh = ax.hist( x=ratio, bins='doane', color=histcolor, histtype='bar', **style , alpha=0.85) #, label=f'ratio, $\sigma$={np.nanstd( ratio ) :.2f}')
	ax.axvline( x=1, ls='--', c='k', alpha=0.99)
	ax.axvline( x=median_r, ls='-.', c=edge_rgb, label=f'median = {median_r :.2f}', alpha=0.9 )	
	ax.axvline( x=mean_r, ls=':', c=edge_rgb, label=f'mean = {mean_r :.2f}', alpha=0.7 )
	ax.fill_between(x=[q16, q84] , y1=[0,0], y2= hh[0].max() + 2, step='mid', facecolor='gray', zorder=1, alpha=0.19,	
		label=rf'(16-84)%, $\sigma={ np.nanstd(ratio) :.2f}$' )		# take the maximum of the hist for upper y2 limit
	ax.set( xlabel= f'{var1.name} / {var2.name}', ylabel='counts', ylim=[0, hh[0].max() + 2], title=ptitle )
	ax.legend()
	[ fig.savefig( f'Figures_{fig_ext.strip(".")}/' + ptitle.replace(' ', '_') + fig_ext , bbox_inches='tight') for fig_ext in ('.png', '.pdf') ]
	# plt.show()
	plt.close()


def plot_Fv_compare( df, v_obs, k_sim, rdata='sim', Tavg=122, run_name='', errors=True):

	F_thick_sim = [ thick_flux( v_obs, dist, r_max=df.R_sim.iloc[i] *au_to_rad, l_star=df.L_tot.iloc[i]) for i in range( len(df.R_sim)) ]
	thin_flux_sim = df.M_sim/100 * const.M_sun.cgs.value * k_sim * planck_bbody( v_obs, T=Tavg) / dist.cgs.value**2  *1e23		# [Jy] 
	# thin_flux_sim2 = df.M_sim/100 * const.M_sun.cgs.value * k_sim * planck_bbody( v_obs, 
	#  							T= disk_avg_T(df.L_tot, r=df.R_sim) ) / dist.cgs.value**2  *1e23		# [Jy] 
	Rdata = df.R_obs if rdata=='obs' else df.R_sim		# use either just for plotting purposes

	ptitle = 'Flux thickness' + run_name
	fig, ax = plt.subplots( figsize=(6,4), tight_layout=True)
	# ax.scatter( x=df.R_obs, y=df.F_thick, marker='s', c='k', label='Thick flux', alpha=0.8 )	# thick fluxes
	# ax.scatter( x=df.R_sim, y=thin_flux_sim2, marker='v', c='r', label='Thin flux from Msim (T(r)=Tung+24)', alpha=0.6, zorder=4 )
	ax.scatter( x=df.R_sim, y=F_thick_sim, 	 marker='s', c='k', label='Thick flux from Rsim', alpha=0.6)
	ax.scatter( x=df.R_sim, y=thin_flux_sim, marker='v', c='r', label=f'Thin flux from Msim (T={Tavg :1.0f} K)', alpha=0.6)
	ax.scatter( x= Rdata, 	y= df.Fv_count,  marker='o', c='b', label='Counts flux', alpha=0.2 )		# "observed" fluxes from direct counts
	if errors: 
		scatter_with_errors( ax=ax, x= Rdata, y=df.F_obs, y_lo=df.F_obs_lo, y_up=df.F_obs_up, fmt='o', facecolor='g', marker_alpha=0.7, label='Fitted flux')
	else: 	ax.scatter( x= Rdata, y= df.F_obs, marker='o', c='g', label='Fitted flux', alpha=0.7 )		# "observed" fluxes from galario fitting
	ax.set( xlabel= fr'$ R_\mathrm{{{rdata}}} $ [au]', ylabel= r'$ F_{\nu} $ [Jy]', xscale='log', yscale='log', title=ptitle )
	# ax.grid( True, axis='x', alpha=0.5, linestyle=':')
	ax.legend( ) # loc='lower right'
	[ fig.savefig( f'Figures_{fig_ext.strip(".")}/' + ptitle.replace(' ', '_') + f'_k{k_sim :.3f}_T{Tavg :1.0f}K' + fig_ext , bbox_inches='tight') for fig_ext in ('.png', '.pdf') ]
	# plt.show()
	plt.close()
	

def thick_sim_inspo( df, k_sim, Tavg, v_obs, run_name):
	'''directly from Tungs_truth.dat'''
	F_thick_sim = [ thick_flux( v_obs, dist, r_max=df.R_disk.iloc[i] *au_to_rad, l_star=df.L_acc.iloc[i] + df.L_int.iloc[i]) for i in range( len(df.R_disk)) ]
	thin_flux_sim = df.M_disk/100 * const.M_sun.cgs.value * k_sim * planck_bbody( v_obs, T=Tavg) / dist.cgs.value**2  *1e23	
	
	ptitle = 'Simulation thin vs thick spread' + run_name[0:3]
	fig, ax = plt.subplots( figsize=(7,4), tight_layout=True)
	ax.scatter( x=df.R_disk, y= F_thick_sim, marker='s', c='k', label='Thick flux from Rsim', alpha=0.7)
	ax.scatter( x=df.R_disk, y= thin_flux_sim, marker='v', c='r', label=f'Thin flux from Msim (T={Tavg :1.0f} K)', alpha=0.7)
	ax.set( xlabel= r'$ R_\mathrm{obs} $ [au]', ylabel= r'$ F_{\nu} $ [Jy]', xscale='log', yscale='log', title=ptitle )
	# ax.grid( True, axis='x', alpha=0.4, linestyle=':')
	ax.legend( loc='lower right')
	[ fig.savefig( f'Figures_{fig_ext.strip(".")}/' + ptitle.replace(' ', '_') + f'_k{k_sim :.3f}' + fig_ext , bbox_inches='tight') for fig_ext in ('.png', '.pdf') ]
	# plt.show()
	plt.close()
	

def plot_mass_env( df, run_name, ):
	'''
	Compared retrieved mass from obs to simul mass of disks with either simple approx or with annular computation. 
	'''
	M_obs = df.Meo
	M_sim = (df.Mcyl - df.M_sim) /100	# (df.Mcyl - df.M_sim)
	M_ratio = M_obs / M_sim 
	# T_label = f'_T{Tavg :1.0f}K' if simple_M else '_T(r)'

	ptitle = 'Envelope mass comparison' + run_name
	fig, ax = plt.subplots( figsize=(5,5), tight_layout=True )
	ax.axline( xy1=(0.0001, 0.0001), slope=1, ls='--', c='gray' )			# y=x identity
	ax.scatter( x=M_sim, y=M_obs, marker='o', c='r', alpha=0.7)		# observed fluxes
	# ax.text( x=0.01, y=0.85, s= f'mean accuracy: {np.nanmean( M_ratio) :1.1f}x',
	# 	ha='left', va='center', transform=ax.transAxes, color='gray', fontsize=10, alpha=0.8)
	# ax.text( x=0.01, y=0.90, s= f'T={Tavg :1.0f} K' if simple_M else 'T=T(r)',
	# 	ha='left', va='center', transform=ax.transAxes, color='gray', fontsize=12, alpha=1.)
	ax.set( xlabel= r'$ M_\mathrm{env, sim} $ [M$_{\odot}$]', ylabel=r'$ M_\mathrm{env, obs} $ [M$_{\odot}$]', xscale='log', yscale='log', title=ptitle )
	ax.axis( 'square')
	# ax.grid( True, axis='x', alpha=0.5, linestyle=':')
	[ fig.savefig( f'Figures_{fig_ext.strip(".")}/' + ptitle.replace(' ', '_') + fig_ext , bbox_inches='tight') for fig_ext in ('.png', '.pdf') ]
	# plt.show()
	plt.close()

	ptitle = 'Envelope flux'
	fig, ax = plt.subplots( )
	fig.suptitle( ptitle )
	ax.scatter(  x=df.Mcyl , y=df.Fv_env, alpha=0.7 )
	[ax.text( s=df.index[i], x=df.Mcyl[i],  y=df.Fv_env[i], horizontalalignment='left', verticalalignment='bottom', fontsize=5 ) for i in range(len(df)) ]
	ax.set(  xlabel= 'Mcyl', xscale='log', ylabel= 'F_env', yscale='log')
	# fig.supylabel( r'$\delta_M$', fontsize=12 )
	[ fig.savefig( f'Figures_{fig_ext.strip(".")}/' + ptitle + fig_ext , bbox_inches='tight') for fig_ext in ('.png', '.pdf') ]
	plt.close()


def plot_mass_compare( df, run_name, Tavg, simple_M, errors=True):
	'''
	Compared retrieved mass from obs to simul mass of disks with either simple approx or with annular computation. 
	'''
	if simple_M:
		M_obs, M_obs_lo, M_obs_up = df.M_obs_simple, df.M_obs_simple_lo, df.M_obs_simple_up
	else: 
		M_obs, M_obs_lo, M_obs_up = df.M_obs, df.M_obs_lo, df.M_obs_up
	M_ratio = M_obs / (df.M_sim /100)	 # accuracy_ratio( M_obs, df.M_sim/100 )
	# mean_r = max( np.nanmean( M_ratio), 1/np.nanmean( M_ratio) )		# to have the form 1.#x
	qs = np.nanquantile( M_ratio, [0.16, 0.5, 0.84] )
	T_label = f'_T{Tavg :1.0f}K' if simple_M else '_T(r)'

	ptitle = 'Disk mass comparison' + run_name
	fig, ax = plt.subplots( figsize=(4,4), tight_layout=True )
	ax.axline( xy1=(0.0001, 0.0001), slope=1, ls='--', c='gray' )			# y=x identity
	if errors: 
		scatter_with_errors( ax=ax, x= df.M_sim/100, y=M_obs, y_lo=M_obs_lo, y_up=M_obs_up, fmt='o', facecolor='tab:red', marker_alpha=0.76 )
	else:	ax.scatter( x=df.M_sim/100, y=M_obs, marker='o', c='r', alpha=0.7)		# observed fluxes
	# ax.text( x=0.01, y=0.85, s= f'median accuracy: {qs[0] :1.1f}x \n$\sigma =${np.std(M_obs - df.M_sim/100) :1.1f}' + '[M$_{\odot}$]',
	# 	ha='left', va='center', transform=ax.transAxes, color='gray', fontsize=10, alpha=0.8)
	ax.text( x=0.01, y=0.90, s= f'T={Tavg :1.0f} K' if simple_M else 'T=T(r)',
		ha='left', va='center', transform=ax.transAxes, color='gray', fontsize=12, alpha=1.)
	ax.text( x=0.01, y=0.85, s= f'16%-84% accuracy: {qs[0] :1.1f}x - {qs[2] :1.1f}x',
		ha='left', va='center', transform=ax.transAxes, color='gray', fontsize=10, alpha=0.8)
	axlims = [ 3e-5, 1.2e-2]
	ax.set( xlabel= r'$ M_\mathrm{sim} $ [M$_{\odot}$]', ylabel=r'$ M_\mathrm{obs} $ [M$_{\odot}$]', xscale='log', yscale='log', xlim=axlims, ylim=axlims, aspect='equal', title=ptitle )
	# ax.axis( 'square')
	# ax.set_box_aspect(1)
	# ax.set_aspect('equal', adjustable='box')
	[ fig.savefig( f'Figures_{fig_ext.strip(".")}/' + ptitle.replace(' ', '_') + T_label + fig_ext , bbox_inches='tight') for fig_ext in ('.png', '.pdf') ]
	# plt.show()
	plt.close()

	if not simple_M: ratio_histogram( df.M_obs, df.M_sim/100, run_name, histcolor='tab:red')


def plot_radius_compare( df, res_limit, run_name, errors=True):
	'''
	Assuming R_obs is R_90, in [au]. 
	'''
	r_ratio_90 = df.R_obs / df.R_sim			# accuracy_ratio( df.R_obs, df.R_sim )
	r_ratio_95 = r_ratio_90 *1.14
	R_reslim = res_limit * 2.1436 / np.sqrt(8 * np.log(2))		# resolution limit in terms of R_90 radii, to compare apples with apples

	ptitle = 'Radius comparison' + run_name
	fig, ax = plt.subplots( figsize=(4,4), tight_layout=True)
	ax.fill_between( [0.01, R_reslim, 10], y1=[10, 10, R_reslim], y2=0.01, step='pre', facecolor='gray', alpha=0.16, label=r'$\theta_\mathrm{res}$' )
	ax.axline( xy1=(0.5, 0.5), slope=1, ls='--', c='gray', alpha=0.8 )		# y=x identity
	ax.scatter( x=df.R_sim *au_to_as, y=df.R_obs/1.42 *au_to_as, marker='o', c='r', label='$R_{68\%}$', alpha=0.2)
	if errors:
		scatter_with_errors( ax=ax, x=df.R_sim *au_to_as, y=df.R_obs*au_to_as, y_lo=df.R_obs_lo*au_to_as, y_up=df.R_obs_up*au_to_as,
					fmt='o', facecolor='g', label='$R_{90\%}$', marker_alpha=0.7 )
	else:
		ax.scatter( x=df.R_sim *au_to_as, y=df.R_obs *1 *au_to_as, marker='o', c='g', label='$R_{90\%}$', alpha=0.7, zorder=3.7)		# observed radii
	ax.scatter( x=df.R_sim *au_to_as, y=df.R_obs*1.14 *au_to_as, marker='o', c='b', label='$R_{95\%}$', alpha=0.2)
	ax.text( x=0.01, y=0.7, s=(f'obs/sim accuracy:\n $R_{{90\%}}$: {np.nanmedian( r_ratio_90) :1.1f}x'  #\nmedian accuracy $R_{{90\%}}$: {np.mean( r_ratio_90) :1.1f}x'
		f'\n $R_{{95\%}}$: {np.nanmedian( r_ratio_95) :1.1f}x'),
		ha='left', va='center', transform=ax.transAxes, color='k', fontsize=10, alpha=0.8)
	ax.set( xlabel= r'$ R_\mathrm{sim} $ [arcsec]', ylabel=r'$ R_\mathrm{obs} $ [arcsec]' , xscale='log', yscale='log',
		 title=ptitle, xlim=[0.05,2.4], ylim=[0.05, 2.4], aspect='equal' )
	# ax.grid( True, axis='x', alpha=0.5, linestyle=':')
	ax.legend()
	[ fig.savefig( f'Figures_{fig_ext.strip(".")}/' + ptitle.replace(' ', '_') + fig_ext , bbox_inches='tight') for fig_ext in ('.png', '.pdf') ]
	# plt.show()
	plt.close()

	ratio_histogram( df.R_obs, df.R_sim, run_name, histcolor='tab:green')


def plot_inc_compare( df, run_name):
	'''
	Assuming inc in [deg]. 
	'''
	qs = np.nanquantile( df.i_obs / df.i_sim, [0.16, 0.5, 0.84] )

	ptitle = 'Inclination comparison' + run_name
	fig, ax = plt.subplots( figsize=(4,4), tight_layout=True)
	ax.axline( xy1=(1, 1), slope=1, ls='--', c='gray' )		# y=x identity
	scatter_with_errors( ax=ax, x=df.i_sim, y=df.i_obs, y_lo=df.i_obs_lo, y_up=df.i_obs_up, fmt='o', facecolor='C1', marker_alpha=.9, err_alpha=0.27 )
	# ax.scatter( x=inc, y=df.i_obs, marker='o', c='orange', alpha=0.8)
	ax.text( x=0.01, y=0.85, s= f'16%-84% accuracy: {qs[0] :1.1f}x - {qs[2] :1.1f}x',
		ha='left', va='center', transform=ax.transAxes, color='gray', fontsize=10, alpha=0.8)
	ax.set( xlabel= r'$ i_\mathrm{sim} $ [deg]', ylabel=r'$ i_\mathrm{obs} $ [deg]', aspect='equal', xlim=[0,90], ylim=[0,90], title=ptitle )
	[fig.savefig( f'Figures_{fig_ext.strip(".")}/' + ptitle.replace(' ', '_') + fig_ext , bbox_inches='tight') for fig_ext in ('.png', '.pdf')]
	# plt.show()
	plt.close()


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
	table.open( f'{diskname}.{config_name}.noisy.image.pbcor' )		# noisy image, with pbcor the central target is slightly under corrected?
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
	return F_v		# [Jy] integrated flux observed


def plot_correlations( df, run_name='', logfit=True):
	y = df.epsilon_M  	# mass relative error
	xlabs = ['M_obs', 'M_sim', 'LI0_d', 'LI0_env', 'Mes', 'R_obs', 'R_sim', 'Ri', 'p_idx', 'i_obs', 'i_sim', 'L_tot']		# 
	
	plt.rcParams.update({ 'font.size':8, 'legend.fontsize':6, 'figure.dpi':300})
	ptitle = 'Mass error correlations' + run_name
	fig, axs = plt.subplots( 3, int( np.ceil(len(xlabs)/3)), sharey=True, squeeze=False, tight_layout=True )
	axs = axs.flatten()
	fig.suptitle( ptitle )

	for i in range( len(xlabs)): 
		axs[i].scatter( df[xlabs[i]], y, alpha=0.7 )
		axs[i].set(  xlabel= xlabs[i])
		xscale = 'log' if (logfit and (i not in [2,3,8,9,10])) else 'linear'	# not all log anyway
		axs[i].set( xscale=xscale, yscale='linear')
	fig.supylabel( r'$\delta_M$', fontsize=12 )
	# fig.subplots_adjust( wspace=0.001)
	# llab = '_log' if logfit else ''
	[ fig.savefig( f'Figures_{fig_ext.strip(".")}/' + ptitle.replace(' ', '_') + fig_ext , bbox_inches='tight') for fig_ext in ('.png', '.pdf') ]
	plt.show()

	# y = df.epsilon_M  	# mass relative error
	# ptitle = 'Mass error correlations_alt'
	# fig, ax = plt.subplots( tight_layout=True )
	# fig.suptitle( ptitle )
	# ax.scatter(  df.Mes / df.M_sim , y, alpha=0.7 )
	# ax.set(  xlabel= 'Mdisk / Menv', xscale='linear')
	# fig.supylabel( r'$\delta_M$', fontsize=12 )
	# # [ fig.savefig( f'Figures_{fig_ext.strip(".")}/' + ptitle.replace(' ', '_') + fig_ext , bbox_inches='tight') for fig_ext in ('.png', '.pdf') ]
	# plt.show()


def produce_truths_df():
	'''From Tungs data export a dataframe with the simulation truths of my interest. '''
	import h5py
	catalog = 'disk_01440_rmax_500_f_2_rho_3.8346e-15_thermal_False.h5'
	hf = h5py.File( catalog, 'r')
	disks = {}
	for k in hf.attrs.keys():	#Extract the disk quantities
		disks[k] = hf.attrs[k]

	Rsim = []; Rmean=[]; Msim = []; Mcyl =[]; Lint = []; Lacc = []; dTemp1 = []; dTemp2 = []; multip =[]; 
	angs_x = []; angs_y = []; angs_z = []; hr = []; Mstar = []; age =[]; Menv = [] 
	ids = disks['list_of_disks']

	for i_d in ids:
		prefix = 'disk_' + str(i_d).zfill(5)
		directions = disks[prefix + '_direction'] 	# coordinates (x, y, z) of the normal vector of the disk
		angs_x.append( np.arccos( directions[0]) *180/np.pi)
		angs_y.append( np.arccos( directions[1]) *180/np.pi)    
		angs_z.append( np.arccos( directions[2]) *180/np.pi)    
		Rsim.append( disks[prefix + '_radius'] )
		# Rmean.append( disks[prefix + '_mean_radius'] )	# useless
		Msim.append( disks[prefix + '_mass'] )				# disk mass
		Mcyl.append( disks[prefix + '_mass_cyl'] )			# env mass ?
		Menv.append( disks[prefix + '_mass_env_1000'] )		# env mass ?
		Lint.append( disks[prefix + '_star_lum'] )
		Lacc.append( disks[prefix + '_star_acclum'] )
		dTemp1.append( disks[prefix + '_Temp_mid'] )		# mid, mavg o simple ?
		dTemp2.append( disks[prefix + '_Temp_mavg'] )
		multip.append( disks[prefix + '_multiplicity'])
		# hr.append( disks[prefix + '_hoverr'] )				# scale height?
		Mstar.append( disks[prefix + '_sink_mass'] )		# star mass
		age.append( disks[prefix + '_sink_age'] )

	dfT = pd.DataFrame( np.array([Msim, Mcyl, Menv, Rsim, Rmean, Lint, Lacc, dTemp1, dTemp2, multip, hr, Mstar, age, angs_x, angs_y, angs_z]).T, 
		columns=['M_disk', 'M_cyl', 'M_env', 'R_disk', 'R_mean', 'L_int', 'L_acc', 'Tmid_disk', 'Tmavg_disk', 'multiplicity', 'hr', 'M_star', 'age', 'i_yz', 'i_xz', 'i_xy'], index=ids)
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


def env_mass_annuli( v_obs, LI0, Ri, p_idx, Rmax, kappa, Ltot):
	'''Compute envelope mass in thin approximation but summing on annuli up to Rmax, with Tung Temp profile and fitted I0_env.'''
	r_grid = np.logspace( -10, np.log10(Rmax), 400)		# [rad]
	dF_grid = np.diff( plummer_integral( 10**LI0, Ri, p_idx, r_grid ), axis=0 )		# annulus-integrated flux density [Jy]
	rmid = ( r_grid[:-1] + r_grid[1:] ) / 2 * dist.to_value(u.au)	# midpoint radii [au]
	dM = ( dist.cgs.value )**2 / kappa * dF_grid  / planck_bbody( v_obs, T=temp_profile_envs( lum=Ltot, r=rmid ) )
	Mtot = dM.sum( axis=0) / const.M_sun.cgs.value * 1e-23		# [Msun]
	return Mtot, dF_grid.sum( axis=0) 

def mass_annuli_calc( v_obs, LI0_d, sma, R_obs, kappa, Ltot ):
	'''
	Compute disk mass in thin approximation but summing on annuli over the Robs, with Tung Temp profile and fitted I0_disk. Handles 3 component vectors with errors.
	'''
	r_grid = np.logspace(-10, np.log10(R_obs), 100)		# [rad]
	dF_grid = np.diff( gauss_flux_integral( 10**LI0_d, sma, r_grid ), axis=0 )	# annulus-integrated flux density [Jy]
	rmid = ( r_grid[:-1] + r_grid[1:] ) / 2 * dist.to_value(u.au)	# midpoint radii [au]
	dM = ( dist.cgs.value )**2 / kappa * dF_grid  / planck_bbody( v_obs, T=temp_profile_Tung( lum=Ltot, r=rmid ) )
	Mtot = dM.sum( axis=0 ) / const.M_sun.cgs.value * 1e-23		# [Msun]
	return Mtot, dF_grid.sum( axis=0) 


def main_analysis( targetslist, wle, results_dir, config_name, run_name, T_avg=122, figures=True):
	'''
	Analyse the bestfit parameters of the whole sample and the derived quantities, comparing them to the simulation truths.   
	'''
	v_obs = 299792458.0/wle			# [Hz]		# 100 *1e9   obs frequency
	k_sim = 0.54 if round(wle*1e3)==3 else 0.138	# opTool original opacity for the simulation truths
	if round(wle*1e3)==1: k_sim = 3.5
	k_obs = k_sim # kappa_empir( v_obs, beta=1.5)	# 1.5 good for both 3mm and 7mm (not 0.9mm) # for the OBS # [cm2 / g]

	disklist = sorted( glob.glob( results_dir + 'disk*') )
	print( len(disklist), 'files found')
	truths_df = pd.read_csv( truth_path, sep='\t', index_col=0 ) #.loc[OKlist]	# load my simulation truths file
	paramlist = []

	for fpath in disklist:
		os.chdir( fpath )
		diskID = fpath.strip( results_dir ).strip('disk')		# NN_xx kind
		disk_n = int(diskID.strip( '_yzx'))
		
		if disk_n in targetslist:
			try:
				# read the bestfit params from file for I0 and sma
				pars = np.loadtxt( 'bestfit_params.txt')	# NOW each par is an array([best, 16%, 84%]) !!
				LI0_d = pars[0]                   			# disk peak intensity 	Log[Jy/sr]
				LI0_env = pars[1]							# envelope peak intensity	Log[Jy/sr]
				Ri = np.deg2rad( pars[3] /3600)				# inner env radius	[arcsec --> rad]
				sma = np.deg2rad( pars[2] /3600)     		# gauss disk sigma	[arcsec --> rad]
				i_obs = pars[-4]							# disk inclination [deg]
				p_idx = pars[5]	
				Rout = pars[4] * Ri

				R_68 = sma * np.sqrt( -2 * np.log(1-0.68))      # 68% radius  [rad]
				R_90 = R_68 * 1.42                              # 90% radius
				R_95 = R_68 * 1.62
				R_obs = R_90     # as Tung 
				l_star = truths_df.loc[ disk_n ][['L_acc', 'L_int']].sum()			# L_acc + L_int [Lsun]

				F_v_simple = gauss_flux_integral( 10**LI0_d, sma, 1*R_obs)      # observed flux density of DISK [Jy]
				M_obs_simple = F_v_simple *1e-23 * ( dist.cgs.value )**2 / (k_obs * planck_bbody( v_obs, T_avg) )  / const.M_sun.cgs.value	# [Msun] 
				M_obs, F_v = mass_annuli_calc( v_obs, LI0_d, sma, R_obs, k_obs, l_star)				
				Fv_count = count_flux_sources( 'disk'+ diskID, nRMS=7, config_name=config_name, results_dir=results_dir ) 
				F_v_thicc = thick_flux( v_obs, dist, R_obs[0], l_star=l_star)			# theoretical fully thick disk flux

				M_env_o, Fv_env = env_mass_annuli( v_obs, LI0_env, Ri, p_idx, R_95, k_obs, l_star)

			except: 
				print('No bestfit params found for disk', diskID)
				M_obs = R_obs = F_v = i_obs = Ri = p_idx = LI0_d = LI0_env = F_v_simple = M_obs_simple = M_env_o = Fv_env = Rout = [np.nan, np.nan, np.nan]
				F_v_thicc = Fv_count = np.nan
			
			finally:
				M_sim = truths_df.loc[ disk_n ]['M_disk']		# [Msun] total disk mass from simulations (gas)
				epsilon = (M_obs *100 - M_sim) / M_sim			# obs - truth normalised discrepancy (factor 100 dust-to-gas)
				Menv_sim = truths_df.loc[ disk_n ]['M_env']		# [Msun] env mass from simulations (gas) within 1000 au ??
				Mcyl = truths_df.loc[ disk_n ]['M_cyl']			# [Msun] 
				R_sim = truths_df.loc[ disk_n ]['R_disk']		# [au]
				R_obs = (R_obs * dist).to_value( u.au )			# rad to [au]
				Ri = (Ri * dist).to_value( u.au )				# rad to [au]
				i_sim = truths_df.loc[ disk_n ][ 'i' + diskID.strip( str(disk_n) ) ]
				i_sim = 180 - i_sim if i_sim > 90 else i_sim 	# all between 0 and 90 deg
				L_tot = truths_df.loc[ disk_n ][['L_acc', 'L_int']].sum()		# L_acc + L_int [Lsun]
				F_sim_thin = M_sim/100 * const.M_sun.cgs.value * k_sim * planck_bbody( v_obs, T=T_avg) / dist.cgs.value**2  *1e23	

				paramlist.append( [diskID, R_obs, R_sim, Ri, p_idx, M_obs_simple, M_obs, M_sim, epsilon, Menv_sim, Mcyl, 
					LI0_d, LI0_env, F_v, F_v_simple, Fv_count, F_v_thicc, i_obs, i_sim, L_tot, F_sim_thin, M_env_o, Fv_env, Rout] )

	res_df = pd.DataFrame( paramlist, 
				columns=['source', 'R_obs', 'R_sim', 'Ri', 'p_idx', 'M_obs_simple', 'M_obs', 'M_sim', 'epsilon_M', 'Mes', 'Mcyl',
				'LI0_d', 'LI0_env', 'F_obs', 'Fv_simple', 'Fv_count', 'F_thick', 'i_obs', 'i_sim', 'L_tot', 'Fsim_thin', 'Meo', 'Fv_env', 'Rout']
			).set_index('source')
	
	# some columns contain arrays of length 3 with uncertainties, but better to give each one an independent column of the DataFrame
	for col in ['R_obs', 'Ri', 'p_idx', 'M_obs_simple', 'M_obs', 'LI0_d', 'LI0_env', 'F_obs', 'Fv_simple', 'i_obs', 'Meo', 'Fv_env']:
		res_df[ [col, col+'_lo',col+'_up'] ] = pd.DataFrame( res_df[col].tolist(), index=res_df.index)
	
	os.chdir( results_dir )
	res_df.to_csv( f'analysis_results-{run_name}.txt', sep='\t') #, float_format='%.2e')
	# res_df = pd.read_csv( f'analysis_results-{run_name}.txt', sep='\t', index_col='source')	# to load it
	
	if figures:
		# plot_opacity()
		plot_Fv_compare( res_df, v_obs, k_sim, Tavg=T_avg, rdata='sim', run_name=run_name )
		plot_inc_compare( res_df, run_name )
		plot_mass_compare( res_df, run_name, T_avg, simple_M=True ) ; plot_mass_compare( res_df, run_name, T_avg, simple_M=False )
		plot_mass_env( res_df, run_name )
		theta = alma_resolution( wle=wle, config_name=config_name)
		plot_radius_compare( res_df, theta, run_name )
		# thick_sim_inspo( truths_df, k_sim, T_avg, v_obs, run_name)
		plt.close() 
	return res_df



# def crop_image( img, centre=None, margins=[100, 100] ):
# 	'''Select a subimage of margins pixels around the centre (odd size).'''
# 	if centre is None:      	# use the middle of the image
# 		centre = (np.array( img.shape)/2 ).astype(int)
# 	if margins[0] > min( centre[0], img.shape[0] - centre[0]):
# 		print( 'margins exceed original image boundary, no crop possible.\n')
# 		return img
# 	return img[ centre[0] - margins[0] : centre[0] + margins[0] +1, centre[1] - margins[1] : centre[1] + margins[1] +1]

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
	ptitle = 'SNR plot' + run_name
	fig, ax = plt.subplots( figsize=(6,4), constrained_layout=True)
	dff = df.reset_index()
	dff.plot( xticks=dff.index, rot=90, logy=True, ax=ax, marker='o')
	ax.set_xticklabels( df.index)
	# ax.axhline( y=[0.002], color='gray', ls=':')
	ax.axhline( y=10, color='gray', ls='--')
	ax.axhline( y=100, color='gray', ls='-')
	ax.grid( True, axis='x', alpha=0.5, linestyle=':')
	ax.set( title=ptitle)
	[fig.savefig( f'Figures_{fig_ext.strip(".")}/' + ptitle.replace(' ', '_') + fig_ext, bbox_inches='tight') for fig_ext in ('.png', '.pdf') ]
	# plt.show()
	plt.close()


def assess_SNR( wle, results_dir, config_name, run_name ):
	'''
	Evaluate the SNR of the cleaned image across the entire sample in results_dir. 
	'''
	fitslist = sorted( glob.glob( results_dir + 'disk*') )
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

		noise = min_bkg_rms( img )			# the minimum rms from bkg patches
		# noise = rms( img )				# the rms of the entire image including target source

		snr = peak_beam / noise
		SNRs.append( [projectname.strip( 'disk' ), snr, peak, peak_beam, noise] )
		table.close()

	df = pd.DataFrame( SNRs, columns=['source', 'SNR', 'max peak', 'beam peak', 'noise']).set_index('source')
	os.chdir( results_dir )
	df.to_csv( f'Peak-beam_SNR_{run_name}.txt', sep='\t') #, float_format='%.2e')
	print( '\nMedian SNR of run: \t', np.nanmedian( df.SNR) )
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

	ptitle = 'M_disk empirical fit' + run_name
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
	[ fig.savefig( f'Figures_{fig_ext.strip(".")}/' + ptitle.replace(' ', '_') + llab + fig_ext , bbox_inches='tight') for fig_ext in ('.png', '.pdf') ]
	plt.show()
	return popt



# def inspect_plots( two_comp=True, sampler=None, burnin=None, walksigma=4, results_dir=''):
# 	'''
# 	inspect MCMC plots (chains + corner).
# 	'''
# 	labels_gauss = ['Log($I_0$)', 'Log(Ie)', '$\sigma$', '$i$', 'PA', 'dRA', 'dDec']
# 	labels_2c = [r'Log($I_{0d}$)', r'Log($I_{0e}$)', '$\sigma$', 'R_i', 'p_idx', '$i$', 'PA', 'dRA', 'dDec']
# 	labs_mc = labels_2c if two_comp else labels_gauss

# 	disklist = sorted( glob.glob( results_dir + 'disk*') )
# 	if disklist == []:    
# 		print('NO FILES FOUND, check again the folder path!')
# 		sys.exit()
# 	print( len(disklist), 'files found')

# 	for fpath in disklist:
# 		os.chdir( fpath )
# 		diskname = fpath.replace( results_dir, '' )
# 		print( '\nInspecting:  ', diskname)
# 		# disk_n = int(diskname.strip( '_yzx'))
# 		# os.chdir( diskname )

# 		if sampler is None:
# 			sampler = emcee.backends.HDFBackend( f'{diskname}__sampler.h5', read_only=True )	# will throw store==True error if diskname is wrong
# 		nsteps = sampler.get_chain().shape[0]
# 		if burnin is None:
# 			burnin = nsteps//3
# 		bestfit = mcmc_plots( sampler, labels=labs_mc, burn_in=burnin, walk_clip_thresh=walksigma, figures=True )




def visib_ratios_plot( model='full', quantity='Re', binsize=50e3, max_baseline=5e5, targetslist=OKlist):
	'''
	Visualise for ALL targets in our sample the ratios of quantity between 1,3,7mm as function of the baseline. 
	'''
	import uvplot as uvp
	# mpl.use('macosx')
	from scipy.interpolate import Akima1DInterpolator
	plt.rcParams.update({ 'font.size':9, 'legend.fontsize':7, 'figure.dpi':100})	
	mpl.style.use('fast')

	def plot_quantity( quant, uvtab):
		if quant=='Re':
			return uvtab.bin_re
		elif quant=='Im':
			return uvtab.bin_re
		elif quant=='amp':
			return np.arctan2( uvtab.bin_im, uvtab.bin_re )
		elif quant=='mod':
			return np.sqrt( uvtab.bin_im**2 + uvtab.bin_re**2 )
		
	# prefix =  '/home/PERSONALE/gabriele.columba/run/results/' # '/Users/gcolumba/PostDoc_Mac/sshfs_dir/' 
	resdir_7mm = savedir_prefix + '7mm/run_10800s_2c_xsrc/'
	resdir_3mm = savedir_prefix + '3mm/run_3600s_2c_xsrc/'
	resdir_1mm = savedir_prefix + '1mm/run_300s_2c_xsrc/'
	disk_dirs = sorted(glob.glob( resdir_1mm + 'disk*'))
	n_disks = len(disk_dirs)
	ncols = 10 ; nrows = int(np.ceil( n_disks / ncols))
	ptitle = f'Ratios of {quantity}(V) - {model} model'
	fig, axes = plt.subplots( nrows, ncols, figsize=(2*ncols, 2.35*nrows), squeeze=False, sharex=True, sharey='row')
	axes = axes.flatten()

	wles = [8.9e-4, 3e-3, 7e-3] ; dirs = [resdir_1mm, resdir_3mm, resdir_7mm]
	for d in range( n_disks):	# n_disks
		uvtabs = [0,0,0] ; comptabs = [0,0,0]
		diskname = os.path.basename( disk_dirs[d] )		# "diskNN_xx"	
		if int( diskname[4:6]) in targetslist:
			try:		# Load uvtable using uvplot
				for i in range(3):
					uvtabs[i] = uvp.UVTable( filename= dirs[i] + diskname +'/uvtab.txt', wle=wles[i], columns=uvp.COLUMNS_V0)		# mock-obs data
					if model != 'full':
						mod_vis = [0,0]
						with open( dirs[i] + diskname + '/visib_disk+env.npy', 'rb') as f:		# this requires two separate np.load calls to read back the two arrays
							mod_vis = [np.load( f), np.load( f)] 			# disk_vis, env_vis
						mod_i = 0 if model == 'env' else 1
						comptabs[i] = uvp.UVTable( uvtable=[uvtabs[i].u*wles[i], uvtabs[i].v*wles[i], mod_vis[mod_i].real, mod_vis[mod_i].imag, uvtabs[i].weights], wle=wles[i], columns=uvp.COLUMNS_V0 )
						# disktabs[i] = uvp.UVTable( filename= dirs[i] + diskname+ '/uvtab_disk.txt', wle=wles[i], columns=uvp.COLUMNS_V0 )
						comptabs[i] = comptabs[i].uvcut( maxuv=max_baseline)	; comptabs[i].uvbin( binsize)	# bin it before or AFTER the subtraction ?
						del mod_vis
					uvtabs[i] = uvtabs[i].uvcut( maxuv=max_baseline)	; uvtabs[i].uvbin( binsize)
			except Exception as e:
				print(f"\nCould not load uvtable for {diskname}: {e}\n")
				continue
			
			if model == 'full':
				q1, q3, q7 = [ plot_quantity( quantity, uvtabs[t]) for t in range(3) ]
			else: # model == 'env':
				q1 = uvtabs[0].bin_re - comptabs[0].bin_re	# add other quantities choice
				q3 = uvtabs[1].bin_re - comptabs[1].bin_re 	# 3mm env_re
				q7 = uvtabs[2].bin_re - comptabs[2].bin_re
			# else: 
			# 	print( '\nmodel can only be ["full", "disk", "env"], input option not recognised' )
			uvdist3 = uvtabs[1].bin_uvdist  # np.where( qty > 0 , uvtab3.bin_uvdist, np.nan)	# reference baselines distances
			# q3 = qty # np.where( qty > 0 , qty, np.nan) 			# reference REAL values to compute ratios
			q1 = Akima1DInterpolator( uvtabs[0].bin_uvdist, q1 )( uvdist3 )	# interpolate the real part where the ref value are binned
			q7 = Akima1DInterpolator( uvtabs[2].bin_uvdist, q7 )( uvdist3 )	#  np.arctan2( uvtab1.bin_im ,
			# re1[re1 <= 0] = np.nan ; re7[re7 <= 0] = np.nan ; 		# disregard negative Re fluxes
			ratio31 = q1 / q3
			ratio73 = q3 / q7
			a13 = - np.log10( ratio31) / np.log10( 0.89 / 3 )		# minus sign because i'm dividing for wavel, not frequency
			a37 = - np.log10( ratio73) / np.log10( 3 / 7 )

			# fig, axes = plt.subplots()
			axes[d].axhline( y=3.5, ls='--', c='gray' )			# y=3.5 alpha marker
			axes[d].plot( uvdist3 *1e-3, a37, c='tab:orange', ls='-', lw=1.5, label='3mm/7mm' )		# all three ratios in same subplot for each target
			axes[d].plot( uvdist3 *1e-3, a13, c='tab:blue', ls='-', lw=1.5, label='0.9mm/3mm', alpha=0.85 )
			axes[d].set( xscale='log') #, yscale='log')#, ylim=[1e-1,1e3]) ; 
			axes[d].set_title( diskname, fontsize=8)

			del uvtabs, comptabs, q1, q3, q7
			# plt.show()

	for ax in axes[n_disks:]: ax.set_visible(False)		# hide unused axes
	supylab = r'$\alpha$ index'	# 'Re(V) [Jy]'
	fig.subplots_adjust( wspace=0.001)	# hspace=0.001,
	fig.supylabel( supylab, weight='bold', x=0.08, fontsize=12 )
	fig.supxlabel('uv-distance [k$\lambda$]', fontsize=12 )		#, weight='bold'
	axes[0].legend()
	fig.savefig( ptitle.replace(' ', '_') + '.pdf' , bbox_inches='tight')
	# [ fig.savefig( f'Figures_{fig_ext.strip(".")}/' + ptitle.replace(' ', '_') + fig_ext , bbox_inches='tight') for fig_ext in ('.png', '.pdf') ]
	plt.close()
	print(' Mega uv plot saved')


def collective_uvplot( wle, results_dir, run_name, two_comp, binsize=50e3, targetslist=OKlist):
	'''
	Make uvplots of all regressed targets in one figure
	'''
	from visibfit_functions import galario_model, get_galargs
	from galario import deg, arcsec
	import uvplot as uvp
	# mpl.use('macosx')
	plt.rcParams.update({ 'font.size':7, 'legend.fontsize':6, 'figure.dpi':100})	
	mpl.style.use('fast')

	disk_dirs = sorted(glob.glob( results_dir + 'disk*'))
	n_disks = len(disk_dirs)
	ncols = 11 ; nrows = int(np.ceil( n_disks / ncols))
	ptitle = 'Collective uvplot' + run_name
	fig, axes = plt.subplots( nrows, ncols, figsize=(1.7*ncols, 2*nrows), squeeze=False, sharex=True, sharey=False, layout='tight')
	fig.suptitle( ptitle, fontsize=10)
	axes = axes.flatten()

	for d in range( n_disks):	# n_disks
		diskname = os.path.basename( disk_dirs[d] )		# "diskNN_xx"
		if int( diskname[4:6]) in targetslist:
			try:		# Load uvtable using uvplot
				os.chdir( disk_dirs[d] )
				bestfit = np.loadtxt('bestfit_params.txt')[:,0]		# only take the best values (no errors)
				inc, PA, dRA, dDec = bestfit[-4:]
				inc *= deg ; PA *= deg ; dRA *= arcsec ; dDec *= arcsec ;		# convert to [rad] !
				galargs = get_galargs( wle=wle)
				chi2, vis_mod = galario_model( pars= bestfit, galargs=galargs, two_comp=two_comp )[-2:]
				red_chi2 = chi2/(galargs[2] - len(bestfit)) 	# chi2/(42*(42-1)/2 - len(bestfit))	# with 42 antennas
				u, v, Re_obs, Im_obs, w = galargs[-5:]
				axins = axes[d].inset_axes( [0,-0.2 , 1, 0.2] )
				# observations uv-plot !
				uv = uvp.UVTable( uvtable=[u*wle, v*wle, Re_obs, Im_obs, w], wle=wle, columns=uvp.COLUMNS_V0 )
				uv.apply_phase( -dRA, -dDec)         # center the source on the phase center
				uv.deproject( inc=inc/deg, PA=PA/deg, inplace=True)
				uv.uvbin( binsize)		# , 'zorder':1.9
				mask = uv.bin_count != 0 # slice(None)
				uvdist = uv.bin_uvdist[mask]/1000
				data_dict = {'fmt':'o', 'ms':3, 'color':'k', 'linewidth':0, 'capsize':1.2, 'capthick':1, 'ecolor':'gray', 'elinewidth':0.2, 'label':'Data', 'alpha':0.7}
				axes[d].errorbar( x=uvdist, y=uv.bin_re[mask], yerr=uv.bin_re_err[mask], **data_dict)
				axins.errorbar( x=uvdist, y=uv.bin_im[mask], yerr=uv.bin_im_err[mask], **data_dict)
				del uv
				# model uv-plot : disk (+ env)
				uv_mod = uvp.UVTable( uvtable=[u*wle, v*wle, vis_mod.real, vis_mod.imag, w], wle=wle, columns=uvp.COLUMNS_V0 )
				uv_mod.apply_phase( -dRA, -dDec)    # center the source on the phase center
				uv_mod.deproject( inc=inc/deg, PA=PA/deg, inplace=True)
				uv_mod.uvbin( binsize )
				# uvdist = uv_mod.bin_uvdist[mask]/1000		# should be the same as for obs
				model_dict = { 'ls':'-', 'color':'r', 'linewidth':1.3, 'label':'Model', 'alpha':1}	
				axes[d].errorbar( uvdist, uv_mod.bin_re[mask], **model_dict)
				axins.errorbar( uvdist, uv_mod.bin_im[mask], **model_dict)
				axes[d].text( x=0.92, y=0.92, s= fr'$\chi^2_\nu$={red_chi2 :.3f}', ha='right', va='center', transform=axes[d].transAxes, color='k', alpha=.8)
				del uv_mod

				if two_comp:
					colors, labs, lss = ['tab:blue', 'tab:green'], ['disk','envelope'], ['--', ':']
					with open( disk_dirs[d] + '/visib_disk+env.npy', 'rb') as f:		# this requires two separate np.load calls to read back the two arrays
						mod_vis = [np.load( f), np.load( f)] 		# disk_vis, env_vis
					for i in range( len( mod_vis)):				# separately plot disk and envelope contributions
						uv_mod = uvp.UVTable( uvtable=[u*wle, v*wle, mod_vis[i].real, mod_vis[i].imag, w], wle=wle, columns=uvp.COLUMNS_V0 )
						uv_mod.apply_phase( -dRA, -dDec)     	# center on the phase center
						uv_mod.deproject( inc=inc/deg, PA=PA/deg, inplace=True)
						uv_mod.uvbin( binsize ) #; mask = slice(None) #uv_mod.bin_count != 0
						# uvdist = uv_mod.bin_uvdist[mask] / 1000 
						comp_dict = { 'ls':lss[i], 'color':colors[i], 'lw':1.2, 'label':labs[i], 'alpha':0.95}
						axes[d].errorbar( uvdist, uv_mod.bin_re[mask],  **comp_dict)
						axins.errorbar( uvdist, uv_mod.bin_im[mask], **comp_dict)
					del uv_mod, uvdist

				axes[d].set_title( diskname, fontsize=8)
				axes[d].set( xscale='log', yscale='log') ; axins.set( xscale='log')
				if axes[d].get_ylim()[0] < 1e-4: axes[d].set( ylim=[1e-4, axes[d].get_ylim()[1]] )		# force lower ylim at 1e-5
				# axes[d].tick_params(axis='both', left=True, top=False, right=False, bottom=False, labelleft=True, labeltop=False, labelright=False, labelbottom=False)
				# axins.tick_params(axis='both', left=False, top=False, right=False, bottom=True, labelleft=False, labeltop=False, labelright=False, labelbottom=True)

			except Exception as e:
				print(f"\nCould not do for {diskname}: {e}\n")
				continue
		else:
			axes[d].set_visible(False)

	for ax in axes[n_disks:]: ax.set_visible(False)		# hide unused axes
	supylab = 'Re(V)'
	fig.supylabel( supylab, x=0.0, fontsize=10 )
	fig.supxlabel('uv-distance [k$\mathrm{\lambda$}]', fontsize=10 )		#, weight='bold'
	axes[0].legend()
	fig.savefig( results_dir + ptitle.replace(' ', '_') + '.pdf' , bbox_inches='tight')
	# [ fig.savefig( f'Figures_{fig_ext.strip(".")}/' + ptitle.replace(' ', '_') + fig_ext , bbox_inches='tight') for fig_ext in ('.png', '.pdf') ]
	# plt.show()
	plt.close()
	print('Collective uvplot saved')


def collective_residuals_plot( results_dir, run_name, as_margin=2, targetslist=OKlist):
	'''
	Make uvplots of all regressed targets in one figure
	'''
	plt.rcParams.update({ 'font.size':7, 'legend.fontsize':7, 'figure.dpi':200})
	mpl.style.use('fast')

	disk_dirs = sorted(glob.glob( results_dir + 'disk*'))
	n_disks = len(disk_dirs)
	ncols = 11 ; nrows = int(np.ceil( n_disks / ncols))
	casa_table = cto.table()
	casa_table.open( disk_dirs[1] + '/bestmod/residuals.image' )		# just read once, it's the same for given run
	pixscale = abs( casa_table.getkeyword('coords')['direction0']['cdelt'][0])		# [rad/pix] of noisy image
	pixcut_m = int( as_margin / np.rad2deg(pixscale) / 3600 )			# margin in pixel

	ptitle = 'Normalised residuals' + run_name
	fig, axes = plt.subplots( nrows, ncols, figsize=(2*ncols, 2*nrows), layout='constrained')
	fig.suptitle( ptitle, fontsize=10)
	axes = axes.flatten()

	for d in range( n_disks):	# n_disks
		diskname = os.path.basename( disk_dirs[d] )		# "diskNN_xx"
		if int( diskname[4:6]) in targetslist:
			print( 'reading residuals of ', diskname)
			try:
				# os.chdir( disk_dirs[d] )
				res_img = np.load( disk_dirs[d] + '/bestmod/best_residuals.npy')		# Load normalised residuals
				res_crop = crop_image( res_img, margins=[ pixcut_m, pixcut_m] )
				cb = axes[d].imshow( res_crop.T, origin='lower', cmap='RdBu_r', norm=mpl.colors.CenteredNorm( vcenter=0), aspect='equal', interpolation=None ) 
				axes[d].set_title( diskname, fontsize=9)
				axes[d].axis( 'off' )
				fig.colorbar( cb, cax= axes[d].inset_axes( [1,0 , 0.07, 1] ), ax=axes[d] ) 	# shrink=0.8, pad=0.00,

			except Exception as e:
				print(f"\nCould not do for {diskname}: {e}\n")
				continue
		else:
			axes[d].set_visible(False)
	for ax in axes[n_disks:]: ax.set_visible(False)		# hide unused axes

	fig.savefig( results_dir + ptitle.replace(' ', '_') + '.pdf' , bbox_inches='tight')
	# plt.show()
	plt.close()
	print('Collective residuals saved')






if __name__=='__main__':

	parser = argparse.ArgumentParser()		# parsing the name of the disk file to read
	parser.add_argument('RT_wavel', type=int, help='obs wavelength (3000 or 7000 [um]) (default: 3000)')
	parser.add_argument('-config', type=str, help='ALMA antenna configuration (default: 11.7)')
	parser.add_argument('-Texp', type=int, default=3600, help='exposure time (default: 3600s)')
	parser.add_argument('-2c', action='store_true', help='use two-component model (default: False)')
	parser.add_argument('-fullsamp', action='store_true', help='analyse all OK targets (default: False)')
	parser.add_argument('-monosrc', action='store_true', help='do NOT use multi-source fit and clip the extra sources (default: False)')
	parser.add_argument('-Tavg', type=int, default=122, help='average temperature of disks for flux-mass conversion (default: 122K)')
	# parser.add_argument('-simple_M', action='store_true', help='calc mass with simplest thin case approx (default: False)')
	args = vars( parser.parse_args() )

	model_comps = '2c' if args['2c'] else '1c'
	xsrc_flag = 'mono' if args['monosrc'] else 'xsrc'
	wle = float(args["RT_wavel"]) *1e-6		# [m]	assuming wle is exact as names
	folder_wle = f'{round(wle*1e3)}mm/'
	savedir = savedir_prefix + folder_wle + f'run_{args["Texp"]}s_{model_comps}_{xsrc_flag}/' 		# results directory name
	config_name = 'alma.cycle' + args['config']
	run_suffix = f'-{ folder_wle.strip("/") } {args["Texp"]}s {model_comps}'
	os.makedirs( savedir + 'Figures_png/', exist_ok=True ) ; os.makedirs( savedir + 'Figures_pdf/', exist_ok=True )

	assess_SNR( wle=wle, results_dir=savedir, config_name=config_name, run_name=run_suffix )
	analist = OKlist if args["fullsamp"] else prettylist
	rdf = main_analysis( analist, wle=wle, results_dir=savedir, config_name=config_name, run_name=run_suffix, T_avg=args['Tavg'], figures=True )
	# fit_Mobs( results_dir=savedir, run_name=run_suffix, logfit=True)
	# inspect_plots( two_comp=args['2c'], results_dir=savedir )
	# plot_correlations( Rdf, run_name=run_suffix )
	collective_uvplot( wle, results_dir=savedir, run_name=run_suffix, two_comp=args['2c'])
	# collective_residuals_plot( results_dir=savedir, run_name=run_suffix )
	# visib_ratios_plot( model='full', quantity='mod')



# Texp = 3600
# model_comps = '2c'
# xsrc_flag = 'xsrc'
# wle = 0.003
# folder_wle = f'{round(wle*1e3)}mm/'
# savedir = savedir_prefix + folder_wle + f'run_{Texp}s_{model_comps}_{xsrc_flag}/'
# run_name = f' - { folder_wle.strip("/") }  {Texp}s  {model_comps}'
# config_name = 'alma.cycle11.7' 
# diskname = 'disk53_xz'

# rdf = pd.read_csv( f'analysis_results-{run_name}.txt', sep='\t', index_col='source')
# # plot_correlations( rdf, run_name )

# ptitle = 'Mass env'
# fig, ax = plt.subplots( )
# fig.suptitle( ptitle )
# ax.scatter(  rdf.Mcyl , rdf.Fv_env, alpha=0.7 )
# [ax.text( s=rdf.index[i], x=rdf.Mcyl[i],  y=rdf.Fv_env[i], horizontalalignment='left', verticalalignment='top', fontsize=5 ) for i in range(len(rdf)) ]
# ax.set(  xlabel= ' Menv', xscale='log', ylabel= ' Fv_env', yscale='log')
# # fig.supylabel( r'$\delta_M$', fontsize=12 )
# # [ fig.savefig( ptitle + fig_ext , bbox_inches='tight') for fig_ext in ('.png', '.pdf') ]
# plt.show()

# fig, ax = plt.subplots( )
# fig.suptitle( ptitle )
# ax.scatter(  rdf.M_sim , rdf.Fv_env, alpha=0.7 )
# [ax.text( s=rdf.index[i], x=rdf.M_sim[i],  y=rdf.Fv_env[i], horizontalalignment='left', verticalalignment='top', fontsize=5 ) for i in range(len(rdf)) ]
# ax.set(  xlabel= ' Menv', xscale='log', ylabel= ' Fv_env', yscale='log')
# # fig.supylabel( r'$\delta_M$', fontsize=12 )
# # [ fig.savefig( ptitle + fig_ext , bbox_inches='tight') for fig_ext in ('.png', '.pdf') ]
# plt.show()

# M_envsim = rdf.Mcyl - rdf.M_sim		# non grandché

# fig, ax = plt.subplots( )
# fig.suptitle( ptitle )
# ax.scatter(  M_envsim , rdf.Fv_env, alpha=0.7 )
# [ax.text( s=rdf.index[i], x=M_envsim[i],  y=rdf.Fv_env[i], horizontalalignment='left', verticalalignment='top', fontsize=5 ) for i in range(len(rdf)) ]
# ax.set(  xlabel= ' Menv', xscale='log', ylabel= ' Fv_env', yscale='log')
# # fig.supylabel( r'$\delta_M$', fontsize=12 )
# # [ fig.savefig( ptitle + fig_ext , bbox_inches='tight') for fig_ext in ('.png', '.pdf') ]
# plt.show()

# fig, ax = plt.subplots( )
# fig.suptitle( ptitle )
# ax.scatter(  M_envsim , rdf.Meo, alpha=0.7 )
# [ax.text( s=rdf.index[i], x=M_envsim[i],  y=rdf.Meo[i], horizontalalignment='left', verticalalignment='top', fontsize=5 ) for i in range(len(rdf)) ]
# ax.set(  xlabel= ' Menv', xscale='log', ylabel= 'M_env_obs', yscale='log')
# # fig.supylabel( r'$\delta_M$', fontsize=12 )
# # [ fig.savefig( ptitle + fig_ext , bbox_inches='tight') for fig_ext in ('.png', '.pdf') ]
# plt.show()