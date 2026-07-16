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
from astropy import constants as const
# import emcee, corner
from scipy.optimize import curve_fit
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops, regionprops_table
from skimage.morphology import closing, footprints
from local_variables import *
from visibfit_functions import crop_image
plt.rcParams.update({ 'font.size':9, 'legend.fontsize':8, 'errorbar.capsize':2, 'scatter.edgecolors':'None', 'figure.dpi':200})

tungslist = np.array([17, 20, 30, 42, 50, 52, 53, 57, 65, 67, 70, 78, 79]) 		# disk numbers fitted in Tung+24, wb 43 ?? not shown
OKlist =    np.array([17, 20, 30, 42, 50, 53, 57, 65, 67, 72, 79, 82, 83, 43]) 	# (70, 78, 29, 52 no bc binary, , 63 75 no bc no info in truths)
prettylist =np.array([17, 20, 30, 42, 50, 53, 57, 65, 67, 72, 79, 83])			# 43 misterious and thick, 82 bahh
# pixscale = 9.92063492063492e-6      	# deg
# sr_to_pix = np.deg2rad( pixscale )**2   # convert Jy/sr to Jy/pix
dist = 140 *u.pc  # parsec	[1pc = 206265 au]
au_to_rad = 1 / dist.to_value(u.au)
au_to_as = 1 / dist.to_value(u.au) * 180 / np.pi * 3600		# from au to arcsec
# Rmax = 8	# [arcsec]		# issue is the min baselines not the Rmax really
uvd_min = 3e4 # np.deg2rad( Rmax / 3600) / 1.22	# [lambda units], for the uvcut


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


def temp_profile_Tung( lum, r, T0=71):
	'''Average T(r) profile from Tung24 fit. lum=Lint+Lacc in [Lsun] and r in [au]. '''
	return lum**0.25 * ( r / 35 )**(-0.52) * T0   # Kelvin
	#return lum**0.25 * ( r / 50 )**(-0.22) * 36   # Kelvin	(Max's fit formula)


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
	L80_dict = {'4':369.2, '5':623.8, '6':1172.5, '7':1673.1, '8':3527.3 , '9':6482.6}	# 80 percentile baselines lenght [m]
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
	ax.plot( opac_df.lam *1e-3, opac_df.ksca, label='K_sca', alpha=0.2)
	ax.plot( opac_df.lam *1e-3, k_v_15, label=r'k($\beta=1.5$)', c='b', ls='-.')
	ax.plot( opac_df.lam *1e-3, k_v_1, label=r'k($\beta=1.0$)', c='r')
	ax.vlines( x=[0.89, 3, 7], ymin=1e-2, ymax=1e5, colors='gray', alpha=0.6, linestyles=':', linewidths=1)
	ax.set( xscale='log', yscale='log', xlabel='$\lambda$ [mm]', ylabel='$\kappa$ [cm2 / g]', xlim=[1e-4, 20], ylim=[1e-2, 1e5])
	ax.legend( loc='lower left')
	# ax.grid( True, axis='both', alpha=0.5, linestyle=':')
	plt.show()


def skymodel_ratio( diskname , folder_wles=[3000,7000], margin=None, figure=True):
	'''
	Calculate for given diskname the spectral index alpha of the original RT skymodels at 3 and 7mm and plot the image.
	'''
	from astropy.io import fits

	for wle_um in folder_wles:
		fname = data_prefix + f'{round(wle_um/1e3)}mm/' + diskname + f'_{wle_um}um.fits'
		hdul = fits.open( fname )
		hdul.info()
		pixscale = hdul[0].header['CDELT1']		# deg / pix
		au_to_pix = au_to_as / (3600*pixscale)	# pix / au

		main_beam = np.rad2deg( 1.13 * min(folder_wles) / 1e6 / 12	)		# [deg]	,  needs to be the same for both frames
		pixcut = margin if type(margin) is not type(None) else int( main_beam / pixscale / 2 )							# margin in pixel
		skycut = crop_image( hdul[0].data, margins=[ pixcut, pixcut])
		if wle_um == 3000:
			skycut_3 = skycut
		else: skycut_7 = skycut
	
	skycut_ratio = - np.log10( skycut_3 / skycut_7) / np.log10( 3 / 7 )
	if figure:
		plt.figure( figsize=(4.5,4.5))
		plt.imshow(  skycut_ratio[:, ::-1 ], origin='lower', cmap='inferno_r')	# norm=mpl.colors.LogNorm( vmin=None, vmax=None)
		plt.colorbar()
		# plt.contour( skycut_3[:, ::-1 ], levels=5, origin=None)
		plt.contour( skycut_ratio[:, ::-1 ], levels=[2], colors='w', origin=None)
		plt.axis( 'off' )
		# plt.savefig( 'sky_model' + fig_ext, bbox_inches='tight', dpi=300)
		plt.show()
		plt.close()
	print( '\nMedian of skymodel ratio: ', np.median( skycut_ratio) )
	return skycut_ratio, au_to_pix


def skymodel_ratiosplot( diskname, res_df, projections=['xy', 'yz'], margin=25, fs=(5.2,2.4)):
	'''
	Extra figure for visualising the ratio on the skymodels of a disc seen in two projections. With a freaking well-behaving colorbar, jeez.
	'''
	from mpl_toolkits.axes_grid1 import AxesGrid
	ptitle = 'Sky model spectral index'
	fig = plt.figure( figsize=fs)
	fig.suptitle( ptitle)
	grid = AxesGrid( fig, 111, nrows_ncols=(1, 2), axes_pad=0.02, share_all=True,
		cbar_location="right", cbar_mode="single", cbar_size="7%", cbar_pad="2%", )

	for ax, p in zip(grid, projections):
		try:
			img, pix_au = skymodel_ratio( diskname=diskname[:-2] + p, margin=margin, figure=False)
			diskID = diskname.strip('disk').strip('xyz') + p
			R_sim = res_df.loc[ diskID ]['R_sim']	# [au]
			i_sim = res_df.loc[ diskID ]['i_sim']	# [deg]
			PA_disc = res_df.loc[ diskID ]['PA']	# [deg]
		except:
			print( p, 'projection not found!')
			continue
		im = ax.imshow( img, origin='lower', cmap='turbo_r', vmin=1.5)
		ax.annotate( p, (0.5, 0.9), xycoords='axes fraction', ha='center', color='white', alpha=0.8)
		width = R_sim * pix_au ; height = width * np.cos( np.deg2rad(i_sim) )
		R_patch = mpl.patches.Ellipse( (margin, margin), width=width, height=height, angle= 90 - PA_disc  , #transform=ax.transAxes
					fill=False, edgecolor='w', linewidth=1, ls=':', alpha=1 )
		ax.add_patch( R_patch )
		xc, yc = margin, margin
		rulersize = 50 	# [au]
		rls_pix = rulersize	* pix_au
		ax.plot( [xc - 0.5*rls_pix, xc + 0.5*rls_pix], (yc - 0.9*margin )*np.array([1,1]), c='w', lw=1.5, alpha=.9)		# ruler patch
		ax.text( xc-0.8*rls_pix , yc-0.9*margin, s=f'{rulersize :3.0f} au', color='w', ha='right', va='center', alpha=.8, fontsize='x-small') 
		ax.text( xc+0.8*rls_pix , yc-0.9*margin, s=f'{rulersize/140 :0.2f}"', color='w', ha='left', va='center', alpha=.8, fontsize='x-small')
		ax.axis('off')

	grid.cbar_axes[0].colorbar( im, label=r'$\alpha_\mathrm{(3-7)mm}$')
	[ fig.savefig( savedir_prefix + ptitle.replace(' ', '_') + '-' + diskname[:-3] + fig_ext , bbox_inches='tight') for fig_ext in ('.png', '.pdf') ]
	plt.show()


def compare_distros( df, var='i_sim', condition="M_obs / M_sim *100 < 1", bins='doane', sampcol='mediumorchid', wle=3):
	def darken_edge( col):
		return tuple( np.clip( np.array( mpl.colors.to_rgb( col) ) * 0.8, 0, 1))		# compute a slightly darker edge color automatically

	# condition = "M_obs / M_sim *100 < 1"		# "df.M_obs/df.M_sim*100 < 1"
	# df.M_sim = rdf.M_sim /100		# no puedo perché modifica l'originale
	filt = df.query( condition)[var]
	sample = df[var]
	style = { 'linewidth': 1.5, 'zorder':2}

	ptitle = f'{var} [{condition}] {wle}mm'
	fig, ax = plt.subplots( figsize=(3,3.3), sharex=True, layout='tight')
	axins = ax.inset_axes( [0,-0.3 , 1, 0.26] )
	axs = [ax, axins]
	fig.subplots_adjust( bottom=0.3 )		# makes space for the inset in the figure area
	hs = axs[0].hist( sample, bins=bins, color=sampcol, **style, edgecolor=darken_edge(sampcol), alpha=0.92, label='MHD sample')
	hf = axs[0].hist( filt, bins=hs[1], color='grey', **style, edgecolor=darken_edge('grey'), alpha=0.9, label= condition)
	axs[1].stairs(  hf[0]/hs[0], edges=hs[1], color=sampcol, **style, baseline=None, alpha=0.9, label='cond/sample')
	axs[0].set( xlabel=var, ylabel='count')
	axs[1].set( xlabel=var, ylabel='ratio')
	axs[0].legend()
	fig.canvas.manager.set_window_title( ptitle.replace(' ','_') )
	#axs[1].legend()
	plt.show()


def plot_sample_props( results_dir, run_name, analistdir, bins='doane'):

	df = pd.read_csv( results_dir + f'analysis_results{run_name.replace(" ","_")}-{analistdir}'[:-1] + '.txt', sep='\t', index_col='source')
	vars = ['M_sim', 'R_sim', 'i_sim']
	units = ['M$_\odot$', 'au', '°']
	colors = ['tab:red', 'tab:green', 'C1']

	for i, v in enumerate( vars):
		if i != 2:
			values = np.log10(df[v].values)
			xlab = f'Log({v}) [{units[i]}]'
		else: 
			values = df[v].values
			xlab = v +  f' [{units[i]}]'
		mean = np.nanmean( values )
		base_rgb = np.array(mpl.colors.to_rgb(colors[i]))
		darken_factor = 0.8			# compute a slightly darker edge color automatically
		edge_rgb = tuple(np.clip(base_rgb * darken_factor, 0, 1))
		style = {'edgecolor': edge_rgb, 'linewidth': 1.5, 'zorder':2}

		ptitle =  f'MHD {v} distribution'
		fig, ax = plt.subplots( figsize =(3.5,3.5), tight_layout=True )
		# fig.suptitle( ptitle )
		hh = ax.hist( x=values, bins=bins, color=colors[i], histtype='bar', **style , alpha=0.85) #, label=f'ratio, $\sigma$={np.nanstd( ratio ) :.2f}')
		ax.axvline( x=mean, ls='-.', lw=2, c=edge_rgb, label=f'mean = {mean :.2f}', alpha=0.9 )	
		ax.set( xlabel= xlab, ylabel='counts', title=ptitle )	# , xlim=xlims
		ax.legend()
		figname = results_dir + f'Figures_{fig_ext.strip(".")}/' + analistdir + ptitle.replace(' ', '_')  + fig_ext 
		[ fig.savefig( figname, bbox_inches='tight') for fig_ext in ('.png', '.pdf') ]
		# plt.show()
		plt.close()
	print('simulated sample properties saved to plot!\n')


def sim_temp_plot( cmap='turbo', au_margin=110, rulersize=50, contlevels=[], discN='20'):
	'''
	Plot the temperature map for the disc provided by Ugo, mass averaged, integrated along a line of sight crossing the source edge-on.
	rulersize and margin in [au]
	'''
	fname = f'/Users/gcolumba/PostDoc_Mac/PostProc/simulations/temperature_sink{discN}.pkl'
	t = np.load( fname, allow_pickle=True, encoding='bytes')	# dictionary of log(T)
	Timg = 10**t['temperature']		# linear T array
	print('max T:', Timg.max(), 'min T:', Timg.min() )
	
	pixscale = 500 / 1024			# [au / pix]
	pixcut = int( au_margin / pixscale )			# margin in pixel
	Tcut = crop_image( Timg, margins=[ pixcut, pixcut])[:, ::-1 ]		# [K] 
	xc, yc = np.array( Tcut.shape ) / 2

	ptitle = 'Temperature map source n.' + discN	# mass-averaged, edge-on
	fig, ax = plt.subplots( figsize=(4,4), layout='constrained')
	cb = ax.imshow( Tcut, origin='lower', cmap=cmap, norm=mpl.colors.LogNorm(), aspect='equal', interpolation=None ) 
	if contlevels!=[]: ax.contour( Tcut, levels=contlevels )
	rls_pix = rulersize / pixscale 	# [au / (au/pix) = pix]
	ax.plot( [xc - 0.5*rls_pix, xc + 0.5*rls_pix], (yc - 0.9*pixcut )*np.array([1,1]), c='w', lw=2, alpha=.9)		# ruler patch
	ax.text( xc-0.8*rls_pix , yc-0.9*pixcut, s=f'{rulersize :3.0f} au', color='w', ha='right', va='center', alpha=.8, fontsize=8) 
	ax.text( xc+0.8*rls_pix , yc-0.9*pixcut, s=f'{rulersize/140 :0.2f}"', color='w', ha='left', va='center', alpha=.8, fontsize=8)
	
	ax.set( title=ptitle, )
	ax.axis( 'off' )
	fig.colorbar( cb, cax= ax.inset_axes( [1,0 , 0.07, 1] ), ax=ax, label=r'$T_\mathrm{avg}$ [K]' ) 	# shrink=0.8, pad=0.00,
	[ fig.savefig( savedir_prefix + ptitle.replace(' ', '_') + fig_ext , bbox_inches='tight', dpi=300) for fig_ext in ('.png', '.pdf') ]
	plt.show()


def scatter_with_errors( ax, x, y, x_lo=None, x_up=None, y_lo=None, y_up=None,
						fmt='o', facecolor='C0', edge_darken=0.9, ecolor=None, # capsize=3
						marker_alpha=0.8, err_alpha=0.25, label=None, **kwargs):
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
	plotline, caplines, barlines = ax.errorbar( x, y, xerr=xerr, yerr=yerr, fmt=fmt, markerfacecolor=facecolor, markeredgecolor='None',
					ecolor=ecolor, elinewidth=1, alpha=marker_alpha, label=label, **kwargs)

	[bar.set_alpha(err_alpha) for bar in barlines]
	[cap.set_alpha( err_alpha) for cap in caplines]
	return # markerline


def ratio_histogram( var1, var2, run_name, analistdir, histcolor='tab:green', bins='doane', xlims=None, y_max=0, Tlab='', logratio=False, figsz=3.3, Ax=None):
	'''
	Plot a histogram of the ratio between var1/var2 and write the mean and std of the distribution.
	'''
	ratio = var1 / var2 	# generally obs/sim
	if logratio: 
		ratio = np.log10( ratio )
		loglab = 'Log '
		ident_x = 0
	else:
		loglab = '' ; ident_x = 1
	mean_r = np.nanmean( ratio)
	print( f'min and max {var1.name[0]} ratios: ', ratio.min(), ratio.max() )
	outliers = 0
	if xlims is not None:
		binflag = '-fix' 
		if (ratio.min() < xlims[0]) | (ratio.max() > xlims[1]):
			print( f'There are some values outside the given hist xlims!')
			outliers = len( ratio[ ratio < xlims[0]] ) + len( ratio[ ratio > xlims[1]] ) 
	else: binflag = f'-{bins}'
	base_rgb = np.array(mpl.colors.to_rgb(histcolor))
	darken_factor = 0.8			# compute a slightly darker edge color automatically
	edge_rgb = tuple(np.clip(base_rgb * darken_factor, 0, 1))
	style = {'edgecolor': edge_rgb, 'linewidth': 1.5, 'zorder':2}

	if Ax == None:
		ptitle =  loglab + f'{var1.name}_{var2.name} ratio' + run_name + Tlab
		shortle = run_name[1:4] + f' {var1.name[0]} ratio' 
		fig, ax = plt.subplots( figsize=(figsz, figsz), tight_layout=True )
	else: ax = Ax
	q16, median_r, q84 = np.nanquantile( ratio, [0.16, 0.5, 0.84])
	hh = ax.hist( x=ratio, bins=bins, range=xlims, color=histcolor, histtype='bar', **style , alpha=0.85) #, label=f'ratio, $\sigma$={np.nanstd( ratio ) :.2f}')
	ax.axvline( x=ident_x, ls='--', lw=2.5, c='k', alpha=0.99)
	ax.axvline( x=median_r, ls='-.', lw=2.5, c=edge_rgb, label=f'median = {median_r :.2f}', alpha=0.9 )	
	# ax.axvline( x=mean_r, ls=':', lw=1.5, c=edge_rgb, label=f'mean = {mean_r :.2f}', alpha=0.7 )
	hymax = max( hh[0].max()*1.1, y_max)
	ax.fill_between(x=[q16, q84] , y1=[0,0], y2= hh[0].max() + 100, step='mid', facecolor='gray', zorder=1, alpha=0.19,	
				label='(16-84)%' + '\n' + f'$\Delta/2={ (q84 - q16)/2 :.2f}$' ) 
	ax.set( xlabel= loglab + f'{var1.name} / {var2.name}', ylabel='counts', ylim=[0, hymax] )	# , xlim=xlims
	ax.legend()
	ax.text( x=0.88, y=0.5, s=run_name[1:4], ha='center', va='center', transform=ax.transAxes, 
		 color='k', fontweight='bold', bbox=dict(boxstyle='round', fc="w", ec="k"))
	ax.text( x=0.95, y=0.6, s=Tlab,	ha='right', va='center', transform=ax.transAxes, color='gray', alpha=1.)
	if outliers > 0: ax.text( x=0.97, y=0.4, s=f'+{outliers} outlier(s)', ha='right', va='center', transform=ax.transAxes, color=edge_rgb, size='x-small', style='italic', alpha=.8)
	
	if Ax==None:
		ax.set( title=shortle)
		[ fig.savefig( f'Figures_{fig_ext.strip(".")}/' + analistdir + ptitle.replace(' ', '_') + binflag + fig_ext , bbox_inches='tight') for fig_ext in ('.png', '.pdf') ]
		plt.close()
	else: return hymax


def plot_Fv_compare( df, v_obs, k_sim, rdata='sim', Tavg=122, run_name='', analistdir='OKlist/', errors=True):

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
	[ fig.savefig( f'Figures_{fig_ext.strip(".")}/' + analistdir + ptitle.replace(' ', '_') + f'_k{k_sim :.3f}_T{Tavg :1.0f}K' + fig_ext , bbox_inches='tight') for fig_ext in ('.png', '.pdf') ]
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
	

def plot_mass_env( df, run_name, analistdir ):
	'''
	Compared retrieved mass from obs to simul mass of disks with either simple approx or with annular computation. 
	'''
	M_obs = df.Meo
	M_sim = df.Mes/100 		# (df.Mcyl - df.M_sim) /100	
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
	[ fig.savefig( f'Figures_{fig_ext.strip(".")}/'+ analistdir + ptitle.replace(' ', '_') + fig_ext , bbox_inches='tight') for fig_ext in ('.png', '.pdf') ]
	# plt.show()
	plt.close()

	ptitle = 'Envelope flux'
	fig, ax = plt.subplots( )
	fig.suptitle( ptitle )
	ax.scatter(  x=df.Mcyl , y=df.Fv_env, alpha=0.7 )
	[ax.text( s=df.index[i], x=df.Mcyl[i],  y=df.Fv_env[i], horizontalalignment='left', verticalalignment='bottom', fontsize=5 ) for i in range(len(df)) ]
	ax.set(  xlabel= 'Mcyl', xscale='log', ylabel= 'F_env', yscale='log')
	# fig.supylabel( r'$\delta_M$', fontsize=12 )
	[ fig.savefig( f'Figures_{fig_ext.strip(".")}/'+ analistdir + ptitle + fig_ext , bbox_inches='tight') for fig_ext in ('.png', '.pdf') ]
	plt.close()


def plot_mass_compare( df, run_name, Tavg, simple_M, analistdir, errors=True, logratio=True, Ax=None, fs=(3,3.1)):
	'''
	Compared retrieved mass from obs to simul mass of disks with either simple approx or with annular computation. 
	'''
	if simple_M:
		M_obs, M_obs_lo, M_obs_up = df.M_obs_simple, df.M_obs_simple_lo, df.M_obs_simple_up
	else: 
		M_obs, M_obs_lo, M_obs_up = df.M_obs, df.M_obs_lo, df.M_obs_up
	M_ratio = ( M_obs / (df.M_sim /100))**1 	# accuracy_ratio( M_obs, df.M_sim/100 )
	qs = np.nanquantile( M_ratio, [0.16, 0.5, 0.84] )
	T_label = f'T={Tavg :1.0f}K' if simple_M else 'T=T(r)'

	ptitle = 'Disk mass comparison' + run_name
	shortle = 'Disc mass'  # run_name[1:4] + 
	if Ax == None:
		fig, ax = plt.subplots( figsize=fs, tight_layout=True)
	else : 
		ax = Ax
	ax.axline( xy1=(0.0001, 0.0001), slope=1, ls='--', c='gray', zorder=1 )			# y=x identity
	if errors: 
		scatter_with_errors( ax=ax, x= df.M_sim/100, y=M_obs, y_lo=M_obs_lo, y_up=M_obs_up, fmt='o', facecolor='tab:red', marker_alpha=0.7, markersize=6 )
	else:	ax.scatter( x=df.M_sim/100, y=M_obs, marker='o', c='tab:red', alpha=0.7, s=36)		# observed fluxes
	# [ax.text( s=df.index[i], x=df.M_sim[i]/100,  y=df.M_obs[i], ha='left', va='bottom', fontsize=5 ) for i in range(len(df)) ]		# source IDs
	ax.text( x=0.01, y=0.86, s= T_label,	# f'T={Tavg :1.0f} K' if simple_M else 'T=T(r)'
		ha='left', va='center', transform=ax.transAxes, color='gray', alpha=1.)
	ax.text( x=0.01, y=0.76, s= f'16%-84% accuracy: \n{qs[0] :1.1f}x - {qs[2] :1.1f}x',
		ha='left', va='center', transform=ax.transAxes, fontsize='small', color='gray', alpha=0.8)
	ax.text( x=0.5, y=0.93, s=run_name[1:4], ha='center', va='center', transform=ax.transAxes, 
		 color='k', fontweight='bold', bbox=dict(boxstyle='round', fc="w", ec="k"))
	axlims = [ 2e-5, 3e-2]
	ax.set( xlabel= r'$ M_\mathrm{sim} $ [M$_{\odot}$]', ylabel=r'$ M_\mathrm{obs} $ [M$_{\odot}$]', xlim=axlims, ylim=axlims, aspect='equal' )

	if Ax==None: 
		ax.set( xscale='log', yscale='log') ; ax.set_title( shortle, fontsize='medium')
		#plt.show()
		[ fig.savefig( f'Figures_{fig_ext.strip(".")}/' + analistdir + (ptitle +'_'+ T_label[2:]).replace(' ', '_') + fig_ext , bbox_inches='tight') for fig_ext in ('.png', '.pdf') ]
		plt.close()
		if not simple_M: 
			histlims = [-2.11, 1.] if logratio else [0, 2.9]		# for fixed-x comparison
			ratio_histogram( df.M_obs, df.M_sim/100, run_name, analistdir, histcolor='tab:red', bins=8, xlims=histlims, logratio=logratio)	# fixed
			ratio_histogram( df.M_obs, df.M_sim/100, run_name, analistdir, histcolor='tab:red', bins='doane', logratio=logratio)
			ratio_histogram( df.M_sim/100, df.M_obs, run_name, analistdir, histcolor='tab:red', bins='auto', logratio=logratio)		# swapped ratio
	else:
		return None


def plot_radius_compare( df, res_limit, run_name, analistdir, errors=True, r95=True, Ax=None, fs=(3,3.1)):
	'''
	Assuming R_obs is R_95, in [au] if r95=True, else R_90. 
	'''
	if r95:
		R_95 = df.R_obs
		R_90 = df.R_obs / 1.1408
		not_R_obs = R_90
		lab_err = '$R_{95\%}$' ; lab_sca = '$R_{90\%}$'
	else: 
		R_95 = df.R_obs * 1.1408
		R_90 = df.R_obs 
		not_R_obs = R_95
		lab_err = '$R_{90\%}$' ; lab_sca = '$R_{95\%}$'
	r_ratio_90 = R_90 / df.R_sim 			# accuracy_ratio( df.R_obs, df.R_sim )
	r_ratio_95 = R_95 / df.R_sim 
	R_reslim = res_limit * 2.1436 / np.sqrt(8 * np.log(2))		# resolution limit in terms of R_90 radii, to compare apples with apples

	ptitle = 'Radius comparison' + run_name
	shortle = 'Disc radius' 
	if Ax == None:
		fig, ax = plt.subplots( figsize=fs, tight_layout=True)
	else : 
		ax = Ax
	ax.fill_between( [0.01, R_reslim, 10], y1=[10, 10, R_reslim], y2=0.01, step='pre', facecolor='gray', alpha=0.16, label=r'$\theta_\mathrm{res}$' )
	ax.axline( xy1=(0.5, 0.5), slope=1, ls='--', c='gray', alpha=0.8 )		# y=x identity
	#ax.scatter( x=df.R_sim *au_to_as, y=df.R_obs/1.42 *au_to_as, marker='o', c='r', label='$R_{68\%}$', alpha=0.2)
	if errors:
		scatter_with_errors( ax=ax, x=df.R_sim *au_to_as, y=df.R_obs*au_to_as, y_lo=df.R_obs_lo*au_to_as, y_up=df.R_obs_up*au_to_as,
					fmt='o', facecolor='g', label=lab_err, marker_alpha=0.6 )
	else:
		ax.scatter( x=df.R_sim *au_to_as, y=df.R_obs *1 *au_to_as, marker='o', c='g', label=lab_err, alpha=0.7, zorder=3.7)		# observed radii
	# [ ax.text( s=df.index[i], x=df.R_sim[i]*au_to_as,  y=df.R_obs[i]*au_to_as, ha='left', va='bottom', fontsize=5 ) for i in range(len(df)) ]		# source IDs
	ax.scatter( x=df.R_sim *au_to_as, y=not_R_obs *au_to_as, marker='o', c='b', label=lab_sca, alpha=0.2)
	ax.text( x=0.01, y=0.76, s=(f'accuracy:\n $R_{{90\%}}$: {np.nanmedian( r_ratio_90) :1.2f}x'  #\nmedian accuracy $R_{{90\%}}$: {np.mean( r_ratio_90) :1.1f}x'
		f'\n $R_{{95\%}}$: {np.nanmedian( r_ratio_95) :1.2f}x'), ha='left', va='center', transform=ax.transAxes, color='k', fontsize='small', alpha=0.8)
	ax.set( xlabel= r'$ R_\mathrm{sim} $ [arcsec]', ylabel=r'$ R_\mathrm{obs} $ [arcsec]' ,# xscale='log', yscale='log',
		xlim=[0.05,2.1], ylim=[0.05, 2.1], aspect='equal' )
	ax.text( x=0.5, y=0.93, s=run_name[1:4], ha='center', va='center', transform=ax.transAxes, 
		 color='k', fontweight='bold', bbox=dict(boxstyle='round', fc="w", ec="k"))

	if Ax==None: 
		ax.set( title=shortle, xscale='log', yscale='log')
		ax.legend( loc='lower right')
		[ fig.savefig( f'Figures_{fig_ext.strip(".")}/' + analistdir + ptitle.replace(' ', '_') + fig_ext , bbox_inches='tight') for fig_ext in ('.png', '.pdf') ]
		plt.close()

		ratio_histogram( df.R_obs, df.R_sim, run_name, analistdir, histcolor='tab:green', bins=9, xlims=[0.2, 2.5])
		ratio_histogram( df.R_obs, df.R_sim, run_name, analistdir, histcolor='tab:green', bins='doane')
	else:
		return None


def plot_inc_compare( df, run_name, analistdir, Ax=None, fs=(3,3.1)):
	'''
	Assuming inc in [deg]. 
	'''
	qs = np.nanquantile( df.i_obs / df.i_sim, [0.16, 0.5, 0.84] )
	if Ax == None:
		shortle = 'Disc inclination' 
		ptitle = 'Inclination comparison' + run_name
		fig, ax = plt.subplots( figsize=fs, tight_layout=True)
	else:
		ax = Ax
	ax.axline( xy1=(1, 1), slope=1, ls='--', c='gray' )		# y=x identity
	scatter_with_errors( ax=ax, x=df.i_sim, y=df.i_obs, y_lo=df.i_obs_lo, y_up=df.i_obs_up, fmt='o', facecolor='C1', marker_alpha=.75, err_alpha=0.27 )
	# ax.scatter( x=inc, y=df.i_obs, marker='o', c='orange', alpha=0.8)
	ax.text( x=0.01, y=0.8, s= f'16%-84% accuracy: \n{qs[0] :1.1f}x - {qs[2] :1.1f}x',
		ha='left', va='center', transform=ax.transAxes, color='gray', fontsize='small', alpha=0.8)
	ax.text( x=0.5, y=0.93, s=run_name[1:4], ha='center', va='center', transform=ax.transAxes, 
		 color='k', fontweight='bold', bbox=dict(boxstyle='round', fc="w", ec="k"))
	ax.set( xlabel= r'$ i_\mathrm{sim} $ [deg]', ylabel=r'$ i_\mathrm{obs} $ [deg]', aspect='equal', xlim=[-2,94], ylim=[-2,94] )
	
	if Ax == None:
		ax.set( title=shortle)
		[fig.savefig( f'Figures_{fig_ext.strip(".")}/' + analistdir + ptitle.replace(' ', '_') + fig_ext , bbox_inches='tight') for fig_ext in ('.png', '.pdf')]
		plt.close()
	else:
		return None


def column_plotter( analistdir, conf_flag, model_comps, xsrc_flag, Texps=[300,3600,10800], logratio=False, y_max=16):
	'''
	Create vertical triple plots of the comparison between Obs/sim quantities [radii, mass, inc].
	'''
	plt.rcParams.update({ 'font.size':9, 'legend.fontsize':8, 'lines.markersize':5.5, 'figure.dpi':300})
	wles = [1, 3, 7]	# [mm]
	configs = ['4', '7', '8']
	varnames = ['radius', 'mass', 'inclination']

	for var in varnames:
		ptitle = 'Disc ' + var 
		fig, axs = plt.subplots( 3,1, figsize=(2.8,8.4), sharex=True, sharey=True )		# for scatter plots
		ptitle_r = var.title() + ' ratio' 
		fig_r, axs_r = plt.subplots( 3,1, figsize=(2.8,8.4), sharex=True, sharey=True )		# for ratio histograms

		for i in range(len( wles)):
			folder_wle = f'{round( wles[i])}mm/'
			savedir = savedir_prefix + folder_wle + f'run_{Texps[i]}s_{model_comps}_{xsrc_flag}_{conf_flag}/' 		# results directory name
			run_name = f'-{ wles[i]}mm {Texps[i]}s {model_comps} {conf_flag}'
			df = pd.read_csv( savedir + f'analysis_results{run_name.replace(" ","_")}-{analistdir}'[:-1] + '.txt', sep='\t', index_col='source')
			if var == varnames[0]:
				res = alma_resolution( wle=wles[i]/1000, config_name=configs[i])
				histlims = [-1., 1.] if logratio else [0.2, 2.5]		# for fixed-x comparison
				plot_radius_compare( df, res, run_name, analistdir, errors=True, Ax=axs[i])
				ratio_histogram( df.R_obs, df.R_sim, run_name, analistdir, histcolor='tab:green', bins=9, xlims=histlims, y_max=y_max, logratio=logratio, Ax=axs_r[i])
			elif var == varnames[1]:
				histlims = [-1.5, 1.] if logratio else [0, 2.9]		# for fixed-x comparison
				plot_mass_compare( df, run_name, 122, False, analistdir, errors=True, logratio=logratio, Ax=axs[i])
				ratio_histogram( df.M_obs,  df.M_sim/100, run_name, analistdir, histcolor='tab:red', bins=10, xlims=histlims, y_max=y_max, logratio=logratio, Ax=axs_r[i])
			elif var == varnames[2]:
				histlims = [-1.5, 1.5] if logratio else [0, 5.5]
				plot_inc_compare( df, run_name, analistdir, Ax=axs[i])
				ratio_histogram( df.i_obs, df.i_sim, run_name, analistdir, histcolor='tab:orange', bins=9, xlims=histlims, y_max=y_max, logratio=logratio, Ax=axs_r[i])
			if i==0: 	
				axs[i].set( title=ptitle) ; axs_r[i].set( title=ptitle_r )
				if var == varnames[0]: axs[i].legend( loc='lower right')
		
		if var != 'inclination': axs[i].set( xscale='log', yscale='log')
		fig.subplots_adjust( hspace=0 ) ; fig_r.subplots_adjust( hspace=0 )
		[ fig.savefig( savedir_prefix + ptitle.replace(' ', '_')  + '-3plot' + f'_{conf_flag}' + fig_ext , bbox_inches='tight') for fig_ext in ('.png', '.pdf') ]
		[ fig_r.savefig( savedir_prefix + ptitle_r.replace(' ', '_')  + '-3plot' + f'_{conf_flag}' + fig_ext , bbox_inches='tight') for fig_ext in ('.png', '.pdf') ]
		print( ptitle, '3-plots saved')
		plt.close()
	

	#plt.show()


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
	import casatools as cto
	table = cto.table()
	table.open( results_dir + diskname + f'/{diskname}.{config_name}.noisy.image.pbcor' )		# noisy image, with pbcor the central target is slightly under corrected?
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


def plot_Mobs_multiwave( analistdir, conf_flag, model_comps, xsrc_flag, Texp_1mm=300, Texp_3mm=3600):
	'''
	Plot M_obs at 3000um (3mm) vs M_obs at 890um (1mm) from their respective result folders.
	'''
	wle_labels = ['1mm', '3mm']
	Texps      = [Texp_1mm, Texp_3mm]

	dfs = {}
	for wle, Texp in zip(wle_labels, Texps):
		savedir  = savedir_prefix + wle + '/' + f'run_{Texp}s_{model_comps}_{xsrc_flag}_{conf_flag}/'
		run_name = f'-{wle[0]}mm {Texp}s {model_comps} {conf_flag}'
		fpath    = savedir + f'analysis_results{run_name.replace(" ","_")}-{analistdir}'[:-1] + '.txt'
		dfs[wle] = pd.read_csv( fpath, sep='\t', index_col='source')

	# align on common sources
	common = dfs['1mm'].index.intersection( dfs['3mm'].index )
	m1 = dfs['1mm'].loc[common, 'M_obs']
	m3 = dfs['3mm'].loc[common, 'M_obs']

	fig, ax = plt.subplots(figsize=(4, 4), tight_layout=True)
	ax.axline( xy1=(1e-4, 1e-4), slope=1, ls='--', c='gray', alpha=0.7)
	ax.scatter( m1, m3, marker='o', alpha=0.75)
	ax.set( xlabel=r'$M_\mathrm{1mm}$  [M$_\odot$]', ylabel=r'$M_\mathrm{3mm}$  [M$_\odot$]',
		xscale='log', yscale='log', aspect='equal', xlim=[1e-5,2e-2], ylim=[1e-5,2e-2])
	plt.show()
	return fig, ax


def check_Robs_multiwave( analistdir, conf_flag, model_comps, xsrc_flag, Texps=[300,3600,10800]):
	'''
	Plot R_obs at 3000um (3mm) vs M_obs at 890um (1mm) from their respective result folders.
	'''
	wle_labels = ['1mm', '3mm', '7mm']

	dfs = {}
	for wle, Texp in zip(wle_labels, Texps):
		savedir  = savedir_prefix + wle + '/' + f'run_{Texp}s_{model_comps}_{xsrc_flag}_{conf_flag}/'
		run_name = f'-{wle[0]}mm {Texp}s {model_comps} {conf_flag}'
		res_path    = savedir + f'analysis_results{run_name.replace(" ","_")}-{analistdir}'[:-1] + '.txt'
		dfs[wle] = pd.read_csv( res_path, sep='\t', index_col='source')

	# align on common sources
	common = dfs['7mm'].index.intersection( dfs['1mm'].index.intersection( dfs['3mm'].index ))
	r1 = dfs['1mm'].loc[common, 'R_obs']
	r3 = dfs['3mm'].loc[common, 'R_obs']
	r7 = dfs['7mm'].loc[common, 'R_obs']
	rarr = np.array( [r1/r1, r3/r1, r7/r1]).T
	
	ptitle = 'R vs $\lambda$'
	fig, ax = plt.subplots(figsize=(5, 2.5), tight_layout=True)
	for rr in rarr:
		ax.plot( ['$R_\mathrm{1mm}$','$R_\mathrm{3mm}$', '$R_\mathrm{7mm}$'], rr, alpha=0.5)
	ax.set(  ylabel=r'$R_\mathrm{obs} / R_\mathrm{1mm}$ ', title=ptitle)
		#xscale='log', yscale='log', aspect='equal', xlim=[1e-5,2e-2], ylim=[1e-5,2e-2])
	plt.show()
	return fig, ax


def produce_truths_df():
	'''From Tungs data export a dataframe with the simulation truths of my interest. '''
	import h5py
	catalog = 'disk_01440_rmax_500_f_2_rho_3.8346e-15_thermal_False.h5'
	hf = h5py.File( catalog, 'r')
	disks = {}
	for k in hf.attrs.keys():	#Extract the disk quantities
		disks[k] = hf.attrs[k]

	Rsim = []; Rmean=[]; Msim = []; Mcyl =[]; Lint = []; Lacc = []; dTemp1 = []; dTemp2 = []; multip =[]; 
	angs_x = []; angs_y = []; angs_z = []; hr = []; Mstar = []; age =[]; Menv = [] ; M1000 = []
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
		Mcyl.append( disks[prefix + '_mass_cyl'] )			# mass inside the disk cylinder
		Menv.append( disks[prefix + '_mass_env_1000'] )		# env mass 
		#M1000.append( disks[prefix + '_mass_1000'] )		# M1000 = Menv + Mdisk
		Lint.append( disks[prefix + '_star_lum'] )
		Lacc.append( disks[prefix + '_star_acclum'] )
		dTemp1.append( disks[prefix + '_Temp_mid'] )		# mid, mavg o simple ?
		dTemp2.append( disks[prefix + '_Temp_mavg'] )
		multip.append( disks[prefix + '_multiplicity'])
		# hr.append( disks[prefix + '_hoverr'] )				# scale height?
		Mstar.append( disks[prefix + '_sink_mass'] )		# star mass
		age.append( disks[prefix + '_sink_age'] )

	dfT = pd.DataFrame( np.array([Msim, Mcyl, Menv, Rsim, Lint, Lacc, dTemp1, dTemp2, multip, Mstar, age, angs_x, angs_y, angs_z]).T, 
		columns=['M_disk', 'M_cyl', 'M_env', 'R_disk', 'L_int', 'L_acc', 'Tmid_disk', 'Tmavg_disk', 'multiplicity', 'M_star', 'age', 'i_yz', 'i_xz', 'i_xy'], index=ids)
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


def main_analysis( targetslist, wle, results_dir, config_name, run_name, analistdir, T_avg=122, r95=True, figures=True):
	'''
	Analyse the bestfit parameters of the whole sample and the derived quantities, comparing them to the simulation truths.   
	'''
	print( '\nPerforming MAIN analysis of the sample.')
	v_obs = 299792458.0/wle			# [Hz]			# 100 *1e9   obs frequency
	k_sim = 0.54 if round(wle*1e3)==3 else 0.138	# opTool original opacity for the simulation truths
	if round(wle*1e3)==1: k_sim = 3.5
	k_obs = k_sim # kappa_empir( v_obs, beta=1.5)	# 1.5 good for both 3mm and 7mm (not 0.9mm) # for the OBS # [cm2 / g]

	disklist = sorted( glob.glob( results_dir + 'disk*') )
	print( len(disklist), 'files found in ', results_dir)
	truths_df = pd.read_csv( truth_path, sep='\t', index_col=0 ) #.loc[OKlist]	# load my simulation truths file
	paramlist = []

	for fpath in disklist:
		diskID = fpath.strip( results_dir ).strip('disk')		# NN_xx kind
		disk_n = int(diskID.strip( '_yzx'))
		
		if disk_n in targetslist:
			try:
				# read the bestfit params from file for I0 and sma
				pars = np.loadtxt( fpath + 'bestfit_params.txt')	# NOW each par is an array([best, 16%, 84%]) !!
				LI0_d = pars[0]                   			# disk peak intensity 	Log[Jy/sr]
				LI0_env = pars[1]							# envelope peak intensity	Log[Jy/sr]
				Ri = np.deg2rad( pars[3] /3600)				# inner env radius	[arcsec --> rad]
				sma = np.deg2rad( pars[2] /3600)     		# gauss disk sigma	[arcsec --> rad]
				i_obs = pars[-4]							# disk inclination [deg]
				disc_PA = pars[-3][0]							# disk position angle [deg]
				p_idx = pars[5]	
				Rout = pars[4] * Ri

				R_68 = sma * np.sqrt( -2 * np.log(1-0.68))      # 68% radius  [rad]
				R_90 = R_68 * 1.4216                            # 90% radius
				R_95 = R_68 * 1.6215
				R_obs = R_95 if r95 else R_90    # unlike Tung who used 90%
				l_star = truths_df.loc[ disk_n ][['L_acc', 'L_int']].sum()			# L_acc + L_int [Lsun]

				F_v_simple = gauss_flux_integral( 10**LI0_d, sma, 1*R_obs)      # observed flux density of DISK [Jy]
				M_obs_simple = F_v_simple *1e-23 * ( dist.cgs.value )**2 / (k_obs * planck_bbody( v_obs, T_avg) )  / const.M_sun.cgs.value	# [Msun] 
				M_obs, F_v = mass_annuli_calc( v_obs, LI0_d, sma, R_obs, k_obs, l_star)				
				Fv_count = np.nan #count_flux_sources( 'disk'+ diskID, nRMS=7, config_name=config_name, results_dir=results_dir ) 
				F_v_thicc = thick_flux( v_obs, dist, R_obs[0], l_star=l_star)			# theoretical fully thick disk flux

				M_env_o, Fv_env = env_mass_annuli( v_obs, LI0_env, Ri, p_idx, Rout, k_obs, l_star)

			except: 
				print('No bestfit params found for disk', diskID)
				M_obs = R_obs = F_v = i_obs = Ri = p_idx = LI0_d = LI0_env = F_v_simple = M_obs_simple = M_env_o = Fv_env = Rout = [np.nan, np.nan, np.nan]
				F_v_thicc = Fv_count = disc_PA = np.nan
			
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
				M_star = truths_df.loc[ disk_n ]['M_star']		# [Msun]
				age = truths_df.loc[ disk_n ]['age'] / 1000			# [kyr]

				paramlist.append( [diskID, R_obs, R_sim, Ri, p_idx, M_obs_simple, M_obs, M_sim, epsilon, Menv_sim, Mcyl, disc_PA,
					LI0_d, LI0_env, F_v, F_v_simple, Fv_count, F_v_thicc, i_obs, i_sim, L_tot, F_sim_thin, M_env_o, Fv_env, Rout, M_star, age] )

	res_df = pd.DataFrame( paramlist, 
				columns=['source', 'R_obs', 'R_sim', 'Ri', 'p_idx', 'M_obs_simple', 'M_obs', 'M_sim', 'epsilon_M', 'Mes', 'Mcyl', 'PA',
				'LI0_d', 'LI0_env', 'F_obs', 'Fv_simple', 'Fv_count', 'F_thick', 'i_obs', 'i_sim', 'L_tot', 'Fsim_thin', 'Meo', 'Fv_env', 'Rout', 'M_star', 'age']
			).set_index('source')
	
	# some columns contain arrays of length 3 with uncertainties, but better to give each one an independent column of the DataFrame
	for col in ['R_obs', 'Ri', 'p_idx', 'M_obs_simple', 'M_obs', 'LI0_d', 'LI0_env', 'F_obs', 'Fv_simple', 'i_obs', 'Meo', 'Fv_env']:
		res_df[ [col, col+'_lo',col+'_up'] ] = pd.DataFrame( res_df[col].tolist(), index=res_df.index)
	
	res_df.to_csv( results_dir + f'analysis_results{run_name.replace(" ","_")}-{analistdir}'[:-1] + '.txt', sep='\t') #, float_format='%.2e')
	# res_df = pd.read_csv( f'analysis_results{run_name.replace(" ","_")}-{analistdir}'[:-1] + '.txt', sep='\t', index_col='source')	# to load it
	
	if figures:
		# plot_opacity()
		#plot_Fv_compare( res_df, v_obs, k_sim, Tavg=T_avg, rdata='sim', run_name=run_name, results_dir=results_dir, analistdir=analistdir )
		plot_inc_compare( res_df, run_name, analistdir )
		#plot_mass_compare( res_df, run_name, T_avg, True, analistdir )
		plot_mass_compare( res_df, run_name, T_avg, False, analistdir, errors=True, logratio=False)
		# plot_mass_env( res_df, run_name, analistdir )
		theta = alma_resolution( wle=wle, config_name=config_name)
		plot_radius_compare( res_df, theta, run_name, analistdir, r95=r95 )
		# thick_sim_inspo( truths_df, k_sim, T_avg, v_obs, run_name)
		plt.close() 
	return res_df



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
	# print( 'average beam radius [pix]: ', r_beam, '\nbeam sma ["]: ', a )
	peak_idx = np.unravel_index( np.argmax( crop_image(image, margins=[40,40]) ), shape=(81,81) )
	delta_centre = np.array(peak_idx ) - [40,40]	# offsets of the photocentre
	peak_centre = np.array( image.shape ) / 2 + np.roll( delta_centre, 1) 	# roll to put correct x and y offset in full image
	#if comp=='disc':
	flux_disc = np.nanmean( image[ circular_region( image, 2.5*r_beam, peak_centre) ] )
	#elif comp=='env':
	disc_env_img = np.clip( image, a_min=0, a_max=None).copy()				# avoid negative flux
	disc_env_img[ circular_region(disc_env_img, 2*r_beam) ] = np.nan		# nan on disc region
	flux_env = np.nanmean( disc_env_img[ circular_region( disc_env_img, 6*r_beam, peak_centre) ] )
	#else:
	flux_peak = np.nanmean( image[circular_region( image, r_beam, peak_centre)] )
	return flux_peak, flux_disc, flux_env


def plot_SNR( df, run_name, results_dir,simple=False ):
	ptitle = 'SNR ' + run_name[1:] 
	fig, ax = plt.subplots( figsize=(6,4), constrained_layout=True)
	if simple:
		col_list = [ 'SNR_simple', 'bkg noise (Jy/beam)'] 
		med_SNR = [np.nanmedian( df.SNR_simple)]
		c = ['tab:blue'] ; lab = [f'median SNR: ']
	else:
		col_list =[ 'SNR_disc', 'SNR_env', 'bkg noise (Jy/beam)']
		med_SNR = np.nanmedian( df[['SNR_disc', 'SNR_env']], axis=0)
		c = ['tab:blue', 'tab:orange'] ; lab = ['median SNR (<2.5 beam): ', 'median SNR (2-6 beam): ']
	dff = df[ col_list ].reset_index()
	dff.plot( xticks=dff.index, rot=90, logy=True, ax=ax, marker='o', legend=False)
	ax.set_xticklabels( df.index)
	# ax.axhline( y=[0.002], color='gray', ls=':')
	# ax.axhline( y=10, color='gray', ls=':')
	for i, m in enumerate( med_SNR):
		ax.axhline( y=m, color=c[i], ls='--', alpha=0.7, label=lab[i] + f'{m :4.0f}')
	ax.grid( True, axis='x', alpha=0.5, linestyle=':')
	ax.set( title=ptitle)
	ax.legend()
	figname = results_dir + f'Figures_{fig_ext.strip(".")}/' + ptitle.replace(' ', '_') + fig_ext
	[fig.savefig( figname, bbox_inches='tight') for fig_ext in ('.png', '.pdf') ]
	# plt.show()
	plt.close()


def assess_SNR( results_dir, config_name, run_name, simple=False):
	'''
	Evaluate the SNR of the cleaned image across the entire sample in results_dir. 
	'''
	import casatools as cto
	fitslist = sorted( glob.glob( results_dir + 'disk*') )
	print( len(fitslist), 'files found')
	SNRs = []
	truths_df = pd.read_csv( truth_path, sep='\t', index_col=0 ) #.loc[OKlist]	# load my simulation truths file

	for fname in fitslist:
		# diskID = diskname.strip('disk')		# NN_xx kind
		# disk_n = int( diskID.strip( '_yzx') )
		# R_sim = truths_df.loc[ disk_n ]['R_disk']
		diskname = os.path.basename( fname )	 	# each one in a separate folder
		img_tab = f'{results_dir}{diskname}/{diskname}.{config_name}.noisy.image'		# cleaned simanalyze image
		table = cto.table()
		table.open( img_tab )
		img = table.getcol('map').squeeze().copy() 
		peak = np.max( crop_image(img, margins=[35,35]) )	# find the peak flux in a region around the centre
		peak_beam, sign_disc, sign_env = peak_beam_avg( img, table=table)
		noise = min_bkg_rms( img )			# the minimum rms from bkg patches
		full_rms = rms( img )				# the rms of the entire image including target source
		snr_simple = peak_beam / noise
		snr_disc = sign_disc / noise
		snr_env = sign_env / noise
		
		SNRs.append( [diskname.strip( 'disk' ), snr_simple, snr_disc, snr_env, peak, peak_beam, noise, full_rms] )
		table.close()

	df = pd.DataFrame( SNRs, columns=['source', 'SNR_simple', 'SNR_disc', 'SNR_env', 'max peak', 'beam peak', 'bkg noise (Jy/beam)', 'RMS_full']).set_index('source')
	df.to_csv( results_dir+  f'SNR_dataframe_{run_name}.txt', sep='\t') #, float_format='%.2e')
	print( '\nMedian simple SNR of run: \t', np.nanmedian( df.SNR_simple) )
	plot_SNR( df, run_name, results_dir, simple=simple)


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


def fit_Mobs( results_dir, run_name, analistdir, logfit=True):
	'''
	Regress the Mobs relation with a simple curve fit. 
	'''
	df = pd.read_csv( results_dir + f'analysis_results{run_name.replace(" ","_")}-{analistdir}'[:-1] + '.txt', sep='\t')		# import the results dataframe

	df.drop(df[df['source'] == '29_xz'].index, inplace=True)
	#df.dropna( inplace=True )
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
	print( r'Empirical relation fitted:  \t $ M = a \cdot L_{bol}^\alpha \cdot R_{obs}^\beta \cdot F_{\nu}^\gamma$')
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
	[ fig.savefig( results_dir + f'Figures_{fig_ext.strip(".")}/' + ptitle.replace(' ', '_') + llab + fig_ext , bbox_inches='tight') for fig_ext in ('.png', '.pdf') ]
	plt.show()
	return popt


def klambda_to_au(x): return 1.22 * 1e-3/x * dist.to_value(u.au)		# add a physical ruler for size understanding
def au_to_klambda(x): return 1.22 * 1e-3/x * dist.to_value(u.au)		# 1e-3 I guess because the first xaxis is in klambda

	
def alpha_Menv_plot( model, quantity):

	data = np.loadtxt( 'alpha137_env.txt')

	ptitle = 'spectral index vs M_env' + f'-{model}-{quantity}'
	fig, ax = plt.subplots( figsize=(6,4), layout='constrained')
	ax.scatter( data[:,2], data[:,3], c='tab:blue', label=r'$\alpha(1-3)$', alpha= 0.7)
	ax.scatter( data[:,2], data[:,1], c='tab:orange', label=r'$\alpha(3-7)$')
	#ax.vlines( x=[0.89, 3, 7], ymin=1e-2, ymax=1e5, colors='gray', alpha=0.6, linestyles=':', linewidths=1)
	ax.set( xlabel='$ M_\mathrm{env} $ [M$_{\odot}$]', ylabel=r'$\alpha$', title=ptitle) #,  xscale='log', yscale='log', xlim=[1e-4, 20], ylim=[1e-2, 1e5])
	ax.legend()
	# ax.grid( True, axis='both', alpha=0.5, linestyle=':')
	# plt.show()
	[fig.savefig( ptitle.replace(' ', '_') + fig_ext, bbox_inches='tight', dpi=200) for fig_ext in ('.png', '.pdf') ]



def alpha_vs_Rdisc( alphalist, uvd, radii, savedir_prefix):
	'''
	Create a plot à la Garufi et al (2025), of the average alpha_disc vs measured disc size. 
	'''
	#radiil = 60 	# [au]
	#a_limit = np.broadcast_to( klambda_to_au( uvd), shape=(len(radii), len(uvd))) < np.array( [radii]).T		# condition: within each disc size
	uvd_au = klambda_to_au( uvd)
	a_obs13 = alphalist[0::4]
	a_obs37 = alphalist[1::4]

	ptitle = 'Spectral index vs size' # + f' {model} '#_{quantity}'
	fig, ax = plt.subplots( figsize=(4,3), layout='constrained')
	for i in range(len(radii)):
		a_limit = uvd_au < 2*radii[i]
		a_mean13 = np.nanmean( a_obs13[i][ a_limit ] )	# average on each disc 
		a_mean37 = np.nanmean( a_obs37[i][ a_limit ] )
		ax.scatter( radii[i], a_mean13, c='tab:blue', )#label=r'$\alpha_{(1-3)mm}$')
		ax.scatter( radii[i], a_mean37, c='tab:orange')#, label=r'$\alpha_{(3-7)mm}$')

	ax.set( xlabel='Disc radius [au]', ylabel=r'$\alpha$ index', ylim=[None, None], xscale='linear', title=ptitle)
	ax.legend()
	plt.show()
	#[fig.savefig( savedir_prefix + ptitle.replace(' ', '_') + quantity + fig_ext, bbox_inches='tight', dpi=200) for fig_ext in ('.png', '.pdf') ]


def median_alpha_plot( alphalist, uvd, model, quantity, savedir_prefix):
	'''
	Create a plot of the spectral indexes 1/3 and 3/7 from our mock observervations, as the median of the whole sample. 
	'''
	# print( alphalist)
	plt.rcParams.update({ 'font.size':9, 'legend.fontsize':7.5, 'figure.dpi':200})
	a_obs13 = np.nanquantile( alphalist[0::4], q=[.16, .5, .84], axis=0)
	a_obs37 = np.nanquantile( alphalist[1::4], q=[.16, .5, .84], axis=0)
	a_mth13 = np.nanmedian( alphalist[2::4], axis=0)
	a_mth37 = np.nanmedian( alphalist[3::4], axis=0)

	ptitle = 'Median spectral index' + f' {model} '#_{quantity}'
	fig, ax = plt.subplots( figsize=(4,3), layout='constrained')
	ax_top = ax.secondary_xaxis( 'top', functions=(klambda_to_au, au_to_klambda))
	ax_top.set_xlabel( 'physical scale [au]', color='grey', fontsize=8)
	ax_top.tick_params( axis="x", which='both', direction="in", colors='grey', pad=0, labelsize=8)
	
	#ax.scatter( data[:,2], data[:,3], c='tab:blue', label=r'$\alpha(1-3)$', alpha= 0.7)
	ax.plot( uvd, a_mth13, c='tab:blue', ls='-', label=r'theoretical $\alpha_{(1-3)mm}$', alpha=0.3 )
	ax.plot( uvd, a_mth37, c='tab:orange', ls='-', label=r'theoretical $\alpha_{(3-7)mm}$', alpha=0.5 )
	ax.fill_between( uvd, y1=a_obs13[2], y2=a_obs13[0], color='tab:blue', alpha=0.1, edgecolor=None )
	ax.fill_between( uvd, y1=a_obs37[2], y2=a_obs37[0], color='tab:orange', alpha=0.1, edgecolor=None )
	ax.scatter( uvd, a_obs13[1], c='tab:blue', label=r'observed $\alpha_{(1-3)mm}$')
	ax.scatter( uvd, a_obs37[1], c='tab:orange', label=r'observed $\alpha_{(3-7)mm}$')
	ax.set( xlabel='uv-distance [k$\lambda$]', ylabel=r'$\alpha$ index', ylim=[1.5, None], xscale='log', title=ptitle) #,  xscale='log', yscale='log', xlim=[1e-4, 20], ylim=[1e-2, 1e5])
	ax.legend()
	# ax.grid( True, axis='both', alpha=0.5, linestyle=':')
	# plt.show()
	[fig.savefig( savedir_prefix + ptitle.replace(' ', '_') + quantity + fig_ext, bbox_inches='tight', dpi=200) for fig_ext in ('.png', '.pdf') ]



def visib_ratios_plot( model='full', quantity='mod', binsize=40e3, max_baseline=5e5, logbins=False, CC=True, targetslist=OKlist):
	'''
	Visualise for ALL targets in our sample the ratios of quantity between 1,3,7mm as function of the baseline. 
	'''
	import uvplot as uvp
	from galario import deg, arcsec
	# mpl.use('macosx')
	from scipy.interpolate import Akima1DInterpolator
	plt.rcParams.update({ 'font.size':9, 'legend.fontsize':9, 'figure.dpi':100})	
	mpl.style.use('fast')
	cc_dict = { 8.9e-4: 'concat', 3e-3: 'concat', 7e-3: 'concat' }		# compact configuration for each wavelength, env-oriented
	sc_dict = { 8.9e-4: '11.4', 3e-3: '11.7', 7e-3: '11.8' }			# regular configuration for each wavelength, disc-oriented
	conf_dict = sc_dict if not CC else cc_dict

	def plot_quantity( quant, uvtab):
		if quant=='Re':
			return uvtab.bin_re
		elif quant=='Im':
			return uvtab.bin_re
		elif quant=='amp':
			return np.arctan2( uvtab.bin_im, uvtab.bin_re )
		elif quant=='mod':
			return np.sqrt( uvtab.bin_im**2 + uvtab.bin_re**2 )
		
	# savedir_prefix =  '/Users/gcolumba/PostDoc_Mac/sshfs_dir/' 
	# resdir_7mm = savedir_prefix + '7mm/run_5400s_2c_xsrc_CC/'
	# resdir_3mm = savedir_prefix + '3mm/run_1800s_2c_xsrc_CC/'
	# resdir_1mm = savedir_prefix + '1mm/run_150s_2c_xsrc_CC/'
	resdir_7mm = savedir_prefix + '7mm/run_10800s_2c_xsrc_SC/'
	resdir_3mm = savedir_prefix + '3mm/run_3600s_2c_xsrc_SC/'
	resdir_1mm = savedir_prefix + '1mm/run_300s_2c_xsrc_SC/'
	disk_dirs = sorted(glob.glob( resdir_3mm + 'disk*'))
	n_disks = len(disk_dirs)
	ncols = 10 ; nrows = int(np.ceil( n_disks / ncols))
	ptitle = f'Ratios of {quantity}(V) - {model} model'
	fig, axes = plt.subplots( nrows, ncols, figsize=(2*ncols, 2.35*nrows), squeeze=False, sharex=True, sharey='row')
	axes = axes.flatten()

	wles = [8.9e-4, 3e-3, 7e-3] ; dirs = [resdir_1mm, resdir_3mm, resdir_7mm]
	v_obs = 299792458.0/np.array(wles)

	a_tab = [] ; ratio_book = [] ; R_disc = []
	truths_df = pd.read_csv( truth_path, sep='\t', index_col=0 )
	analistdir = ''
	rdf_1mm = pd.read_csv( resdir_1mm + f'analysis_results{run_name.replace(" ","_")}-{analistdir}'[:-1] + '.txt', sep='\t', index_col='source')
	for d in range( n_disks):	# n_disks
		uvtabs = [0,0,0] ; comptabs = [0,0,0]
		diskname = os.path.basename( disk_dirs[d] )		# "diskNN_xx"	
		if int( diskname[4:6]) in targetslist:
			try:		# Load uvtable using uvplot
				for i in range(3):		# iterate on wavelength
					dRA, dDec = np.loadtxt( dirs[i] + diskname + '/bestfit_params.txt' )[-2:, 0]
					dRA *= arcsec ; dDec *= arcsec 
					uvtabs[i] = uvp.UVTable( filename= dirs[i] + diskname + f'/uvtab_C{conf_dict[wles[i]]}.txt', wle=wles[i], columns=uvp.COLUMNS_V0)		# mock-obs data
					if model != 'full':
						mod_vis = [0,0]
						with open( dirs[i] + diskname + '/visib_disk+env.npy', 'rb') as f:		# this requires two separate np.load calls to read back the two arrays
							mod_vis = [np.load( f), np.load( f)] 			# disk_vis, env_vis
						mod_i = 0 if model == 'env' else 1
						# mod_i = 1 if model == 'env' else 0		# this to compute the ratios directly on the models 
						# comptabs[i] = uvp.UVTable( uvtable=[uvtabs[i].u*wles[i], uvtabs[i].v*wles[i], mod_vis[mod_i].real, mod_vis[mod_i].imag, uvtabs[i].weights], wle=wles[i], columns=uvp.COLUMNS_V0 )
						comptabs[i] = uvp.UVTable( uvtable=[uvtabs[i].u*wles[i], uvtabs[i].v*wles[i], uvtabs[i].re - mod_vis[mod_i].real, uvtabs[i].im - mod_vis[mod_i].imag, uvtabs[i].weights], wle=wles[i], columns=uvp.COLUMNS_V0 )
						comptabs[i].apply_phase( -dRA, -dDec)  
						comptabs[i] = comptabs[i].uvcut( maxuv=max_baseline, minuv=9e3)	; comptabs[i].uvbin( binsize, logbins=logbins)	# bin it before or AFTER the subtraction ?
						del mod_vis
					uvtabs[i].apply_phase( -dRA, -dDec)  
					uvtabs[i] = uvtabs[i].uvcut( maxuv=max_baseline, minuv=9e3)	; uvtabs[i].uvbin( binsize, logbins=logbins)
			except Exception as e:
				print(f"\nCould not load uvtable for {diskname}: {e}\n")
				continue
			
			tabs = uvtabs  if model == 'full'  else comptabs		# single component table (uvtab - model)
			q1, q3, q7 = [ plot_quantity( quantity, tabs[t]) for t in range(3) ]

			uvdist3 = tabs[1].bin_uvdist   # np.where( qty > 0 , uvtab3.bin_uvdist, np.nan)	# reference baselines distances
			uvdist3[ np.isclose( uvdist3, 0) ] = np.nan
			# q3 = qty # np.where( qty > 0 , qty, np.nan) 			# reference REAL values to compute ratios
			q1 = Akima1DInterpolator( tabs[0].bin_uvdist, q1 )( uvdist3 )		# interpolate at uvdist3 points, where the ref value are binned
			q7 = Akima1DInterpolator( tabs[2].bin_uvdist, q7 )( uvdist3 )
			# re1[re1 <= 0] = np.nan ; re7[re7 <= 0] = np.nan ; 		# disregard negative Re fluxes
			ratio13 = q1 / q3
			ratio37 = q3 / q7
			a13 = - np.log10( ratio13) / np.log10( 0.89 / 3 )		# minus sign because i'm dividing for wavel, not frequency
			a37 = - np.log10( ratio37) / np.log10( 3 / 7 )
			
			M_env = truths_df.loc[ int( diskname[4:6]) ].M_env / 100		# [Msun] mass within 1000 au excluding disk
			a_tab.append( [int( diskname[4:6]), np.nanmean( a37[1:7]), M_env , np.nanmean( a13[1:7])] )			# take the first points for envelope scales
			# add theoretical spectral index (beta=1.52 from optool opacity)
			T_profile = temp_profile_Tung( lum=truths_df.loc[int( diskname[4:6])][['L_acc', 'L_int']].sum(), r=(1.22/uvdist3/2 * dist).to_value(u.au) )			# T(uvdist)
			a37_theor = 1.52 + np.log10( planck_bbody( v_obs[1], T=T_profile) / planck_bbody( v_obs[2], T=T_profile)) / np.log10( v_obs[1] / v_obs[2] )
			a13_theor = 1.52 + np.log10( planck_bbody( v_obs[0], T=T_profile) / planck_bbody( v_obs[1], T=T_profile)) / np.log10( v_obs[0] / v_obs[1] )
			ratio_book.append( [a13, a37, a13_theor, a37_theor] )
			# R_disc.append( truths_df.loc[ int( diskname[4:6]) ].R_disk )		# this one for R_sim
			R_disc.append( rdf_1mm.loc[ diskname[4:] ].R_obs )		# this one for R_obs 1mm

			# fig, axes = plt.subplots()
			axes[d].axhline( y=2, ls=':', c='gray', alpha=0.4 )			# optically thick zone
			axes[d].plot( uvdist3 *1e-3, a37_theor, c='tab:orange', ls='-', label=r'theoretical $\alpha (3-7mm)$', alpha=0.5 )
			axes[d].plot( uvdist3 *1e-3, a13_theor, c='tab:blue', ls='-', label=r'theoretical $\alpha (1-3mm)$', alpha=0.3 )
			axes[d].scatter( uvdist3 *1e-3, a37, c='tab:orange', s=16, label='observed 3mm/7mm' )		# all three ratios in same subplot for each target
			axes[d].scatter( uvdist3 *1e-3, a13, c='tab:blue', s=16, label='observed 0.9mm/3mm', alpha=0.75 )
			axes[d].set( xscale='log',  ylim=[1,4]) #, yscale='log')#, ylim=[1e-1,1e3]) ; 
			axes[d].set_title( diskname, fontsize=8)

			del uvtabs, comptabs, q1, q3, q7
			# plt.show()

	# def klambda_to_au(x): return 1.22 * 1e-3/x * dist.to_value(u.au)		# add a physical ruler for size understanding
	# def au_to_klambda(x): return 1.22 * 1e-3/x * dist.to_value(u.au)		# 1e-3 I guess because the first xaxis is in klambda

	for ax in axes[n_disks:]: ax.set_visible(False)		# hide unused axes
	for i, ax in enumerate( axes.reshape(nrows, ncols)[0, :]):
		ax_top = ax.secondary_xaxis( 'top', functions=(klambda_to_au, au_to_klambda))
		if i==int(ncols/2): ax_top.set_xlabel('physical scale [au]')
	
	supylab = r'$\alpha$ index'	# 'Re(V) [Jy]'
	fig.subplots_adjust( wspace=0.001)	# hspace=0.001,
	fig.supylabel( supylab, weight='bold', x=0.08, fontsize=12 )
	fig.supxlabel( 'uv-distance [k$\lambda$]', fontsize=12 )		#, weight='bold'
	axes[0].legend( loc='lower right', bbox_transform=fig.transFigure, bbox_to_anchor=(0.9,0.1))
	fig.savefig( savedir_prefix + ptitle.replace(' ', '_') + '.pdf' , bbox_inches='tight')
	plt.close()

	# median_alpha_plot( np.reshape( ratio_book, (-1, len(uvdist3) )), uvdist3/1000, model, quantity, savedir_prefix)
	alpha_vs_Rdisc( np.reshape( ratio_book, (-1, len(uvdist3) )), uvdist3/1000, R_disc, savedir_prefix )
	np.savetxt( savedir_prefix + 'alpha137_env.txt', np.reshape( a_tab, (-1,4)) )			# save for M_env - alpha correlation
	print('Visib ratio plot saved')



def collective_uvplot( wle, results_dir, run_name, two_comp, config_name, binsize=50e3, logbins=False, targetslist=OKlist):
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
				bestfit = np.loadtxt( disk_dirs[d] + 'bestfit_params.txt')[:,0]		# only take the best values (no errors)
				inc, PA, dRA, dDec = bestfit[-4:]
				inc *= deg ; PA *= deg ; dRA *= arcsec ; dDec *= arcsec ;		# convert to [rad] !
				galargs = get_galargs( wle=wle, config_name=config)
				chi2, vis_mod = galario_model( pars= bestfit, galargs=galargs, two_comp=two_comp )[-2:]
				red_chi2 = chi2/(galargs[2] - len(bestfit)) 	# chi2/(42*(42-1)/2 - len(bestfit))	# with 42 antennas
				u, v, Re_obs, Im_obs, w = galargs[-5:]
				axins = axes[d].inset_axes( [0,-0.2 , 1, 0.2] )
				# observations uv-plot !
				uv = uvp.UVTable( uvtable=[u*wle, v*wle, Re_obs, Im_obs, w], wle=wle, columns=uvp.COLUMNS_V0 )
				# uv = uv.uvcut( maxuv=np.inf, minuv=uvd_min)
				uv.apply_phase( -dRA, -dDec)         # center the source on the phase center
				# uv.deproject( inc=inc/deg, PA=PA/deg, inplace=True)
				uv.uvbin( binsize, logbins=logbins)		# , 'zorder':1.9
				mask = slice(None) # uv.bin_count != 0 # slice(None)
				uvdist = uv.bin_uvdist[mask]/1000
				data_dict = {'fmt':'o', 'ms':3, 'color':'k', 'linewidth':0, 'capsize':1.2, 'capthick':1, 'ecolor':'gray', 'elinewidth':0.2, 'label':'Data', 'alpha':0.7}
				axes[d].errorbar( x=uvdist, y=uv.bin_re[mask], yerr=uv.bin_re_err[mask], **data_dict)
				axins.errorbar( x=uvdist, y=uv.bin_im[mask], yerr=uv.bin_im_err[mask], **data_dict)
				del uv
				# model uv-plot : disk (+ env)
				uv_mod = uvp.UVTable( uvtable=[u*wle, v*wle, vis_mod.real, vis_mod.imag, w], wle=wle, columns=uvp.COLUMNS_V0 )
				# uv_mod = uv_mod.uvcut( maxuv=np.inf, minuv=uvd_min)
				uv_mod.apply_phase( -dRA, -dDec)    # center the source on the phase center
				# uv_mod.deproject( inc=inc/deg, PA=PA/deg, inplace=True)
				uv_mod.uvbin( binsize, logbins=logbins )
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
						# uv_mod = uv_mod.uvcut( maxuv=np.inf, minuv=uvd_min)
						uv_mod.apply_phase( -dRA, -dDec)     	# center on the phase center
						# uv_mod.deproject( inc=inc/deg, PA=PA/deg, inplace=True)
						uv_mod.uvbin( binsize, logbins=logbins ) #; mask = slice(None) #uv_mod.bin_count != 0
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
	fig.supxlabel('uv-distance [k$\mathrm{\lambda}$]', fontsize=10 )		#, weight='bold'
	axes[0].legend()
	fig.savefig( results_dir + ptitle.replace(' ', '_') + '.pdf' , bbox_inches='tight')
	# plt.show()
	plt.close()
	print('Collective uvplot saved')


def collective_residuals_plot( results_dir, run_name, as_margin=2, targetslist=OKlist):
	'''
	Make uvplots of all regressed targets in one figure
	'''
	import casatools as cto
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
			# print( 'reading residuals of ', diskname)
			try:
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
	parser.add_argument('-config', type=str, default='11.7', help='ALMA antenna configuration (default: 11.7)')
	parser.add_argument('-Texp', type=int, default=3600, help='exposure time (default: 3600s)')
	parser.add_argument('-2c', action='store_true', help='use two-component model (default: False)')
	parser.add_argument('-fullsamp', action='store_true', help='analyse all OK targets (default: False)')
	parser.add_argument('-monosrc', action='store_true', help='do NOT use multi-source fit and clip the extra sources (default: False)')
	parser.add_argument('-compconf', action='store_true', help='add a compact configuration observation to data (default: False)')
	parser.add_argument('-Tavg', type=int, default=122, help='average temperature of disks for flux-mass conversion (default: 122K)')
	parser.add_argument('-r90', action='store_true', help='use R_90 as obs radius (default: R_95)')
	# parser.add_argument('-simple_M', action='store_true', help='calc mass with simplest thin case approx (default: False)')
	args = vars( parser.parse_args() )

	model_comps = '2c' if args['2c'] else '1c'
	xsrc_flag = 'mono' if args['monosrc'] else 'xsrc'
	config_name = 'alma.cycle' + args['config']		# main antenna configuration
	wle = float(args["RT_wavel"]) /1e6		# [m]	assuming wle is exact as names
	folder_wle = f'{round(wle*1e3)}mm/'
	cc_dict = { 8.9e-4:1, 3e-3: 4, 7e-3:6 }			# compact configuration for each wavelength, env-oriented
	cc_name = f'alma.cycle11.{cc_dict[wle]}'	
	config_list = [config_name, cc_name] if args['compconf']==True else [config_name]
	config = 'concat' if len( config_list) > 1 else config_name
	conf_flag = 'CC' if args['compconf'] else 'SC'
	savedir = savedir_prefix + folder_wle + f'run_{args["Texp"]}s_{model_comps}_{xsrc_flag}_{conf_flag}/' 		# results directory name

	run_name = f'-{ folder_wle.strip("/") } {args["Texp"]}s {model_comps} {conf_flag}'
	os.makedirs( savedir + 'Figures_png/', exist_ok=True ) ; os.makedirs( savedir + 'Figures_pdf/', exist_ok=True )
	if args["fullsamp"]:
		analist = OKlist
		analistdir = 'OKlist/'
		os.makedirs( savedir + 'Figures_png/' + analistdir, exist_ok=True ) ; os.makedirs( savedir + 'Figures_pdf/' + analistdir, exist_ok=True )
	else:
		analist = prettylist
		analistdir = ''
	#assess_SNR( results_dir=savedir, config_name=config, run_name=run_name, simple=False )

	#rdf = main_analysis( analist, wle, savedir, config_name, run_name, analistdir, T_avg=args['Tavg'], r95=not(args['r90']), figures=True )
	#rdf = pd.read_csv( savedir + f'analysis_results{run_name.replace(" ","_")}-{analistdir}'[:-1] + '.txt', sep='\t', index_col='source')
	# column_plotter( analistdir, conf_flag, model_comps, xsrc_flag, Texps=np.array([300,3600,10800])//1 , y_max=17)
	#skymodel_ratiosplot( 'disk20_xy', rdf, projections=['xy', 'yz'], margin=25, fs=(5.2,2.4))

	# plot_mass_compare( rdf, run_name, Tavg=args['Tavg'], simple_M=False, analistdir=analistdir, errors=False, logratio=False, fs=(3,3.1))
	#plot_radius_compare( rdf, 0.00002, run_name, analistdir, False, fs=(12,12))
	# fit_Mobs( results_dir=savedir, run_name=run_name, analistdir=analistdir, logfit=True)
	# plot_Mobs_multiwave( analistdir, conf_flag, model_comps, xsrc_flag )

	# collective_uvplot( wle, results_dir=savedir, run_name=run_suffix, two_comp=args['2c'], config_name=config, logbins=True)
	# collective_residuals_plot( results_dir=savedir, run_name=run_suffix )
	visib_ratios_plot( model='full', quantity='Re', binsize=30e3, max_baseline=1200e3, logbins=False, CC=args['compconf'], targetslist=analist)
	# alpha_Menv_plot( model='full', quantity='mod' )
	# plot_sample_props( savedir, run_suffix, analistdir )

	# for mod in ['full','env']:
	#             for q in ['Re','mod']:
	#                     visib_ratios_plot( model=mod, quantity=q, binsize=20e3, max_baseline=2e5, targetslist=analist)
	#                     alpha_Menv_plot( model=mod, quantity=q )


