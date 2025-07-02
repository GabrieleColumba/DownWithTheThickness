# # EVALUATE THE SNR OF THE MOCK OBSERVATIONS FOR THE GIVEN EXPOSURE TIME T_exp. 

import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import os
import pandas as pd
import glob
import sys
import casatools as cto
from scipy.optimize import curve_fit



T_exp = 600    # s
config_name = 'alma.cycle11.6'
savedir = '/Users/gcolumba/PostDoc_Mac/PostProc/runs_600s/'		# where the simanalyze products are saved


def crop_image( img, centre=None, margins=[100, 100] ):
	'''Select a subimage of margins pixels around the centre (odd size).'''
	if centre is None:      	# use the middle of the image
		centre = (np.array( img.shape)/2 ).astype(int)
	return img[ centre[0] - margins[0] : centre[0] + margins[0] +1, centre[1] - margins[1] : centre[1] + margins[1] +1]



def analytic_sens( t, a, b, c):
	'''analytical formula for the expected rms based on exposure time t'''
	return a * t**(-b) + c


def fit_sensitivity():
	'''Routine to extract the parameters of a possible analytical formula for the RMS threshold given Texp. '''
	sensit_arr = np.array([[600, 26.79], [1800, 15.47], [3600, 10.94], [5000, 9.28], [1e4, 6.56]])	# hand copied from ALMA calculator
	t_exps = sensit_arr[:, 0]	
	RMSs = sensit_arr[:, 1]
	init_guess = ([1, 1., 0.5])                     # a, b, c guesses
	param_bounds = ( [0.0, 0.0, -np.inf], [ np.inf, 100, 1e10] )      # limits on parameters
	IN_popt, IN_pcov = curve_fit( analytic_sens, xdata=t_exps, ydata=RMSs, p0=init_guess, bounds=param_bounds, absolute_sigma=True )
	fit_stds = np.sqrt(np.diag( IN_pcov ))          # from scipy doc
	# y_pred = height_prof( radii, *IN_popt)          # fit predictions
	# res = y_pred - heights                          # residues
	print( r'fit: a= %.3f , b= %.3f, c= %.3f' % tuple(IN_popt) )
	print( 'Fit 1 sigma errors:', fit_stds )

	x = np.linspace(300, t_exps[-1]+100, 1000)
	plt.scatter( t_exps, RMSs)
	plt.plot( x, analytic_sens( x, *IN_popt) , c='r')
	plt.show()
	return IN_popt


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


def rms(arr):
	return np.sqrt( np.sum( arr**2) / len(arr.flatten()) )


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


def list_under( threshold, df, column):
	'''List all the sources with a value in column under the given threshold.'''
	df_under = df[ df[column] < threshold ]
	print( df_under)
	return df_under


def plot_it(df):

	fig, ax = plt.subplots( figsize=(10,6), tight_layout=True)
	dff = df.reset_index()
	dff.plot( xticks=dff.index, rot=90, logy=True, ax=ax, marker='o')
	ax.set_xticklabels( df.index)
	ax.axhline( y=[0.002], color='gray', ls=':')
	ax.axhline( y=10, color='gray', ls='--')
	ax.axhline( y=100, color='gray', ls='-')
	ax.grid( True, axis='x', alpha=0.5, linestyle=':')
	fig.savefig( savedir + 'rough_SNR_plot.pdf' , bbox_inches='tight')
	plt.show()




if __name__=='__main__':


	fitslist = sorted( glob.glob( savedir + 'disk*') )
	if fitslist == []:    
		print('NO FILES FOUND, check again the folder path!')
		sys.exit()
	# print('Files in the list:\n')
	# print( repr( fitslist)) 
	print( len(fitslist), 'files found')

	SNRs = []

	for fname in fitslist:

		projectname = fname.strip( savedir ).strip('_3000um.fits')  	# each one a separate folder
		img_tab = f'{savedir}{projectname}/{projectname}.{config_name}.noisy.image'		# cleaned simanalyze image
		table = cto.table()
		table.open( img_tab )
		img = table.getcol('map').squeeze().copy() 
		# mpl.use('macosx')
		# plt.imshow( img.T, origin='lower', norm=mpl.colors.LogNorm(), cmap='gnuplot2')
		# plt.show()
		# print( img.shape )

		peak = np.max( crop_image(img, margins=[35,35]) )	# find the peak flux in a region around the centre
		peak_beam = peak_beam_avg( img, table=table)

		# bkg_patch = img[0:img.shape[0]//3, 0:img.shape[0]//3]	# a third of the img avoiding centre
		# noise = ( img.std() + bkg_patch.std() ) / 2		# maybe more robust than picking all or just a part
		# noise = ( rms(img) + rms(bkg_patch) ) / 2
		# noise = min_bkg_rms( img )		# the minimum rms from bkg patches
		noise = rms( img )


		snr = peak_beam / noise
		# SNRs.append( [projectname, f'{snr :.2f}', peak, peak_beam] ) 
		SNRs.append( [projectname.strip( 'disk' ), snr, peak, peak_beam, noise] )
		table.close()

	
	df = pd.DataFrame( SNRs, columns=['source', 'SNR', 'max peak', 'beam peak', 'noise']).set_index('source')
	df.to_csv( savedir + f'Peak_rough-rms_SNR_{T_exp}s.txt', sep='\t')#, float_format='%.2e')
	# np.savetxt( savedir + f'Peak-rms_SNR_{T_exp}s.txt', np.array( SNRs).reshape( len(fitslist), -1), fmt='%s %.2f %.3e %.3e' )

	# dfu = list_under( 0.002, df, 'beam peak')
	plot_it( df )