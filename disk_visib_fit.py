# # # Full pipeline to mock obs + galario fit. 

import glob
import sys
from local_variables import *		# file with the local path pointers and cpu settings
from visibfit_functions import *
import argparse

Tung_nofit = [29, 43, 63, 72, 75, 82, 83]		# targets excluded by Tung+24 study (because multiples)	


if __name__=='__main__':

	parser = argparse.ArgumentParser()		# parsing the name of the disk file to read
	parser.add_argument('diskname', type=str, help='name(s) of the diskNN_xx .fits file (default: None)')
	parser.add_argument('RT_wavel', type=int, help='obs wavelength (3000um or 7000um) (default: 3000)')
	parser.add_argument('-Texp', type=int, default=3600, help='exposure time (default: 5000s)')
	parser.add_argument('-nsteps', type=int, default=5000, help='MCMC steps (default: 5000)')
	parser.add_argument('-2c', action='store_true', help='use two-component model (default: False)')
	parser.add_argument('-damp', action='store_true', help='damp the sky model (default: False)')
	parser.add_argument('-monosrc', action='store_true', help='do NOT use multi-source fit and clip the extra sources (default: False)')
	args = vars( parser.parse_args() )

	if args['diskname'] == 'all':

		fitslist = sorted( glob.glob( data_folder + '*.fits') )
		if fitslist == []:
			print('NO FILES FOUND, check again the folder path!')
			sys.exit()
		print( len(fitslist), 'files found')

		for fname in fitslist:
			diskname = fname.replace( data_folder, '' ).replace( f'_{args["RT_wavel"]}um', '').strip('.fits')
			if int( diskname.strip( 'disk_xyz') ) in Tung_nofit:
				print('Skipping NO-FIT target: ', fname , '\n')
			else:
				try:
					generate_mock_obs( fname, T_exp=args['Texp'], damp=args['damp'], monosource=args['monosrc'],
					    data_folder=data_folder, savedir=savedir, ptgfile=ptgfile)	
					mcmc_regress( diskname, nsteps=args['nsteps'], two_components=args['2c'], monosource=args['monosrc'], Ncpu=Ncpu, savedir=savedir)
				except: print( 'skipping', diskname)
				# try:
				# 	os.chdir( savedir + diskname )
				# 	residuals_mock_plot( diskname=diskname, T_exp=args['Texp'], ptgfile=ptgfile)
				# except: print('No res for ', diskname)
	else:
		# # open the fits file with header
		fname = data_folder + args['diskname'] + f'_{args["RT_wavel"]}um.fits'
		# main_fit( fname, T_exp=args['Texp'], nsteps=args['nsteps'], Ncpu=Ncpu, two_components=True, damp=False, 
		# 	  	 data_folder=data_folder, savedir=savedir, ptgfile=ptgfile)


		generate_mock_obs( fname, T_exp=args['Texp'], damp=args['damp'], monosource=args['monosrc'],
				   data_folder=data_folder, savedir=savedir, ptgfile=ptgfile )

		mcmc_regress( args['diskname'], args['Texp'], nsteps=args['nsteps'], two_components=args['2c'],
			    Ncpu=Ncpu, savedir=savedir, monosource=args['monosrc'])
		# os.chdir( savedir + args['diskname'] )
		# bestfit_plots( args['diskname'], args['Texp'], two_comp=args['2c'], monosource=args['monosrc'], walksigma=4 )




