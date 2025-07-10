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
	parser.add_argument('RT_wavel', type=int, help='obs wavelength (3000 or 7000 [um]) (default: 3000)')
	parser.add_argument('-Texp', type=int, default=3600, help='exposure time (default: 5000s)')
	parser.add_argument('-nsteps', type=int, default=5000, help='MCMC steps (default: 5000)')
	parser.add_argument('-2c', action='store_true', help='use two-component model (default: False)')
	parser.add_argument('-damp', action='store_true', help='damp the sky model (default: False)')
	parser.add_argument('-monosrc', action='store_true', help='do NOT use multi-source fit and clip the extra sources (default: False)')
	args = vars( parser.parse_args() )

	# wle = 0.00299792458 if args['RT_wavel']=='3000' else 
	wle = float(args["RT_wavel"]) *1e-6		# [m]	assuming wle is exact as names
	folder_wle = f'{round(wle*1e3)}mm/'

	if args['diskname'] == 'all':

		fitslist = sorted( glob.glob( data_folder + folder_wle + '*.fits') )
		print( len(fitslist), 'files found')
		if fitslist == []:
			print('NO FILES FOUND, check again the folder path!')
			sys.exit()

		for fname in fitslist:
			diskname = fname.replace( data_folder + folder_wle, '' ).replace( f'_{args["RT_wavel"]}um', '').strip('.fits')
			if int( diskname.strip( 'disk_xyz') ) in Tung_nofit:
				print('Skipping NO-FIT target: ', fname , '\n')
			else:
				#try:
				generate_mock_obs( fname, T_exp=args['Texp'], damp=args['damp'], monosource=args['monosrc'],
					data_folder=data_folder+folder_wle, savedir=savedir+folder_wle, ptgfile=ptgfile, wle=wle)	
				mcmc_regress( diskname, args['Texp'], nsteps=args['nsteps'], two_components=args['2c'], monosource=args['monosrc'], Ncpu=Ncpu, savedir=savedir+folder_wle, wle=wle)
				#except: print( 'skipping', diskname)
				# try:
				# 	os.chdir( savedir+folder_wle + diskname )
				# 	bestfit_plots( diskname=diskname, T_exp=args['Texp'], ptgfile=ptgfile)
				# except: print('No res for ', diskname)
	
	else:
		fname = data_folder + folder_wle + args['diskname'] + f'_{args["RT_wavel"]}um.fits'

		generate_mock_obs( fname, T_exp=args['Texp'], damp=args['damp'], monosource=args['monosrc'], nRMS=1.5,
				   data_folder=data_folder + folder_wle, savedir=savedir+folder_wle, ptgfile=ptgfile, wle=wle )

		# mcmc_regress( args['diskname'], args['Texp'], nsteps=args['nsteps'], two_components=args['2c'],
		# 		Ncpu=Ncpu, savedir=savedir+folder_wle, monosource=args['monosrc'], nRMS=1.5, wle=wle)
		# os.chdir( savedir+folder_wle + args['diskname'] )
		# bestfit_plots( args['diskname'], args['Texp'], two_comp=args['2c'], monosource=args['monosrc'], walksigma=4, wle=wle )




