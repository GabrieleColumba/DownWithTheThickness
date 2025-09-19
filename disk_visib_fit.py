# # # Full pipeline to mock obs + galario fit. 

import glob
import sys
from local_variables import *		# file with the local path pointers and cpu settings
from visibfit_functions import *
import argparse

# Tung_nofit = [29, 43, 63, 72, 75, 82, 83]		# targets excluded by Tung+24 study (because multiples)	
NOfit = [29, 63, 75]

if __name__=='__main__':

	parser = argparse.ArgumentParser()		# parsing the name of the disk file to read
	parser.add_argument('diskname', type=str, help='name of the diskNN_xx .fits file OR INDEX (default: None)')
	parser.add_argument('RT_wavel', type=int, help='obs wavelength (3000 or 7000 [um]) (default: 3000)')
	parser.add_argument('-Texp', type=int, default=3600, help='exposure time (default: 3600s)')
	parser.add_argument('-config', type=str, default='11.7', help='ALMA antenna configuration (default: 11.7)')
	parser.add_argument('-nsteps', type=int, default=5000, help='MCMC steps (default: 5000)')
	parser.add_argument('-2c', action='store_true', help='use two-component model (default: False)')
	parser.add_argument('-replot_only', action='store_true', help='replot all the bestfit plots for the fitting part (default: False)')
	parser.add_argument('-damp', action='store_true', help='damp the sky model (default: False)')
	parser.add_argument('-monosrc', action='store_true', help='do NOT use multi-source fit and clip the extra sources (default: False)')
	args = vars( parser.parse_args() )

	model_comps = '2c' if args['2c'] else 'g+'
	xsrc_flag = 'mono' if args['monosrc'] else 'xsrc'
	config_name = 'alma.cycle' + args["config"]
	wle = float(args["RT_wavel"]) *1e-6		# [m]	assuming wle is exact as names
	folder_wle = f'{round(wle*1e3)}mm/'
	savedir = savedir_prefix + folder_wle + f'run_{args["Texp"]}s_{model_comps}_{xsrc_flag}/'		# results directory name
	try: os.mkdir( savedir )
	except FileExistsError: print('Parent run directory already existent.')

	fitslist = sorted( glob.glob( data_folder + folder_wle + '*.fits') )
	if fitslist == []:
		print('NO FILES FOUND, check again the folder path!')
		sys.exit()
	
	if args['diskname'] == 'all':

		for fname in fitslist:
			diskname = fname.replace( data_folder + folder_wle, '' ).replace( f'_{args["RT_wavel"]}um', '').strip('.fits')

			if args["replot_only"]:
				bestfit_plots( diskname, args['Texp'], two_comp=args['2c'], monosource=args['monosrc'], walksigma=4, wle=wle, savedir=savedir, config_name=config_name )
			else:
				if int( diskname.strip( 'disk_xyz') ) in NOfit:
					print('Skipping NO-FIT target: ', fname , '\n')
				else:
					#try:
					generate_mock_obs( fname, T_exp=args['Texp'], damp=args['damp'], monosource=args['monosrc'],
						data_folder=data_folder+folder_wle, savedir=savedir, ptgfile=ptgfile, wle=wle, config_name=config_name)	
					mcmc_regress( diskname, args['Texp'], nsteps=args['nsteps'], two_components=args['2c'], monosource=args['monosrc'],
					Ncpu=Ncpu, savedir=savedir, wle=wle, config_name=config_name)
					#except: print( 'skipping', diskname)

	else:
		try:
			idx = int( args['diskname'] )		# if it's a number
			fname = fitslist[ idx ]
		except:
			fname = data_folder + folder_wle + args['diskname'] + f'_{args["RT_wavel"]}um.fits'
		
		diskname = fname.replace( data_folder + folder_wle, '' ).replace( f'_{args["RT_wavel"]}um', '').strip('.fits')

		if args["replot_only"]:
			bestfit_plots( diskname, args['Texp'], two_comp=args['2c'], monosource=args['monosrc'], walksigma=4, wle=wle, savedir=savedir, config_name=config_name )
		else:
			if int( diskname.strip( 'disk_xyz') ) in NOfit:
				print('Skipping NO-FIT target: ', fname , '\n')
			else:
				print( '\nRunning for: \t', diskname )

				generate_mock_obs( fname, T_exp=args['Texp'], damp=args['damp'], monosource=args['monosrc'], nRMS=1.5,
						data_folder=data_folder+folder_wle, savedir=savedir, ptgfile=ptgfile, wle=wle, config_name=config_name )

				mcmc_regress( diskname, args['Texp'], nsteps=args['nsteps'], two_components=args['2c'],
						Ncpu=Ncpu, savedir=savedir, monosource=args['monosrc'], nRMS=1.5, wle=wle, config_name=config_name)
