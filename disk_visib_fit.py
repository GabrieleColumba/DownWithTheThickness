# # # Full pipeline to perform mock obs + galario & MCMC fitting. 

import glob, os
import local_variables as loc		# file with the local path pointers and cpu settings
import visibfit_functions as visf
import argparse

# Tung_nofit = [29, 43, 63, 72, 75, 82, 83]		# targets excluded by Tung+24 study (because multiples ?)	
NOfit = [29, 52, 63, 70, 75]

if __name__=='__main__':

	parser = argparse.ArgumentParser()		# parsing the name of the disk file to read
	parser.add_argument('diskname', type=str, help='name of the diskNN_xx .fits file OR INDEX (default: None)')
	parser.add_argument('RT_wavel', type=int, help='obs wavelength (3000 or 7000 [um]) (default: 3000)')
	parser.add_argument('-Texp', type=int, default=3600, help='exposure time (default: 3600s)')
	parser.add_argument('-config', type=str, default='11.7', help='ALMA antenna configuration (default: 11.7)')
	parser.add_argument('-nsteps', type=int, default=5000, help='MCMC steps (default: 5000)')
	parser.add_argument('-2c', action='store_true', help='use two-component model (default: False)')
	parser.add_argument('-replot_only', action='store_true', help='replot all the bestfit plots for the fitting part (default: False)')
	parser.add_argument('-nRMS', type=float, default=10, help='nRMS to threshold the xsrc detection (default: 10)')
	parser.add_argument('-compconf', action='store_true', help='add a compact configuration observation to data (default: False)')
	parser.add_argument('-damp', action='store_true', help='damp the sky model (default: False)')		# deprecated now
	parser.add_argument('-monosrc', action='store_true', help='do NOT use multi-source fit and clip the extra sources (default: False)')
	args = vars( parser.parse_args() )

	model_comps = '2c' if args['2c'] else '1c'
	xsrc_flag = 'mono' if args['monosrc'] else 'xsrc'
	config_name = 'alma.cycle' + args["config"]		# main antenna configuration (extended, disk oriented)
	wle = float(args["RT_wavel"]) /1e6				# [m]	assuming wle is exact as names
	cc_dict = { 8.9e-4:1, 3e-3: 4, 7e-3:6 }			# compact configuration for each wavelength, env-oriented
	cc_name = f'alma.cycle11.{cc_dict[wle]}'	
	config_list = [config_name, cc_name] if args['compconf']==True else [config_name]
	conf_flag = 'CC' if args['compconf']==True else 'SC'
	folder_wle = f'{round(wle*1e3)}mm/'
	data_path  = loc.data_prefix + folder_wle
	savedir = loc.savedir_prefix + folder_wle + f'run_{args["Texp"]}s_{model_comps}_{xsrc_flag}_{conf_flag}/'		# results directory name
	try: os.mkdir( savedir )
	except FileExistsError: print('Parent run directory already existent.')

	filepath = savedir + 'disk*'  if args["replot_only"]  else  data_path + '*.fits'	# check either the results or the sky models
	disklist = sorted( glob.glob( filepath ) )
	
	if args['diskname'] == 'all':		# run all disk regressions sequentially

		for fname in disklist:
			diskname = os.path.basename( fname ).replace( f'_{args["RT_wavel"]}um', '').strip('.fits')
				
			if int( diskname.strip( 'disk_xyz') ) in NOfit:
				print('Skipping NO-FIT target: ', fname , '\n')
			else:
				if not args["replot_only"]: 				# perform the regression from scratch
					print( '\nRunning for: \t', diskname )
					visf.perform_mock_obs( fname, T_exp=args['Texp'], damp=args['damp'], monosource=args['monosrc'], nRMS=args['nRMS'],
							data_folder=data_path, savedir=savedir, ptgfile=loc.ptgfile, wle=wle, config_name=config_list )
					#visf.mcmc_regress( diskname, nsteps=args['nsteps'], two_components=args['2c'],
					#		Ncpu=Ncpu, savedir=savedir, wle=wle, config_name=config_list)
				
				# if os.path.exists( savedir + diskname + '/galario_sky-model_bestfit.png' ):
				# 	pass
				# else: 
				#visf.bestfit_plots( diskname, args['Texp'], two_comp=args['2c'], nRMS=args['nRMS'], walksigma=3, wle=wle, savedir=savedir, config_name=config_list )

	else:		# regress one disk per task (suited for sbatch arrays)
		
		try:
			idx = int( args['diskname'] )		# if it's a number
			fname = disklist[ idx ]
			diskname = os.path.basename( fname ).replace( f'_{args["RT_wavel"]}um', '').strip('.fits')
		except:
			diskname = args['diskname']
			fname = data_path + diskname + f'_{args["RT_wavel"]}um.fits'
		
		if int( diskname.strip( 'disk_xyz') ) in NOfit:
			print('Skipping NO-FIT target: ', fname , '\n')
		else:
			print( '\nRunning for: \t', diskname )
			if not args["replot_only"]:
				visf.perform_mock_obs( fname, T_exp=args['Texp'], damp=args['damp'], monosource=args['monosrc'], nRMS=args['nRMS'],
							data_folder=data_path, savedir=savedir, ptgfile=loc.ptgfile, wle=wle, config_name=config_list )
				visf.mcmc_regress( diskname, nsteps=args['nsteps'], two_components=args['2c'],
							Ncpu=loc.Ncpu, savedir=savedir, wle=wle, config_name=config_list)

			MSname = savedir + diskname + f'/{diskname}.concat.noisy.ms' if args['compconf'] else savedir + diskname + f'/{diskname}.{config_name}.noisy.ms'
			run_name = f'{round(wle*1e3)}mm_' + os.path.basename( savedir[:-1] ).replace('run_', '').replace('_xsrc', '')
			visf.triplot( diskname, MSname, run_name, as_margin=7, rulersize=500)
			visf.bestfit_plots( diskname, args['Texp'], two_comp=args['2c'], nRMS=args['nRMS'], walksigma=3, wle=wle, savedir=savedir, config_name=config_list )
			
			if args['compconf']:
				print('\nCleaning up the various intermediate files !\n')
				# os.system('rm -rf *concat*')		# required to replot
				# os.system( f'rm -rf *.last {diskname}.alma*' )
				# os.system( f'rm -rvf bestmod xsrc_sub  *.last {diskname}.alma*' )