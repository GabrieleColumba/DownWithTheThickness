# # # Full pipeline CLI trigger, to perform mock obs + galario & MCMC fitting. 

import os, argparse
import local_variables as loc		# file with the local path pointers and cpu settings
import visibfit_functions as vf
from mockobs import perform_mock_obs
from mcmc_regress import regress_mcmc
from diskplots import bestfit_plots, triplot

# Tung_nofit = [29, 43, 63, 72, 75, 82, 83]		# targets excluded by Tung+24 study (because multiples ?)	
NOfit_list = [29, 52, 63, 70, 75]
cc_dict = { 8.9e-4: 1, 3e-3: 4, 7e-3: 6 }			# compact configuration for each wavelength, env-oriented


def main( run_meta: vf.RunParams, NOfit_list=NOfit_list ):
	'''
	Main routine: run the mock observation and/or the MCMC regression for the given target(s).
	'''
	if run_meta.disk_N in NOfit_list:
		return print('Skipping NO-FIT target: ', run_meta.diskname , '\n')
	
	else:
		print( '\nRunning for: \t', run_meta.diskname )
		if not run_meta.replot:
			perform_mock_obs( run_meta=run_meta, ptgfile=loc.ptgfile )
			regress_mcmc( run_meta=run_meta )

		bestfit_plots( run_meta=run_meta, walksigma=3 )
		# triplot( run_meta=run_meta, as_margin=7, rulersize=500)		# one-off fig for paper
		return print( f'\nDone with {run_meta.diskname} !\n' )



if __name__=='__main__':

	parser = argparse.ArgumentParser()		# parsing the run settings
	parser.add_argument('diskname', type=str, help='name of the diskNN_xx, index, or "all" (default: None)')
	parser.add_argument('wle_um', type=int, help='obs wavelength in [um] (default: 3000)')
	parser.add_argument('-Texp', type=int, default=3600, help='exposure time (default: 3600s)')
	parser.add_argument('-config', type=str, default='11.7', help='ALMA antenna configuration (default: 11.7)')
	parser.add_argument('-nsteps', type=int, default=5000, help='MCMC steps (default: 5000)')
	parser.add_argument('-two_comp', action='store_true', help='use two-component model (default: False)')
	parser.add_argument('-replot_only', action='store_true', help='just replot all the best-fit figures (default: False)')
	parser.add_argument('-nRMS', type=float, default=10, help='nRMS to threshold the xsrc detection (default: 10)')
	parser.add_argument('-compact_conf', action='store_true', help='add a compact configuration observation to data (default: False)')
	parser.add_argument('-damp', action='store_true', help='damp artificially the sky model (default: False) [deprecated]')		# deprecated
	parser.add_argument('-monosrc', action='store_true', help='clip the extra sources before mock-obs (default: False)')
	args = vars( parser.parse_args() )

	meta = vf.RunParams( **args, local=loc, cc_dict=cc_dict )		# manage the run parameters
	
	if args['diskname'] == 'all':				# run all targets sequentially
		for fname in meta.disklist:
			diskname = os.path.basename( fname ).replace( f'_{args["wle_um"]}um', '').replace('.fits','')
			meta.set_target( diskname=diskname, fitspath=fname )		# update the diskname and fitspath (skymodel) in the metadata

			main( run_meta=meta )
			
	else:		# regress one disk per task (in parallel, suited for sbatch arrays)
		try:
			idx = int( args['diskname'] )		# if it's a number
			fname = meta.disklist[ idx ]
			diskname = os.path.basename( fname ).replace( f'_{args["wle_um"]}um', '').replace('.fits','')
		except:
			diskname = args['diskname']			# if it's actually like "diskNN_xx"
			fname = meta.datadir + diskname + f'_{args["wle_um"]}um.fits'

		meta.set_target( diskname=diskname, fitspath=fname )		# update the diskname and fitspath (skymodel) in the metadata
		main( run_meta=meta )



	# 		# if args['compconf']:
	# 		# 	print('\nCleaning up the various intermediate files !\n')
	# 			# os.system('rm -rf *concat*')		# required to replot
	# 			# os.system( f'rm -rf *.last {diskname}.alma*' )
	# 			# os.system( f'rm -rvf bestmod xsrc_sub  *.last {diskname}.alma*' )
