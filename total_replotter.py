# # # Plot again the bestfit figures for the entire sample 
import glob
import sys
from local_variables import *		# file with the local path pointers and cpu settings
from visibfit_functions import *
import argparse

# [10,11,12,13,14,15,16,17,18,25,40,44,46]

if __name__=='__main__':

	parser = argparse.ArgumentParser()		# parsing the name of the disk file to read
	parser.add_argument('diskname', type=str, help='name(s) of the diskNN_xx .fits file (default: None)')
	parser.add_argument('RT_wavel', type=int, help='obs wavelength (3000um or 7000um) (default: 3000)')
	parser.add_argument('-Texp', type=int, default=3600, help='exposure time (default: 3600s)')
	parser.add_argument('-2c', action='store_true', help='use two-component model (default: False)')
	parser.add_argument('-damp', action='store_true', help='damp the sky model (default: False)')
	parser.add_argument('-monosrc', action='store_true', help='do NOT use multi-source fit and clip the extra sources (default: False)')
	args = vars( parser.parse_args() )

	model_comps = '2c' if args['2c'] else 'g+'
	xsrc_flag = 'mono' if args['monosrc'] else 'xsrc'
	wle = float(args["RT_wavel"]) *1e-6		# [m]	assuming wle is exact as names
	folder_wle = f'{round(wle*1e3)}mm/'
	savedir = savedir_prefix + f'run_{args["Texp"]}s_{model_comps}_{xsrc_flag}/' + folder_wle		# results directory name

	fitslist = sorted( glob.glob( savedir + 'disk*') )
	print( len(fitslist), 'files found')
	if fitslist == []:
		print('NO FILES FOUND, check again the folder path!')
		sys.exit()

		for fname in fitslist:
			diskname = fname.replace( savedir, '' )
			try:
				os.chdir( savedir + diskname )
				bestfit_plots( diskname, args['Texp'], two_comp=args['2c'], monosource=args['monosrc'], walksigma=4, wle=wle, savedir=savedir )
			except: print( 'skipping', diskname)



