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
	parser.add_argument('-2c', action='store_true', help='use two-component model (default: False)')
	parser.add_argument('-damp', action='store_true', help='damp the sky model (default: False)')
	parser.add_argument('-monosrc', action='store_true', help='do NOT use multi-source fit and clip the extra sources (default: False)')
	args = vars( parser.parse_args() )

	# if args['diskname'] == 'all':

	fitslist = sorted( glob.glob( savedir + 'disk*') )
	print( len(fitslist), 'files found')
	if fitslist == []:
		print('NO FILES FOUND, check again the folder path!')
		sys.exit()

		for fname in fitslist:
			diskname = fname.replace( savedir, '' )
			try:
				os.chdir( savedir + diskname )
				bestfit_plots( diskname, args['Texp'], two_comp=args['2c'], monosource=args['monosrc'], walksigma=4 )
			except: print( 'skipping', diskname)



