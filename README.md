# DownWithTheThickness
Code for the mock-observation with CASA in ALMA band 1 &amp; 3 of simulated sky models + GALARIO fitting 


Feel free to contact me for any inquiries!







# example of file local_variables.py

%# LOCAL MAC version for pointing at the right files
import os
os.environ["OMP_NUM_THREADS"] = "9" 	# set the number of threads to use for galario

data_prefix =    '/Users/gcolumba/PostDoc_Mac/PostProc/simulations/'        # directory with RT sky images in the given band
savedir_prefix = '/Users/gcolumba/PostDoc_Mac/PostProc/testruns/'           # local
%# savedir_prefix = '/Users/gcolumba/PostDoc_Mac/sshfs_dir/'                   # cluster sshfs
ptgfile = '/Users/gcolumba/PostDoc_Mac/PostProc/Scripts/DownWithTheThickness/pointing_single1x.ptg.txt'
truth_path = '/Users/gcolumba/PostDoc_Mac/PostProc/Scripts/DownWithTheThickness/Tungs_truths.dat'
Ncpu = None 	# number of CPUs to use for emcee ?

progbar = True
Nwalkers = 40		# 30 local (Mac), 50 remote (cluster)
fig_ext = '.png'
