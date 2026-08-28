import glob, os


class RunParams:
	'''Metadata of the given run.'''

	def __init__(self, diskname, wle_um, Texp, config, nsteps, nRMS, two_comp, compact_conf, monosrc, replot_only, damp,
			local, cc_dict):
		self.diskname = 'unassigned'
		self.wle_um = int(wle_um)
		self.wle = wle_um / 1e6
		self.Texp = int(Texp)
		self.config = config
		self.sc_name = f'alma.cycle{config}'
		self.cc_name = f'alma.cycle{config[0:2]}.{cc_dict[self.wle]}'
		self.config_list = [self.sc_name, self.cc_name] if compact_conf else [self.sc_name]
		self.nsteps = nsteps
		self.nRMS = nRMS
		self.two_comp = two_comp
		self.comp_flag = '2c' if two_comp else '1c'
		self.compact_conf = compact_conf
		self.conf_flag = 'CC' if compact_conf else 'SC'
		self.monosrc = monosrc
		self.xsrc_flag = 'mono' if monosrc else 'xsrc'
		self.replot = replot_only
		self.damp = damp

		self.folder_wle = f'{round(self.wle * 1e3)}mm/'
		self.datadir = local.data_prefix + self.folder_wle
		self.savedir = (local.savedir_prefix + self.folder_wle
			+ f'run_{self.Texp}s_{self.comp_flag}_{self.xsrc_flag}_{self.conf_flag}/')

		if not self.replot:
			os.makedirs(self.savedir, exist_ok=True)
		self.targetnames = self.savedir + 'disk*' if self.replot else self.datadir + '*.fits'
		self.disklist = sorted(glob.glob(self.targetnames))
		self.run_name = self.folder_wle[:-1] + '_' + os.path.basename(self.savedir[:-1]).replace('run_', '').replace('_xsrc', '')

	def set_target(self, diskname, fitspath):
		self.diskname = diskname
		self.fitspath = fitspath

	@property
	def disk_N(self):
		'''Get the disk number from the name.'''
		return int(self.diskname.strip('disk_xyz'))

	@property
	def targetpath(self):
		return self.savedir + self.diskname + '/'

	def get_MS_path(self, config_name):
		'''Return the MeasurementSet pathname for a configuration or configuration list.'''
		if type(config_name) == list and len(config_name) > 1:
			MSpath = self.targetpath + f'{self.diskname}.concat.noisy.ms'
		else:
			if type(config_name) == list:
				config_name = config_name[0]
			MSpath = self.targetpath + f'{self.diskname}.{config_name}.noisy.ms'
		return MSpath
