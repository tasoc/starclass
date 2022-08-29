#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T'DA Simulation Training Sets.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import os.path
from astropy.table import Table, hstack
from . import TrainingSet

#--------------------------------------------------------------------------------------------------
def _load_targets(self):

	starlist = Table.read(os.path.join(self.input_folder, 'Data_Batch_TDA4_r1.txt'),
		format='ascii.csv',
		delimiter=',',
		comment='#',
		names=['starname', 'tmag', 'cadence', 'duration', 'eclat', 'eclon', 'Teff', 'Teff_error', 'logg', 'logg_error', 'starclass'])

	diagnostics_file = os.path.join(self.input_folder, 'diagnostics.txt')
	if os.path.exists(diagnostics_file):
		diagnostics = Table.read(diagnostics_file,
			format='ascii.csv',
			delimiter=',',
			comment='#',
			names=['variance', 'rms_hour', 'ptp'])

		starlist = hstack([starlist, diagnostics])

	starlist['starid'] = [int(row['starname'][4:]) for row in starlist]

	# Path to lightcurve:
	if self.key == 'tdasim-raw':
		starlist['lightcurve'] = [f'sysnoise/{name:s}.sysnoise' for name in starlist['starname']]
	elif self.key == 'tdasim':
		starlist['lightcurve'] = [f'noisy/{name:s}.noisy' for name in starlist['starname']]
	elif self.key == 'tdasim-clean':
		starlist['lightcurve'] = [f'clean/{name:s}.clean' for name in starlist['starname']]

	# Translation of Mikkel's identifiers into the broader
	# classes we have defined in StellarClasses:
	if self.level == 'L1':
		translate = {
			'Solar-like': 'SOLARLIKE',
			'Transit': 'ECLIPSE',
			'Eclipse': 'ECLIPSE', # short period EBs should be CONTACT_ROT', not ECLIPSE
			'multi': 'ECLIPSE',
			'MMR': 'ECLIPSE',
			'RR Lyrae': 'RRLYR_CEPHEID',
			'RRab': 'RRLYR_CEPHEID',
			'RRc': 'RRLYR_CEPHEID',
			'RRd': 'RRLYR_CEPHEID',
			'Cepheid': 'RRLYR_CEPHEID',
			'FM': 'RRLYR_CEPHEID',
			'1O': 'RRLYR_CEPHEID',
			'1O2O': 'RRLYR_CEPHEID',
			'FM1O': 'RRLYR_CEPHEID',
			'Type II': 'RRLYR_CEPHEID',
			'Anomaleous': 'RRLYR_CEPHEID',
			'SPB': 'GDOR_SPB',
			'dsct': 'DSCT_BCEP',
			'bumpy': 'GDOR_SPB',
			'gDor': 'GDOR_SPB',
			'bCep': 'DSCT_BCEP',
			#'roAp': 'RAPID', # Target will be ignored
			#'sdBV': 'RAPID', # Target will be ignored
			#'Flare': 'TRANSIENT', # Target will be ignored
			'Spots': 'CONTACT_ROT',
			'LPV': 'APERIODIC',
			'MIRA': 'APERIODIC',
			'SR': 'APERIODIC',
			'Constant': 'CONSTANT',
			'gDor+dSct hybrid': 'DSCT_BCEP;GDOR_SPB',
			'dSct+gDor hybrid': 'DSCT_BCEP;GDOR_SPB',
			'bCep+SPB hybrid': 'DSCT_BCEP;GDOR_SPB'
		}
	elif self.level == 'L2':
		translate = {
			'Solar-like': 'SOLARLIKE',
			'Transit': 'ECLIPSE',
			'Eclipse': 'ECLIPSE',
			'multi': 'ECLIPSE',
			'MMR': 'ECLIPSE',
			'RR Lyrae': 'RRLYR',
			'RRab': 'RRLYR',
			'RRc': 'RRLYR',
			'RRd': 'RRLYR',
			'Cepheid': 'CEPHEID',
			'FM': 'CEPHEID',
			'1O': 'CEPHEID',
			'1O2O': 'CEPHEID',
			'FM1O': 'CEPHEID',
			'Type II': 'CEPHEID',
			'Anomaleous': 'CEPHEID',
			'SPB': 'SPB',
			'dsct': 'DSCT',
			'bumpy': 'DSCT', # This is not right - Should we make a specific class for these?
			'gDor': 'GDOR',
			'bCep': 'BCEP',
			'roAp': 'ROAP',
			'sdBV': 'SDB',
			'Flare': 'FLARE',
			'Spots': 'SPOTS',
			'LPV': 'LPV',
			'MIRA': 'LPV',
			'SR': 'LPV',
			'Constant': 'CONSTANT',
			'gDor+dSct hybrid': 'GDOR;DSCT',
			'dSct+gDor hybrid': 'DSCT;GDOR',
			'bCep+SPB hybrid': 'BCEP;SPB'
		}

	translated_starclass = []
	for row in starlist:
		lbls = []
		for lbl in row['starclass'].split(';'):
			newlbl = translate.get(lbl.strip())
			if newlbl is not None:
				lbls.append(translate[lbl.strip()])

		# Remove duplicated labels:
		lookup = set()  # a temporary lookup set
		lbls = [x for x in lbls if x not in lookup and lookup.add(x) is None]

		translated_starclass.append(';'.join(lbls))
	starlist['starclass'] = translated_starclass

	# Remove targets where there is no stellar class defined:
	indx = (starlist['starclass'] != '')
	starlist = starlist[indx]

	return starlist

#--------------------------------------------------------------------------------------------------
class tdasim(TrainingSet):
	# Class constants:
	key = 'tdasim'

	def __init__(self, *args, **kwargs):

		self.input_folder = self.tset_datadir('https://tasoc.dk/pipeline/starclass_trainingsets/tdasim.zip')

		# Initialize parent
		# NOTE: We do this after setting the input_folder, as it depends on that being set:
		super().__init__(*args, **kwargs)

	def load_targets(self):
		return _load_targets(self)

#--------------------------------------------------------------------------------------------------
class tdasim_raw(TrainingSet):
	# Class constants:
	key = 'tdasim-raw'

	def __init__(self, *args, **kwargs):

		self.input_folder = self.tset_datadir('https://tasoc.dk/pipeline/starclass_trainingsets/tdasim-raw.zip')

		# Initialize parent
		# NOTE: We do this after setting the input_folder, as it depends on that being set:
		super().__init__(*args, **kwargs)

	def load_targets(self):
		return _load_targets(self)

#--------------------------------------------------------------------------------------------------
class tdasim_clean(TrainingSet):
	# Class constants:
	key = 'tdasim-clean'

	def __init__(self, *args, **kwargs):

		# Key for this training-set:
		self.input_folder = self.tset_datadir('https://tasoc.dk/pipeline/starclass_trainingsets/tdasim-clean.zip')

		# Initialize parent
		# NOTE: We do this after setting the input_folder, as it depends on that being set:
		super().__init__(*args, **kwargs)

	def load_targets(self):
		return _load_targets(self)
