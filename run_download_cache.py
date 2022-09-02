#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Command-line interface downloading auxillary data.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import argparse
import logging
import starclass

#--------------------------------------------------------------------------------------------------
def main():
	# Parse command line arguments:
	parser = argparse.ArgumentParser(description='Download all auxillary data for pipeline.')
	parser.add_argument('-d', '--debug', help='Print debug messages.', action='store_true')
	parser.add_argument('-q', '--quiet', help='Only report warnings and errors.', action='store_true')
	parser.add_argument('--all', help='Download all training sets.', action='store_true')
	parser.add_argument('-t', '--trainingset',
		default=None,
		choices=starclass.trainingset_list,
		action='append',
		metavar='{TSET}',
		help='Download this training-set. Choises are ' + ", ".join(starclass.trainingset_list) + '.')
	args = parser.parse_args()

	# Set logging level:
	logging_level = logging.INFO
	if args.quiet:
		logging_level = logging.WARNING
	elif args.debug:
		logging_level = logging.DEBUG

	# Setup logging:
	formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
	console = logging.StreamHandler()
	console.setFormatter(formatter)
	logger = logging.getLogger(__name__)
	logger.addHandler(console)
	logger.setLevel(logging_level)
	logger_parent = logging.getLogger('starclass')
	logger_parent.addHandler(console)
	logger_parent.setLevel(logging_level)

	# Make sure we have turned plotting to non-interactive:
	starclass.plots.plots_noninteractive()

	# Select wich training-sets to download:
	trainingsets = 'all' if args.all else args.trainingset

	# Download all data:
	starclass.download_cache(trainingsets=trainingsets)

#--------------------------------------------------------------------------------------------------
if __name__ == '__main__':
	main()
