#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Command-line interface for creating todo-file by scanning directory for light curve files..

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import argparse
import logging
import starclass

#--------------------------------------------------------------------------------------------------
def main():
	# Parse command line arguments:
	parser = argparse.ArgumentParser(description='')
	parser.add_argument('-d', '--debug', help='Print debug messages.', action='store_true')
	parser.add_argument('-q', '--quiet', help='Only report warnings and errors.', action='store_true')
	parser.add_argument('-o', '--overwrite', help='Overwrite existing todo-file.', action='store_true')
	parser.add_argument('--pattern', type=str, default=None, help='File pattern to search for light curves with.')
	parser.add_argument('--name', type=str, default='todo.sqlite', help='Name of todo-file to create.')
	parser.add_argument('input_folder', type=str, help='Directory containing light curves to build todo-file from.')
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

	# Download all data:
	starclass.todolist.create_fake_todolist(args.input_folder,
		name=args.name,
		pattern=args.pattern,
		overwrite=args.overwrite)

#--------------------------------------------------------------------------------------------------
if __name__ == '__main__':
	main()
