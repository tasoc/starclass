#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging
import starclass

#--------------------------------------------------------------------------------------------------
if __name__ == '__main__':

	# Parse command line arguments:
	parser = argparse.ArgumentParser(description='')
	parser.add_argument('-d', '--debug', help='Print debug messages.', action='store_true')
	parser.add_argument('-q', '--quiet', help='Only report warnings and errors.', action='store_true')
	parser.add_argument('-o', '--overwrite', action='store_true')
	parser.add_argument('--pattern', type=str, default=None)
	parser.add_argument('--output', type=str, default='todo.sqlite')
	parser.add_argument('input_folder', type=str)
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
		output_todo=args.output,
		file_pattern=args.pattern,
		overwrite=args.overwrite)
