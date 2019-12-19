#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Notes for creating the TODO-file used in testing.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import sqlite3
import os.path
import shutil
from contextlib import closing

if __name__ == '__main__':

	input_todo = './todo.sqlite'
	#input_dir = '/aadc/tasoc/archive/S01_DR01/lightcurves-combined/'

	todo_file = os.path.abspath('../tests/input/todo.sqlite')
	#zippath = os.path.abspath('./corrections_tests_input.zip')

	print("Copying TODO-file...")
	shutil.copyfile(input_todo, todo_file)

	# Open the SQLite file:
	with closing(sqlite3.connect(todo_file)) as conn:
		conn.row_factory = sqlite3.Row
		cursor = conn.cursor()

		# Remove tables that are not needed
		print("Removing not needed tables...")
		cursor.execute("DROP TABLE IF EXISTS diagnostics;")
		cursor.execute("DROP TABLE IF EXISTS photometry_skipped;")
		cursor.execute("DROP TABLE IF EXISTS datavalidation;")
		cursor.execute("DROP TABLE IF EXISTS datavalidation_raw;")
		cursor.execute("DROP TABLE IF EXISTS starclass_diagnostics;")
		cursor.execute("DROP TABLE IF EXISTS starclass_results;")
		conn.commit()

		# Only keep targets from a few CCDs
		print("Deleting all targets not from specific CCDs...")
		cursor.execute("DELETE FROM todolist WHERE camera != 1 OR ccd IN (1,2,3);")
		conn.commit()

		# Clear other things:
		print("Cleaning up other stupid stuff...")
		cursor.execute("DELETE FROM diagnostics_corr WHERE diagnostics_corr.priority NOT IN (SELECT todolist.priority FROM todolist);")
		cursor.execute("DELETE FROM datavalidation_corr WHERE datavalidation_corr.priority NOT IN (SELECT todolist.priority FROM todolist);")
		conn.commit()

		# Create indicies
		print("Making sure indicies are there...")
		cursor.execute("CREATE INDEX IF NOT EXISTS datavalidation_corr_approved_idx ON datavalidation_corr (approved);")
		conn.commit()

		# Optimize tables
		print("Optimizing tables...")
		try:
			conn.isolation_level = None
			cursor.execute("VACUUM;")
			cursor.execute("ANALYZE;")
			cursor.execute("VACUUM;")
			conn.commit()
		finally:
			conn.isolation_level = ''

		"""
		# Crate the ZIP file and add all the files:
		# We do allow for ZIP64 extensions for large files - lets see if anyone complains
		with zipfile.ZipFile(zippath, 'w', zipfile.ZIP_STORED, True) as myzip:

			cursor.execute("SELECT todolist.priority,lightcurve FROM todolist INNER JOIN diagnostics ON diagnostics.priority=todolist.priority INNER JOIN datavalidation_raw ON todolist.priority=datavalidation_raw.priority WHERE status=1 AND datavalidation_raw.approved=1;")
			for row in tqdm(cursor.fetchall()):

				filepath = os.path.join(input_dir, row['lightcurve'])
				if not os.path.exists(filepath):
					raise FileNotFoundError("File not found: '" + filepath + "'")

				# Add the file to the ZIP archive:
				myzip.write(filepath, row['lightcurve'], zipfile.ZIP_STORED)

		cursor.close()
		"""