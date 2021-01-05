#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests of todolist generation.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import pytest
import os.path
import tempfile
import sqlite3
from contextlib import closing
import conftest # noqa: F401
from starclass.todolist import todolist_structure, todolist_insert, create_fake_todolist

#--------------------------------------------------------------------------------------------------
def test_todolist_insert(SHARED_INPUT_DIR):

	with tempfile.TemporaryDirectory(prefix='pytest-private-tsets-') as tmpdir:
		with closing(sqlite3.connect(os.path.join(tmpdir, 'todo.sqlite'))) as conn:
			conn.row_factory = sqlite3.Row
			cursor = conn.cursor()

			todolist_structure(conn)

			with pytest.raises(ValueError):
				todolist_insert(cursor, priority=None)

			with pytest.raises(ValueError):
				todolist_insert(cursor, priority=1, lightcurve=None)

			lightcurve = 'tess00029281992-s01-c1800-dr01-v04-tasoc-cbv_lc.fits.gz'
			todolist_insert(cursor,
				priority=2187,
				starid=12345678,
				tmag=15.6,
				lightcurve=lightcurve,
				datasource='tpf',
				variance=3.14,
				rms_hour=2.71,
				ptp=42.0)

			# TODOLIST table:
			cursor.execute("SELECT * FROM todolist WHERE priority=2187;")
			row = cursor.fetchone()
			assert row['priority'] == 2187
			assert row['starid'] == 12345678
			assert row['tmag'] == 15.6
			assert row['datasource'] == 'tpf'
			assert row['camera'] == 1 # These are constant!
			assert row['ccd'] == 1 # These are constant!
			assert row['cbv_area'] == 111 # These are constant!
			assert row['status'] == 1 # These are constant!
			assert row['corr_status'] == 1 # These are constant!

			# DIAGNOSTICS_CORR table:
			cursor.execute("SELECT * FROM diagnostics_corr WHERE priority=2187;")
			row = cursor.fetchone()
			assert row['lightcurve'] == lightcurve
			assert row['variance'] == 3.14
			assert row['rms_hour'] == 2.71
			assert row['ptp'] == 42.0

			# DATAVALIDATION_CORR table:
			cursor.execute("SELECT * FROM datavalidation_corr WHERE priority=2187;")
			row = cursor.fetchone()
			assert row['approved'] == 1 # These are constant!
			assert row['dataval'] == 0 # These are constant!

			todolist_insert(cursor,
				priority=2188,
				lightcurve=lightcurve)

			# TODOLIST table:
			cursor.execute("SELECT * FROM todolist WHERE priority=2188;")
			row = cursor.fetchone()
			assert row['priority'] == 2188
			assert row['starid'] == 2188 # When not provided, will used priority
			assert row['tmag'] == -99
			assert row['datasource'] == 'ffi'

			# DIAGNOSTICS_CORR table:
			cursor.execute("SELECT * FROM diagnostics_corr WHERE priority=2188;")
			row = cursor.fetchone()
			assert row['lightcurve'] == lightcurve
			assert row['variance'] is None
			assert row['rms_hour'] is None
			assert row['ptp'] is None

			# DATAVALIDATION_CORR table:
			cursor.execute("SELECT * FROM datavalidation_corr WHERE priority=2188;")
			row = cursor.fetchone()
			assert row['approved'] == 1 # These are constant!
			assert row['dataval'] == 0 # These are constant!

#--------------------------------------------------------------------------------------------------
def test_todolist_create(PRIVATE_INPUT_DIR):

	#
	input_folder = os.path.join(PRIVATE_INPUT_DIR, 'create_todolist')
	expected_file = os.path.join(input_folder, 'todo.sqlite')
	assert not os.path.exists(expected_file)

	todo_file = create_fake_todolist(input_folder)

	# todo-file should now exist:
	assert todo_file == expected_file
	assert os.path.isfile(todo_file)

	# Do a deep inspection of the todo-file:
	with closing(sqlite3.connect('file:' + todo_file + '?mode=ro', uri=True)) as conn:
		conn.row_factory = sqlite3.Row
		cursor = conn.cursor()

		cursor.execute("SELECT COUNT(*) FROM todolist;")
		assert cursor.fetchone()[0] == 2

		cursor.execute("SELECT COUNT(*) FROM diagnostics_corr;")
		assert cursor.fetchone()[0] == 2

		cursor.execute("SELECT COUNT(*) FROM datavalidation_corr;")
		assert cursor.fetchone()[0] == 2

		cursor.execute("SELECT lightcurve FROM diagnostics_corr;")
		for row in cursor:
			assert os.path.isfile(os.path.join(input_folder, row['lightcurve']))

	# Running it again, without overwrite, should raise error:
	with pytest.raises(ValueError) as e:
		create_fake_todolist(input_folder)
	assert str(e.value) == 'Todo-file already exists'

	# Running it again with overwrite should suceed:
	create_fake_todolist(input_folder, overwrite=True)

#--------------------------------------------------------------------------------------------------
def test_todolist_pattern(PRIVATE_INPUT_DIR):

	input_folder = os.path.join(PRIVATE_INPUT_DIR, 'create_todolist')

	# Using a pattern which will not match any targets should give an error:
	with pytest.raises(ValueError) as e:
		create_fake_todolist(input_folder, output_todo='todo-nomatch', pattern='*.txt')
	assert str(e.value) == 'No files were found'

	# Use pattern which only has a single match:
	todo_file = create_fake_todolist(input_folder, output_todo='todo-single', pattern='tess*.fits.gz')

	# Check that there is now only one target in the final list:
	with closing(sqlite3.connect('file:' + todo_file + '?mode=ro', uri=True)) as conn:
		cursor = conn.cursor()
		cursor.execute("SELECT COUNT(*) FROM todolist;")
		assert cursor.fetchone()[0] == 1

#--------------------------------------------------------------------------------------------------
if __name__ == '__main__':
	pytest.main([__file__])
