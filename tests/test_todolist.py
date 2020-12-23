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
from starclass.todolist import todolist_structure, todolist_insert

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

			lightcurve = os.path.join(SHARED_INPUT_DIR, 'tess00029281992-s01-c1800-dr01-v04-tasoc-cbv_lc.fits.gz')
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
