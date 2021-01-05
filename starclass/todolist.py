#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import logging
import os
import re
import fnmatch
import sqlite3
from contextlib import closing
from tqdm import tqdm
from .io import load_lightcurve

#--------------------------------------------------------------------------------------------------
def create_fake_todolist(input_folder, name='todo.sqlite', pattern=None,
	overwrite=False):
	"""
	Create todo-file by scanning directory for light curve files.

	Parameters:
		input_folder (str): Path to directory containing light curves to build todo-file from.
		name (str): Name of the todo-file which will be created in ``input_folder``.
		pattern (str): Pattern to use for searching for light curve files in ``input_folder``.
			The pattern must be a sting which can be interpreted by the ``fnmatch`` module.
			The default is to match all FITS files (including compressed files).
		overwrite (bool): Overwrite existing todo-file. Default is to not overwrite.

	Returns:
		str: Path to the generated todo-file.

	.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
	"""

	logger = logging.getLogger(__name__)
	tqdm_settings = {'disable': not logger.isEnabledFor(logging.INFO)}

	# Basic checks of the input:
	if not os.path.isdir(input_folder):
		raise NotADirectoryError(input_folder)
	if not name:
		raise ValueError("Invalid todo-file name")
	if not name.endswith('.sqlite'):
		name += '.sqlite'
	if pattern:
		file_pattern = fnmatch.translate(pattern)
	else:
		file_pattern = r'.*\.fits(\.gz)?$'

	logger.debug("Searching using RegEx pattern: %s", file_pattern)

	# Go through the input directory recursively and find all files which match the pattern:
	logger.info("Searching for files...")
	files = []
	regex = re.compile(file_pattern)
	for root, dirnames, filenames in os.walk(input_folder, followlinks=True):
		for filename in filenames:
			if regex.match(filename):
				files.append(os.path.join(root, filename))

	if not files:
		raise ValueError("No files were found")

	# Check path to the TODO-file, and delete if it already exists:
	todo_file = os.path.join(input_folder, name)
	if os.path.exists(todo_file):
		if overwrite:
			os.remove(todo_file)
		else:
			raise ValueError("Todo-file already exists")

	# Open the todo-file and create the records of the files in it:
	logger.info("Building todo-file...")
	try:
		with closing(sqlite3.connect(todo_file)) as conn:
			cursor = conn.cursor()

			todolist_structure(conn)

			for k, fpath in enumerate(tqdm(files, **tqdm_settings)):

				lightcurve = os.path.relpath(fpath, input_folder)

				lc = load_lightcurve(fpath)
				starid = lc.targetid

				todolist_insert(cursor,
					priority=k+1,
					starid=starid,
					lightcurve=lightcurve)

			conn.commit()
			cursor.close()
	except: # noqa: E722, pragma: no cover
		if os.path.exists(todo_file):
			os.remove(todo_file)
		raise

	# Return to path of the created TODO-file:
	return todo_file

#--------------------------------------------------------------------------------------------------
def todolist_structure(conn):
	"""
	Generate overall database structure for todo.sqlite.

	Parameters:
		conn (sqlite3.connection): Connection to SQLite file.

	.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
	"""

	cursor = conn.cursor()
	cursor.execute("PRAGMA foreign_keys=ON;")

	cursor.execute("""CREATE TABLE todolist (
		priority INTEGER PRIMARY KEY NOT NULL,
		starid BIGINT NOT NULL,
		datasource TEXT NOT NULL DEFAULT 'ffi',
		camera INTEGER NOT NULL,
		ccd INTEGER NOT NULL,
		method TEXT DEFAULT NULL,
		tmag REAL,
		status INTEGER DEFAULT NULL,
		corr_status INTEGER DEFAULT NULL,
		cbv_area INTEGER NOT NULL
	);""")
	cursor.execute("CREATE INDEX status_idx ON todolist (status);")
	cursor.execute("CREATE INDEX corr_status_idx ON todolist (corr_status);")
	cursor.execute("CREATE INDEX starid_idx ON todolist (starid);")

	cursor.execute("""CREATE TABLE diagnostics_corr (
		priority INTEGER PRIMARY KEY NOT NULL,
		lightcurve TEXT,
		elaptime REAL,
		worker_wait_time REAL,
		variance REAL,
		rms_hour REAL,
		ptp REAL,
		errors TEXT,
		FOREIGN KEY (priority) REFERENCES todolist(priority) ON DELETE CASCADE ON UPDATE CASCADE
	);""")

	cursor.execute("""CREATE TABLE datavalidation_corr (
		priority INTEGER PRIMARY KEY NOT NULL,
		approved BOOLEAN NOT NULL,
		dataval INTEGER NOT NULL,
		FOREIGN KEY (priority) REFERENCES todolist(priority) ON DELETE CASCADE ON UPDATE CASCADE
	);""")
	cursor.execute("CREATE INDEX datavalidation_corr_approved_idx ON datavalidation_corr (approved);")

	conn.commit()

#--------------------------------------------------------------------------------------------------
def todolist_insert(cursor, priority=None, lightcurve=None, starid=None,
	tmag=None, datasource='ffi', variance=None, rms_hour=None, ptp=None, elaptime=None):
	"""
	Insert an entry in the todo.sqlite file.

	Parameters:
		cursor (sqlite3.Cursor): Cursor in SQLite file.
		priority (int):
		lightcurve (str):
		starid (int):
		tmag (float): TESS Magnitude.
		datasource (str):
		variance (float):
		rms_hour (float):
		ptp (float):
		elaptime (float):

	.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
	"""

	if priority is None:
		raise ValueError("PRIORITY is required.")
	if lightcurve is None:
		raise ValueError("LIGHTCURVE is required.")
	if starid is None:
		starid = priority
	if datasource not in ('ffi', 'tpf'):
		raise ValueError("DATASOURCE should be FFI of TPF.")
	if tmag is None:
		tmag = -99

	cursor.execute("INSERT INTO todolist (priority,starid,tmag,datasource,status,corr_status,camera,ccd,cbv_area) VALUES (?,?,?,?,1,1,1,1,111);", (
		priority,
		starid,
		tmag,
		datasource
	))
	cursor.execute("INSERT INTO diagnostics_corr (priority,lightcurve,elaptime,variance,rms_hour,ptp) VALUES (?,?,?,?,?,?);", (
		priority,
		lightcurve.replace('\\', '/'),
		elaptime,
		variance,
		rms_hour,
		ptp
	))
	cursor.execute("INSERT INTO datavalidation_corr (priority,approved,dataval) VALUES (?,1,0);", (priority,))
