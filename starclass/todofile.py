#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Todo file functions.

.. codeauthor::
"""
import numpy as np
import logging
import sqlite3
from tqdm import tqdm
from contextlib import closing
import os
from bottleneck import nanmedian, nanvar
from . import utilities
from . import BaseClassifier

#--------------------------------------------------------------------------------------------------
def generate_todolist_newdata(input_folder):
	"""
	Generate todo.sqlite file in training set directory.

	.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
	"""
	logger = logging.getLogger(__name__)

	starlist = [f for f in os.listdir(input_folder + '/stars') if os.path.isfile(os.path.join(input_folder + '/stars', f)) and not f.startswith(".")]

	sqlite_file = os.path.join(input_folder, 'todo.sqlite')
	with closing(sqlite3.connect(sqlite_file)) as conn:
		conn.row_factory = sqlite3.Row
		cursor = conn.cursor()

		# Create the basic file structure of a TODO-list:
		generate_todolist_structure(conn)

		logger.info("Step 3: Reading file and extracting information...")
		pri = 0

		for star in tqdm(starlist, total=len(starlist)):
			# Get starid:
			starname, file_extension = os.path.splitext(star)
			starid = int(starname)
			#starname = '{0:09d}'.format(starid)

			# Path to lightcurve:
			lightcurve = 'stars/' + starname + file_extension

			pri += 1
			generate_todolist_insert(cursor, input_folder=input_folder,
				priority=pri,
				starid=starid,
				lightcurve=lightcurve,
				datasource='ffi')

		conn.commit()
		cursor.close()

	logger.info("Todo file successfully built.")

#----------------------------------------------------------------------------------------------
def generate_todolist_structure(conn):
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

#----------------------------------------------------------------------------------------------
def generate_todolist_insert(cursor, input_folder=None, tset=None, priority=None, lightcurve=None, starid=None,
    tmag=None, datasource=None, variance=None, rms_hour=None, ptp=None):
    """
    Insert an entry in the todo.sqlite file.

    Parameters:
        cursor (sqlite3.Cursor): Cursor in SQLite file.
        priority (int):
        lightcurve (str):
        starid (int, optional):
        tmag (float, optional): TESS Magnitude.
        datasource (str, optional):
        variance (float, optional):
        rms_hour (float, optional):
        ptp (float, optional):

    .. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
    """

    if priority is None:
        raise ValueError("PRIORITY is required.")
    if lightcurve is None:
        raise ValueError("LIGHTCURVE is required.")
    if starid is None:
        starid = priority

    if not input_folder and not tset:
        raise ValueError("An input folder or training set is required")
    if not input_folder:
        input_folder = tset.input_folder

    # Try to load the lightcurve using the BaseClassifier method.
    # This will ensure that the lightcurve can actually be read by the system.
    if not all([datasource, variance, rms_hour, ptp]):
        with BaseClassifier(tset=tset, features_cache=None) as bc:
            # Most of the input is None, simply to silence warnings
            fake_task = {
                'priority': priority,
                'starid': starid,
                'tmag': None,
                'variance': None,
                'rms_hour': None,
                'ptp': None,
                'other_classifiers': None
            }
            features = bc.load_star(fake_task, os.path.join(input_folder, lightcurve))
            lc = features['lightcurve']

    elaptime = np.random.normal(3.14, 0.5)
    if tmag is None:
        tmag = -99
    if variance is None:
        variance = nanvar(lc.flux, ddof=1)
    if rms_hour is None:
        rms_hour = utilities.rms_timescale(lc)
    if ptp is None:
        ptp = nanmedian(np.abs(np.diff(lc.flux)))

    if datasource is None:
        if (lc.time[1] - lc.time[0])*86400 > 1000:
            datasource = 'ffi'
        else:
            datasource = 'tpf'

    #camera = 1
    #if ecllat < 6+24:
    #	camera = 1
    #elif ecllat < 6+2*24:
    #	camera = 2
    #elif ecllat < 6+3*24:
    #	camera = 3
    #else:
    #	camera = 4

    cursor.execute("INSERT INTO todolist (priority,starid,tmag,datasource,status,corr_status,camera,ccd,cbv_area) VALUES (?,?,?,?,1,1,1,1,111);", (
        priority,
        starid,
        tmag,
        datasource
    ))
    cursor.execute("INSERT INTO diagnostics_corr (priority,lightcurve,elaptime,variance,rms_hour,ptp) VALUES (?,?,?,?,?,?);", (
        priority,
        lightcurve,
        elaptime,
        variance,
        rms_hour,
        ptp
    ))
    cursor.execute("INSERT INTO datavalidation_corr (priority,approved,dataval) VALUES (?,1,0);", (priority,))
