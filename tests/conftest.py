#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pytest fixture to create temp copy of input data, shared across all tests.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import pytest
import os.path
import tempfile
import shutil

#--------------------------------------------------------------------------------------------------
@pytest.fixture(scope='session')
def SHARED_INPUT_DIR():
	"""
	Pytest fixture to create temp copy of input data, shared across all tests.
	"""
	INPUT_DIR = os.path.join(os.path.dirname(__file__), 'input')
	with tempfile.TemporaryDirectory(prefix='pytest-shared-input-dir-') as my_tmpdir:
		tmp = os.path.join(my_tmpdir, 'input')
		shutil.copytree(INPUT_DIR, tmp)
		yield tmp

#--------------------------------------------------------------------------------------------------
@pytest.fixture(scope='function')
def PRIVATE_INPUT_DIR():
	"""
	Pytest fixture to create temp copy of input data, shared across all tests.
	"""
	INPUT_DIR = os.path.join(os.path.dirname(__file__), 'input')
	with tempfile.TemporaryDirectory(prefix='pytest-private-input-dir-') as my_tmpdir:
		tmp = os.path.join(my_tmpdir, 'input')
		shutil.copytree(INPUT_DIR, tmp)
		yield tmp

#--------------------------------------------------------------------------------------------------
@pytest.fixture(scope='function')
def PRIVATE_TODO_FILE():
	"""
	Pytest fixture to create temp copy of input todo-file, private to the individual test.
	"""
	TODO_FILE = os.path.join(os.path.dirname(__file__), 'input', 'todo.sqlite')
	with tempfile.TemporaryDirectory(prefix='pytest-private-todo-') as my_tmpdir:
		tmp = os.path.join(my_tmpdir, 'todo.sqlite')
		shutil.copy2(TODO_FILE, tmp)
		yield tmp
