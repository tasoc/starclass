#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests of run_classifier command line interface.
There are additional tests of this under "test_classifiers.py".

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import pytest
import tempfile
import os.path
from conftest import capture_run_cli

#--------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('tf', [-0.1, 1.0, 1.1])
def test_run_training_invalid_testfraction(tf):

	dd = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'starclass', 'data', 'L1'))
	with tempfile.TemporaryDirectory(dir=dd, prefix='testing-') as tmpdir:
		output_dir = os.path.basename(tmpdir)

		out, err, exitcode = capture_run_cli('run_training.py', [
			'--classifier=meta',
			'--trainingset=testing',
			'--level=L1',
			f'--testfraction={tf:f}',
			f'--output={output_dir}'
		])
		assert exitcode == 2
		assert 'error: Testfraction must be between 0 and 1' in err

#--------------------------------------------------------------------------------------------------
if __name__ == '__main__':
	pytest.main([__file__])
