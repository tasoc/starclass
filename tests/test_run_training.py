#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests of run_classifier command line interface.
There are additional tests of this under "test_classifiers.py".

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import pytest
import tempfile
from conftest import capture_run_cli

#--------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('tf', [-0.1, 1.0, 1.1])
def test_run_training_invalid_testfraction(tf):

	with tempfile.TemporaryDirectory(prefix='testing-') as tmpdir:

		out, err, exitcode = capture_run_cli('run_training.py', [
			'--classifier=meta',
			'--trainingset=testing',
			'--level=L1',
			f'--testfraction={tf:f}',
			'--output=' + tmpdir
		])
		assert exitcode == 2
		assert 'error: Testfraction must be between 0 and 1' in err

#--------------------------------------------------------------------------------------------------
if __name__ == '__main__':
	pytest.main([__file__])
