#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Testing for TaskMaster memory loading.

Can be used with line_profiler or similar to test execution time.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import numpy as np
import sys
import os
import timeit
import shutil
import tempfile
if sys.path[0] != os.path.abspath('..'):
	sys.path.insert(0, os.path.abspath('..'))
import starclass

#--------------------------------------------------------------------------------------------------
if __name__ == '__main__':

	original_todo_file = os.path.abspath('../tests/input/todo.sqlite')

	tsetclass = starclass.get_trainingset('keplerq9v3')
	tset = tsetclass(level='L1', linfit=False)

	features_names = ['column' + str(k) for k in range(10)]

	num = 5000

	for in_memory in (True, False):
		with tempfile.TemporaryDirectory() as tmpdir:
			todo_file = os.path.join(tmpdir, 'test.sqlite')
			shutil.copy(original_todo_file, todo_file)

			print('-'*30)
			print(f"Running memory={in_memory}")
			with starclass.TaskManager(todo_file, overwrite=True, load_in_memory=in_memory, classes=tset.StellarClasses) as tm:

				t = timeit.timeit("tm.get_task(classifier='rfgc')", globals=globals(), number=num)
				print(f"get_task: {t:.6f} s, {t/num:.6e} s/it")

				def func():
					task = tm.get_task(classifier='rfgc', chunk=2, change_classifier=False)
					#print(task)
					if task is None:
						return
					tm.start_task(task)
					res = []
					for t in task:
						r = t.copy()
						r.update({
							'starclass_results': {tm.StellarClasses.SOLARLIKE: 0.9, tm.StellarClasses.CONSTANT: 0.1},
							'features_common': dict(zip(features_names, np.random.randn(10))),
							'features': dict(zip(features_names, np.random.randn(10))),
							'status': starclass.STATUS.OK,
							'elaptime': np.random.randn()
						})
						res.append(r)
					tm.save_results(res)

				t = timeit.timeit("func()", globals=globals(), number=num)
				print(f"get+save: {t:.6f} s, {t/num:.6e} s/it")

	print('-'*30)
