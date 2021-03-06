#!/usr/bin/env python
# -*- coding: utf-8 -*-
# flake8: noqa

from .StellarClasses import StellarClassesLevel1, StellarClassesLevel2
from .BaseClassifier import BaseClassifier, STATUS
from .taskmanager import TaskManager
from .RFGCClassifier import RFGCClassifier
from .SLOSH import SLOSHClassifier
from .XGBClassifier import XGBClassifier
from .SortingHatClassifier import SortingHatClassifier
from .MetaClassifier import MetaClassifier
from . import training_sets
from .download_cache import download_cache
from .convenience import get_classifier, get_trainingset, trainingset_available
from .constants import classifier_list, trainingset_list
from .version import get_version

__version__ = get_version()
