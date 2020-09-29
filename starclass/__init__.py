#!/usr/bin/env python
# -*- coding: utf-8 -*-
# flake8: noqa

from .StellarClasses import StellarClasses, StellarClassesLevel2
from .BaseClassifier import BaseClassifier, STATUS
from .taskmanager import TaskManager
from .RFGCClassifier import RFGCClassifier
from .SLOSH import SLOSHClassifier
from .XGBClassifier import XGBClassifier
from .MetaClassifier import MetaClassifier
from .utilities import PICKLE_DEFAULT_PROTOCOL
from . import training_sets
from .download_cache import download_cache
from .version import get_version

__version__ = get_version()
