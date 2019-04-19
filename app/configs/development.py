# -*- coding: utf-8 -*-
import os
from app.configs.common import *  # noqa: F401, F403

DEBUG = True
ENVIRONMENT = os.getenv('ENVIRONMENT', "development")
