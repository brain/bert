# -*- coding: utf-8 -*-
import os
from app.configs.common import *  # noqa: F401, F403

ENVIRONMENT = os.getenv('ENVIRONMENT', "production")
