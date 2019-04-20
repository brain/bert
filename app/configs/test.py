# -*- coding: utf-8 -*-
import os
from app.configs.common import *  # noqa: F401, F403

SQLALCHEMY_DATABASE_URI = '{}/{}?charset=utf8mb4&binary_prefix=1'.format(
    DB_SERVER,  # noqa: F405
    os.getenv('MYSQL_TEST_DATABASE')
)

DEBUG = True
ENVIRONMENT = os.getenv('ENVIRONMENT', "test")
