# -*- coding: utf-8 -*-
"""Application configuration."""
import os
from importlib import import_module
from os.path import abspath, dirname, join


""" Global config """
ENVIRONMENT = os.getenv('ENVIRONMENT', 'development')
c = import_module("app.configs.{}".format(ENVIRONMENT))
c.SECRET_KEY = os.environ.get('SECRET_KEY', 'flasksecret')
c.ENVIRONMENT = ENVIRONMENT


""" Dir paths """
APP_DIR = dirname(dirname(abspath(__file__)))
TEMPLATES_DIR = join(APP_DIR, "templates")


# Get all attributes by dictionary comprehension
config = {
    conf: c.__getattribute__(conf) for conf in dir(c) if '__' not in conf
}

# JWT AUTH
JWT = {
    "SECRET": os.getenv("JWT_SECRET") or "BRAIN-INTERNAL-APIS-JWT-SECRET",
    "ALGO": 'HS256',
}
