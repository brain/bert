# -*- coding: utf-8 -*-
import os


APP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
PROJECT_ROOT = os.path.abspath(os.path.join(APP_DIR, os.pardir))
DB_SERVER = 'mysql://{}:{}@{}'.format(
    os.getenv('MYSQL_USER'),
    os.getenv('MYSQL_PASSWORD'),
    os.getenv('MYSQL_HOST')
)
SQLALCHEMY_TRACK_MODIFICATIONS = False
SQLALCHEMY_DATABASE_URI = '{}/{}?charset=utf8mb4&binary_prefix=1'.format(
    DB_SERVER,
    os.getenv('MYSQL_DATABASE')
)
SQLALCHEMY_POOL_RECYCLE = os.getenv('SQLALCHEMY_POOL_RECYCLE', 300)

DEBUG = False
