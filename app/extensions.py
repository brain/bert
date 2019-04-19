# -*- coding: utf-8 -*-
"""Extensions module. Each extension is initialized in the appfactory.py"""
from flask_restful import Api
from flask_sqlalchemy import SQLAlchemy


api = Api()
db = SQLAlchemy()
