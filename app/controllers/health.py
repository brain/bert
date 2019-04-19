# -*- coding: utf-8 -*-
from flask_restful import (
    Resource
)

from app.extensions import db

session = db.session


class HealthResource(Resource):
    def get(self):
        try:
            session.execute('SELECT 1')
        except Exception:
            return dict(message="No DB connection"), 500

        return dict(message="OK"), 200
