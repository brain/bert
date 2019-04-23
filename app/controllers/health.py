# -*- coding: utf-8 -*-
from flask_restful import (
    Resource
)


class HealthResource(Resource):
    def get(self):
        return dict(message="OK"), 200
