# -*- coding: utf-8 -*-
from app.extensions import api
from app.controllers import (
    health,
)


# Health Routes
api.add_resource(health.HealthResource, '/', '/health')
