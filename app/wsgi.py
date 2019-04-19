# -*- coding: utf-8 -*-
from app.appfactory import create_app
from app.fixers import BrainProxyFix

app = create_app()
app.wsgi_app = BrainProxyFix(app.wsgi_app)
