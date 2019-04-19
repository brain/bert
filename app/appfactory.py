# -*- coding: utf-8 -*-
"""The app module, containing the app factory function."""
import logging
import traceback
from inspect import isclass
from importlib import import_module
from flask import Flask

from app.extensions import db
from app.configs import (
    config,
    TEMPLATES_DIR,
)


def create_app():
    """An application factory, as explained here:
    http://flask.pocoo.org/docs/patterns/appfactories/.

    :param config_object: The configuration object to use.
    """
    app = Flask(__name__.split('.')[0], template_folder=TEMPLATES_DIR)
    app.config.update(config)
    register_urls()
    register_extensions(app)
    register_shellcontext(app)
    register_handlers(app)
    return app


def register_urls():
    import app.urls # noqa


def register_extensions(app):
    """Register Flask extensions."""
    ext_module = import_module('app.extensions')

    for obj_s in [attr for attr in dir(ext_module) if '__' not in attr]:
        obj = getattr(ext_module, obj_s)
        init_app = getattr(obj, 'init_app', None)

        if not isclass(obj) and callable(init_app):
            init_app(app)


def register_shellcontext(app):
    """Register shell context objects."""
    def shell_context():
        """Shell context objects."""
        return {
            'session': db.session,
        }

    app.shell_context_processor(shell_context)


def register_handlers(app):
    """Register handlers """

    def close_db_session(exception=None):
        if db.session:
            logging.debug("Closing db session")
            try:
                db.session.close()
            except Exception as e:
                logging.warn("Failed to close db session - err:{}".format(e))
                traceback.print_exc

    def close_db_session_on_error(exception):
        close_db_session(exception=exception)
        raise exception

    app.teardown_request(close_db_session)
    app.errorhandler(Exception)(close_db_session_on_error)
