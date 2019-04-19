# -*- coding: utf-8 -*-
import json
from flask.testing import FlaskClient
from flask.wrappers import Response

from app.appfactory import create_app
from docs.docs import APIDoc


class JsonApiResponse(Response):
    """
    Wrapper response class for JSON API test
    Since master branch in github Flask respository has JSON methods,
    this should be a temporary workaround.
    """
    def get_json(self):
        """Return dictionary data by parsing response data"""
        return json.loads(self.data.decode('utf-8'))


app = create_app()


class APIClientBase(FlaskClient):
    def open(self, *args, **kwargs):
        if 'json' in kwargs:
            kwargs['data'] = json.dumps(kwargs.pop('json'))
            kwargs['content_type'] = 'application/json'
        return super(APIClientBase, self).open(*args, **kwargs)

    def get(self, *args, **kwargs):
        response = super(APIClientBase, self).get(*args, **kwargs)
        APIDoc.apply(self, *args, type='GET', response=response, **kwargs)
        return response

    def post(self, *args, **kwargs):
        response = super(APIClientBase, self).post(*args, **kwargs)
        APIDoc.apply(self, *args, type='POST', response=response, **kwargs)
        return response

    def patch(self, *args, **kwargs):
        response = super(APIClientBase, self).post(*args, **kwargs)
        APIDoc.apply(self, *args, type='PATCH', response=response, **kwargs)
        return response


class BaseTest(object):

    @classmethod
    def setup_class(cls):
        APIDoc.add_category(cls)

    def setup_method(self, method):
        self.client = APIClientBase(
            app,
            use_cookies=True,
            response_wrapper=JsonApiResponse
        )
        self.client._test_method_name = method.__name__

    def teardown_method(self):
        pass

    @classmethod
    def teardown_class(cls):
        APIDoc.render()
