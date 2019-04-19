# -*- coding: utf-8 -*-
import pytest
from helpers.app_helper import BaseTest


class TestHeathEndpoints(BaseTest):
    @pytest.mark.parametrize("test_text,expected_result", [
        ("", {'message': 'OK'}),
    ])
    def test_health_endpoint(self, test_text, expected_result):
        res = self.client.get('/')
        assert res.status_code == 200
        assert res.get_json() == expected_result

    @pytest.mark.parametrize("test_text,expected_result", [
        ("", {'message': 'OK'}),
    ])
    def test_health_endpoint_(self, test_text, expected_result):
        res = self.client.get('/health')
        assert res.status_code == 200
        assert res.get_json() == expected_result
