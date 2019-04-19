#
#
#
import os
import json
import urllib
from jinja2 import Template


class APIDoc(object):

    category = dict()
    current = None

    @classmethod
    def render(cls):
        tmpl_file = '/brain/src/docs/index.html.md.tmpl'
        render_file = "/brain/src/docs/index.html.md"
        if os.getenv("RENDER_DOCS", "").lower() == "true":
            with open(tmpl_file, 'r') as tmpl:
                template = Template(tmpl.read())
                with open(render_file, 'w') as f:
                    print(cls.category)
                    f.write(template.render(category=cls.category, verbs=["POST", "PATCH"]))

    @classmethod
    def add_category(cls, case):
        cls.current = case.__name__
        if cls.current.endswith("Test"):
            cls.current = cls.current[:-4]
        if cls.current.startswith("Test"):
            cls.current = cls.current[4:]
        cls.category.setdefault(cls.current, dict())
        return cls

    @classmethod
    def apply(cls, client, *args, type=None, **kwargs):
        '''
            TestHealthCheck
                test_health
                    /health
                        req
                        res
                        verb
                        description
        '''
        verb = type
        method_name = client._test_method_name.split(".")[-1]
        if method_name.startswith("test_"):
            method_name = " ".join(method_name[5:].split("_")).title()
        _d = cls.category[cls.current].setdefault(method_name, dict())
        url_path, *rest = args
        content = kwargs.get('response').get_data()
        if content is None:
            content = kwargs.get('response')
        if kwargs.get("query_string"):
            url_path = "{}?{}".format(
                url_path,
                urllib.parse.urlencode(kwargs.get("query_string"))
            )

        _d.setdefault(url_path, []).append(
            dict(
                req=cls.pretty_json(kwargs.get('json') or ''),
                res=cls.pretty_json(content),
                description="TODO: description",
                verb=verb
            ))

    @classmethod
    def pretty_json(cls, content):
        try:
            if type(content) is bytes:
                content = str(content, "utf-8")
            if type(content) is str:
                content = json.loads(content)
            return json.dumps(content, indent=2, sort_keys=True)
        except Exception as ex:
            if os.getenv("RENDER_DOCS", "").lower() == "true":
                print(ex)
        return content
