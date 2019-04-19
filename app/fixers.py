# -*- coding: utf-8 -*-
from werkzeug.contrib.fixers import ProxyFix


class BrainProxyFix(ProxyFix):
    """ This class allows this backend service to have knowledge of the
        brus proxy's url scheme, host, and prefix path for this service.
        The prefix path for this service is specified in the brus project.
    """
    def __call__(self, environ, start_response):
        getter = environ.get
        forwarded_proto = getter('HTTP_X_FORWARDED_PROTO', '')
        forwarded_for = getter('HTTP_X_FORWARDED_FOR', '').split(',')
        forwarded_host = getter('HTTP_X_FORWARDED_HOST', '')
        forwarded_path = getter('HTTP_X_FORWARDED_PATH', '')
        environ.update({
            'werkzeug.proxy_fix.orig_wsgi_url_scheme':  getter('wsgi.url_scheme'), # noqa
            'werkzeug.proxy_fix.orig_remote_addr':      getter('REMOTE_ADDR'), # noqa
            'werkzeug.proxy_fix.orig_http_host':        getter('HTTP_HOST') # noqa
        })
        forwarded_for = [x for x in [x.strip() for x in forwarded_for] if x]
        remote_addr = self.get_remote_addr(forwarded_for)
        if remote_addr is not None:
            environ['REMOTE_ADDR'] = remote_addr
        if forwarded_host:
            environ['HTTP_HOST'] = forwarded_host
        if forwarded_proto:
            environ['wsgi.url_scheme'] = forwarded_proto
        if forwarded_path:
            environ['SCRIPT_NAME'] = forwarded_path
        return self.app(environ, start_response)
