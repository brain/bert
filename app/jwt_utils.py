#
#
#

import jwt
import datetime
from app.configs import JWT


class Jwt(object):
    jwt = jwt

    @classmethod
    def encode(cls, payload):
        payload.update({
            'exp': datetime.datetime.utcnow() + datetime.timedelta(seconds=600)
        })
        return jwt.encode(payload, JWT["SECRET"], algorithm=JWT["ALGO"]).decode('utf-8')

    @classmethod
    def decode(cls, jwt_token):
        return jwt.decode(jwt_token, JWT["SECRET"], algorithms=JWT["ALGO"])
