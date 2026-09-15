from __future__ import annotations

import base64
import json
from flask import Flask, request, Response
from play.lambda_handler import handle_wsgi


def test_v2_adapter_handles_query_unicode_cookies_and_base64_body():
    app = Flask(__name__)
    @app.post('/echo/<value>')
    def echo(value):
        response = Response(json.dumps({'path': value, 'q': request.args['q'], 'body': request.json,
                                        'cookies': dict(request.cookies), 'scheme': request.scheme}), mimetype='application/json')
        response.set_cookie('first', 'a')
        response.set_cookie('second', 'b')
        return response
    event = {'version':'2.0', 'rawPath':'/echo/caf%C3%A9', 'rawQueryString':'q=a%2Bb',
             'headers':{'content-type':'application/json'}, 'cookies':['one=1', 'two=2'],
             'body':base64.b64encode(b'{"action": 12}').decode(), 'isBase64Encoded':True,
             'requestContext':{'http':{'method':'POST'}}}
    response = handle_wsgi(event, app)
    assert response['statusCode'] == 200
    assert len(response['cookies']) == 2
    assert json.loads(response['body']) == {'path':'café','q':'a+b','body':{'action':12},'cookies':{'one':'1','two':'2'},'scheme':'https'}


def test_v2_adapter_closes_iterable_and_supports_binary():
    closed = []
    class Body:
        def __iter__(self):
            yield b'\xff\x00'
        def close(self):
            closed.append(True)
    def app(environ, start_response):
        assert environ['wsgi.version'] == (1, 0)
        start_response('200 OK', [('Content-Type','application/octet-stream')])
        return Body()
    response = handle_wsgi({}, app)
    assert closed and response['isBase64Encoded']
    assert base64.b64decode(response['body']) == b'\xff\x00'
