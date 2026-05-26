"""AWS Lambda handler wrapping the Flask app."""

from __future__ import annotations

import os

# Set up environment for Lambda
os.environ.setdefault("FLASK_ENV", "production")

from .server import create_app

app = create_app()


def handler(event, context):
    """Lambda handler using mangum-style WSGI adapter."""
    try:
        from mangum import Mangum
        mangum_handler = Mangum(app)
        return mangum_handler(event, context)
    except ImportError:
        # Fallback: manual API Gateway v2 event parsing
        return _handle_apigw_v2(event)


def _handle_apigw_v2(event: dict) -> dict:
    """Minimal API Gateway v2 event handler."""
    import json
    from io import BytesIO
    from urllib.parse import urlencode

    request_context = event.get("requestContext", {})
    http_info = request_context.get("http", {})
    method = http_info.get("method", "GET")
    path = event.get("rawPath", "/")
    headers = event.get("headers", {})
    body = event.get("body", "")
    is_base64 = event.get("isBase64Encoded", False)

    if is_base64 and body:
        import base64
        body = base64.b64decode(body)
    elif body:
        body = body.encode("utf-8")
    else:
        body = b""

    query_string = event.get("rawQueryString", "")

    environ = {
        "REQUEST_METHOD": method,
        "PATH_INFO": path,
        "QUERY_STRING": query_string,
        "CONTENT_TYPE": headers.get("content-type", ""),
        "CONTENT_LENGTH": str(len(body)),
        "SERVER_NAME": "lambda",
        "SERVER_PORT": "443",
        "SERVER_PROTOCOL": "HTTP/1.1",
        "wsgi.input": BytesIO(body),
        "wsgi.errors": BytesIO(),
        "wsgi.url_scheme": "https",
    }

    for key, value in headers.items():
        environ[f"HTTP_{key.upper().replace('-', '_')}"] = value

    response_started = []
    response_body = []

    def start_response(status, response_headers, exc_info=None):
        response_started.append((status, response_headers))

    result = app(environ, start_response)
    for data in result:
        response_body.append(data)

    status_code = int(response_started[0][0].split(" ")[0])
    resp_headers = dict(response_started[0][1])
    resp_body = b"".join(response_body).decode("utf-8")

    return {
        "statusCode": status_code,
        "headers": resp_headers,
        "body": resp_body,
    }
