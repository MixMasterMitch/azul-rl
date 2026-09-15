"""API Gateway HTTP API v2 to WSGI adapter; no ASGI dependency or fallback path."""
from __future__ import annotations

import base64
from io import BytesIO
import logging
import json
import sys
import time
from typing import Any, Callable
from urllib.parse import unquote

_app: Any = None
logging.getLogger().setLevel(logging.INFO)


def _get_app() -> Any:
    global _app
    if _app is None:
        started = time.perf_counter()
        logger = logging.getLogger(__name__)
        logger.info(json.dumps({"event": "initialization", "stage": "imports_started"}))
        from .server import create_app
        logger.info(json.dumps({"event": "initialization", "stage": "imports_complete",
                                "duration_ms": round((time.perf_counter() - started) * 1000, 3)}))
        _app = create_app()
        logger.info(json.dumps({"event": "initialization", "stage": "app_ready",
                                "duration_ms": round((time.perf_counter() - started) * 1000, 3)}))
    return _app


def handle_wsgi(event: dict, app: Callable) -> dict:
    http = event.get("requestContext", {}).get("http", {})
    headers = {key.lower(): value for key, value in (event.get("headers") or {}).items()}
    body = event.get("body") or ""
    payload = base64.b64decode(body) if event.get("isBase64Encoded") else body.encode("utf-8")
    if event.get("cookies"):
        headers["cookie"] = "; ".join(event["cookies"])
    environ = {
        "REQUEST_METHOD": http.get("method", "GET"),
        "SCRIPT_NAME": "",
        "PATH_INFO": unquote(event.get("rawPath", "/")).encode("utf-8").decode("latin-1"),
        "QUERY_STRING": event.get("rawQueryString", ""),
        "SERVER_NAME": headers.get("host", "lambda"), "SERVER_PORT": "443",
        "SERVER_PROTOCOL": http.get("protocol", "HTTP/1.1"),
        "REMOTE_ADDR": http.get("sourceIp", ""),
        "CONTENT_TYPE": headers.get("content-type", ""), "CONTENT_LENGTH": str(len(payload)),
        "wsgi.version": (1, 0), "wsgi.url_scheme": "https", "wsgi.input": BytesIO(payload),
        "wsgi.errors": sys.stderr, "wsgi.multithread": False, "wsgi.multiprocess": False,
        "wsgi.run_once": False,
    }
    for key, value in headers.items():
        if key not in {"content-type", "content-length"}:
            environ["HTTP_" + key.upper().replace("-", "_")] = value
    response: dict = {}
    chunks: list[bytes] = []

    def start_response(status: str, response_headers: list[tuple[str, str]], exc_info: Any = None) -> Callable:
        if exc_info and response:
            raise exc_info[1].with_traceback(exc_info[2])
        response.update(statusCode=int(status.split()[0]), headers={}, cookies=[])
        for key, value in response_headers:
            if key.lower() == "set-cookie":
                response["cookies"].append(value)
            else:
                response["headers"][key] = value
        return chunks.append

    result = app(environ, start_response)
    try:
        chunks.extend(result)
    finally:
        if hasattr(result, "close"):
            result.close()
    content = b"".join(chunks)
    # API Gateway decodes binary output; UTF-8 JSON remains directly readable.
    try:
        response["body"] = content.decode("utf-8")
        response["isBase64Encoded"] = False
    except UnicodeDecodeError:
        response["body"] = base64.b64encode(content).decode()
        response["isBase64Encoded"] = True
    return response


def handler(event: dict, context: Any) -> dict:
    return handle_wsgi(event, _get_app())
