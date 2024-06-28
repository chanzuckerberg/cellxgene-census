"""This module defines a plugin class that logs each request to a logfile.

This class needs to be importable by the proxy server which runs in a separate process.
See the user agent tests for usage.
"""

import json
import traceback
from pathlib import Path

import proxy
from proxy.common.flag import flags

flags.add_argument(
    "--request-log-file",
    type=str,
    default="",
    help="Where to log the requests to.",
)


class RequestLoggerPlugin(proxy.http.proxy.HttpProxyBasePlugin):  # type: ignore
    def handle_client_request(self, request: proxy.http.parser.HttpParser) -> proxy.http.parser.HttpParser:
        # If anything fails in here, it just fails to respond
        try:
            with Path(self.flags.request_log_file).open("a") as f:
                record = {
                    "method": request.method.decode(),
                    "url": str(request._url),
                }

                if request.headers:
                    record["headers"] = {k2.decode().lower(): v.decode() for _, (k2, v) in request.headers.items()}
                f.write(f"{json.dumps(record)}\n")
        except Exception as e:
            # Making sure there is some visible output
            print(repr(e))
            traceback.print_exception(e)
            raise e
        return request
