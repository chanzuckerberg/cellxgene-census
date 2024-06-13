"""This module defines a plugin class that logs each request to the logfile."""

import json
from pathlib import Path
from pprint import pprint

import proxy


class ProxyPlugin(proxy.http.proxy.HttpProxyBasePlugin):
    def handle_client_request(self, request: proxy.http.parser.HttpParser) -> proxy.http.parser.HttpParser:
        # If anything fails in here, it just fails to respond
        try:
            with Path(self.flags.log_file).open("a") as f:
                record = json.dumps(
                    {
                        "method": request.method.decode(),
                        "url": str(request._url),
                        "headers": {k2.decode().lower(): v.decode() for _, (k2, v) in request.headers.items()},
                    },
                )
                f.write(f"{record}\n")
        except Exception as e:
            # Making sure there is some visible output
            print(repr(e))
            raise e
        return request
