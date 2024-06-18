"""This module defines a plugin class that logs each request to the logfile."""

import json
import sys
import traceback
from pathlib import Path
from pprint import pprint

import proxy
from proxy.common.flag import flags

flags.add_argument(
    "--request-logfile",
    type=str,
    default="",
    help="Where to log the requests to.",
)


class ProxyPlugin(proxy.http.proxy.HttpProxyBasePlugin):
    def handle_client_request(self, request: proxy.http.parser.HttpParser) -> proxy.http.parser.HttpParser:
        # If anything fails in here, it just fails to respond
        # return request
        # print(request.headers,)
        try:
            with Path(self.flags.request_logfile).open("a") as f:
                record = {
                    "method": request.method.decode(),
                    "url": str(request._url),
                    # "headers": {k2.decode().lower(): v.decode() for _, (k2, v) in request.headers.items()} if request.header else {},
                }

                if request.headers:
                    # record["headers"] = {k.decode(): [v.decode() for v in vs] for k, vs in request.headers.items()}
                    record["headers"] = {k2.decode().lower(): v.decode() for _, (k2, v) in request.headers.items()}
                print("Request:")
                print(record)
                print()
                f.write(f"{json.dumps(record)}\n")
        except Exception as e:
            # Making sure there is some visible output
            print(repr(e))
            traceback.print_exception(e)
            raise e
        return request

    # def handle_upstream_chunk(self, chunk):
    #     print("Response:")
    #     as_bytes = bytes(chunk)
    #     if len(as_bytes) > 1000:
    #         print(as_bytes[:300])
    #     else:
    #         print(as_bytes)
    #     print()
    #     return chunk
