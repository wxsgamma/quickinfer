import http.client
import json
from threading import Thread
import unittest

from quickinfer.server import create_server, parse_request


class ServerTest(unittest.TestCase):
    def test_request_validation(self):
        for body in ({}, [], {"prompts": []}, {"prompts": [True]},
                     {"prompts": ["Hi"], "max_tokens": 0},
                     {"prompts": ["Hi"], "ignore_eos": "false"},
                     {"prompts": ["Hi"], "unknown": 1}):
            with self.subTest(body=body), self.assertRaises((ValueError, TypeError)):
                parse_request(body)
        prompts, params = parse_request({"prompts": ["Hello", [1, 2]], "max_tokens": 5})
        self.assertEqual(prompts, ["Hello", [1, 2]])
        self.assertEqual(params.max_tokens, 5)

    def test_http_routes_and_invalid_json(self):
        # Generation is covered by GPU smoke; this test exercises actual HTTP routing.
        with create_server(None, port=0) as server:
            thread = Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
                connection.request("GET", "/health")
                response = connection.getresponse()
                self.assertEqual(response.status, 200)
                self.assertEqual(json.loads(response.read()), {"status": "ok"})
                connection.request("POST", "/generate", body="invalid")
                response = connection.getresponse()
                self.assertEqual(response.status, 400)
                response.read()
                connection.request("GET", "/missing")
                response = connection.getresponse()
                self.assertEqual(response.status, 404)
                response.read()
                connection.close()
            finally:
                server.shutdown()
                thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
