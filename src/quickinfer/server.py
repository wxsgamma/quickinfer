"""A serialized, non-streaming JSON HTTP adapter for the offline engine."""

from http.server import BaseHTTPRequestHandler, HTTPServer
import json

from quickinfer.sampling_params import SamplingParams


def parse_request(payload):
    if not isinstance(payload, dict):
        raise ValueError("JSON body must be an object")
    unknown = set(payload) - {"prompts", "temperature", "max_tokens", "ignore_eos"}
    if unknown:
        raise ValueError(f"Unknown request fields: {', '.join(sorted(unknown))}")
    prompts = payload.get("prompts")
    if not isinstance(prompts, list) or not prompts:
        raise ValueError("prompts must be a nonempty list of strings or token-ID lists")
    for prompt in prompts:
        if not isinstance(prompt, str) and not (
            isinstance(prompt, list) and prompt and
            all(isinstance(t, int) and not isinstance(t, bool) for t in prompt)
        ):
            raise ValueError("Each prompt must be text or a nonempty list of integer token IDs")
    if not isinstance(payload.get("ignore_eos", False), bool):
        raise ValueError("ignore_eos must be a boolean")
    params = SamplingParams(**{k: v for k, v in payload.items() if k != "prompts"})
    return prompts, params


def create_server(engine, host="127.0.0.1", port=8080):
    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(30)

        def send_json(self, status, data):
            body = json.dumps(data, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self.send_json(200, {"status": "ok"})
            else:
                self.send_json(404, {"error": "Not found"})

        def do_POST(self):
            if self.path != "/generate":
                self.send_json(404, {"error": "Not found"})
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 1048576:
                    raise ValueError("Content-Length must be between 1 and 1048576")
                payload = json.loads(self.rfile.read(size))
                prompts, params = parse_request(payload)
                outputs = engine.generate(prompts, params, use_tqdm=False)
            except (ValueError, TypeError, UnicodeDecodeError) as error:
                self.send_json(400, {"error": str(error)})
                return
            self.send_json(200, {"outputs": outputs})

    # A single request at a time protects the engine's scheduler/global context.
    return HTTPServer((host, port), Handler)


def serve(engine, host="127.0.0.1", port=8080):
    with create_server(engine, host, port) as server:
        print(f"Quickinfer listening on http://{host}:{server.server_port}", flush=True)
        server.serve_forever()
