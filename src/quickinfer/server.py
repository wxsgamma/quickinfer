from concurrent.futures import Future
from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer
import json
from queue import Empty, Queue
from threading import Thread

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


class PDServer(ThreadingHTTPServer):
    daemon_threads = False

    def __init__(self, address, handler, engine):
        self.worker = None
        super().__init__(address, handler)
        self.engine = engine
        self.incoming = Queue()
        self.worker = Thread(target=self.run_engine)
        self.worker.start()

    def generate(self, prompts, params):
        results = [Future() for _ in prompts]
        self.incoming.put((prompts, params, results))
        return [result.result() for result in results]

    def run_engine(self):
        # HTTP 线程只入队，只有这个线程动调度器。上一条还在 D 上时，新请求就能进 P。
        pending = {}
        while True:
            try:
                batch = self.incoming.get(block=not pending)
            except Empty:
                batch = ()
            if batch is None:
                return
            if batch:
                prompts, params, results = batch
                try:
                    ids = self.engine._enqueue_requests(prompts, params)
                    pending.update(zip(ids, results, strict=True))
                except Exception as error:
                    for result in results:
                        result.set_exception(error)
            if pending:
                try:
                    outputs, _ = self.engine.step()
                    for seq_id, ids in outputs:
                        output = {"text": self.engine.tokenizer.decode(ids), "token_ids": ids}
                        pending.pop(seq_id).set_result(output)
                except Exception as error:
                    for result in pending.values():
                        result.set_exception(error)
                    pending.clear()

    def server_close(self):
        # 先等已接收的 HTTP 请求拿到结果，再停调度线程和 GPU worker。
        super().server_close()
        if self.worker is not None:
            self.incoming.put(None)
            self.worker.join()
            self.worker = None


def create_server(engine, host="127.0.0.1", port=8080):
    pd = getattr(engine, "prefill_worker", None) is not None
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
                outputs = self.server.generate(prompts, params) if pd else engine.generate(prompts, params, use_tqdm=False)
            except (ValueError, TypeError, UnicodeDecodeError) as error:
                self.send_json(400, {"error": str(error)})
                return
            except RuntimeError as error:
                self.log_error("Generation failed: %s", error)
                self.send_json(500, {"error": "Generation failed"})
                return
            self.send_json(200, {"outputs": outputs})

    if pd:
        return PDServer((host, port), Handler, engine)
    return HTTPServer((host, port), Handler)


def serve(engine, host="127.0.0.1", port=8080):
    with create_server(engine, host, port) as server:
        print(f"Quickinfer listening on http://{host}:{server.server_port}", flush=True)
        server.serve_forever()
