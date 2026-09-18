#!/usr/bin/env python3
# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Serve the trace viewer, and a page for laying out a kernel from source.

Binds 0.0.0.0 rather than localhost, which is what makes the port reachable
from the host. Serves this directory, so `viewer.html` and any trace JSON
written next to it are both available:

    python3 serve.py --port 8765
    # then, on the host:
    #   http://localhost:8765/viewer.html                  (embedded sample)
    #   http://localhost:8765/viewer.html?trace=my.json    (a trace beside it)

Use --dir to serve traces written elsewhere; `viewer.html` is copied in if it
isn't already there.
"""

import argparse
import functools
import http.server
import json
import os
import shutil
import socketserver
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

# A submitted kernel is compiled, unrolled and laid out; a wide one is slow.
SUBMIT_TIMEOUT = 300


class Handler(http.server.SimpleHTTPRequestHandler):

    def end_headers(self):
        # Traces are rewritten in place while the server runs; never cache.
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, fmt, *args):
        print("  %s - %s" % (self.address_string(), fmt % args), flush=True)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self.path = "/submit.html"
        return super().do_GET()

    def _json(self, payload, status=200):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path.rstrip("/") != "/submit":
            return self.send_error(404)
        try:
            length = int(self.headers.get("Content-Length", 0))
            request = json.loads(self.rfile.read(length) or b"{}")
        except Exception as e:
            return self._json({"error": f"malformed request: {e}"}, 400)

        code = request.get("code", "")
        if not code.strip():
            return self._json({"error": "no source submitted"}, 400)

        # Compile and lay out in a separate process: a kernel that fails to
        # compile, or takes the CUDA-Q runtime down with it, must not take the
        # server too.
        argv = [sys.executable, "-m", "cudaq_qpu_layout.fromsource",
                "--regions", str(int(request.get("regions", 2))),
                "--region-size", str(int(request.get("region_size", 4)))]
        if request.get("entry"):
            argv += ["--entry", str(request["entry"])]
        try:
            done = subprocess.run(argv, input=code, capture_output=True,
                                  text=True, timeout=SUBMIT_TIMEOUT,
                                  cwd=os.path.dirname(HERE))
        except subprocess.TimeoutExpired:
            return self._json({"error": f"timed out after {SUBMIT_TIMEOUT}s. "
                                        "A kernel with many qubits or a long "
                                        "unrolled loop can take a while."})
        try:
            result = json.loads(done.stdout or "{}")
        except json.JSONDecodeError:
            # Warnings are noise here; show what actually went wrong.
            lines = [l for l in (done.stderr or "").splitlines()
                     if "Warning" not in l and not l.startswith("  ")]
            tail = "\n".join(lines).strip()[-2000:]
            return self._json({"error": tail or "the layout run produced "
                                                "nothing"})
        if "error" in result:
            return self._json(result)

        from .viewer import write_viewer
        name = "".join(c if c.isalnum() or c in "-_" else "_"
                       for c in result.get("kernel", "kernel"))
        stem = f"{name}-{int(time.time())}"
        root = self.directory
        with open(os.path.join(root, stem + ".json"), "w") as f:
            json.dump(result["trace"], f)
        write_viewer(result["trace"], os.path.join(root, stem + ".html"))
        return self._json({"viewer": stem + ".html"})


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--dir", default=HERE,
                   help="directory to serve (default: the viewer's own)")
    p.add_argument("--bind", default="0.0.0.0",
                   help="interface to bind; 0.0.0.0 reaches outside the container")
    args = p.parse_args()

    root = os.path.abspath(args.dir)
    for page in ("viewer.html", "submit.html"):
        dst = os.path.join(root, page)
        if not os.path.exists(dst) or os.path.getmtime(dst) < \
                os.path.getmtime(os.path.join(HERE, page)):
            shutil.copy(os.path.join(HERE, page), dst)

    traces = sorted(f for f in os.listdir(root) if f.endswith(".json"))

    handler = functools.partial(Handler, directory=root)
    with Server((args.bind, args.port), handler) as httpd:
        print(f"serving {root} on {args.bind}:{args.port}\n")
        print("  open from the host:")
        print(f"    http://localhost:{args.port}/           (lay out a kernel)")
        print(f"    http://localhost:{args.port}/viewer.html")
        for t in traces:
            print(f"    http://localhost:{args.port}/viewer.html?trace={t}")
        print("\n  If the host cannot reach it, the port is not forwarded.")
        print(f"  VS Code dev container: PORTS panel -> Forward a Port -> {args.port},")
        print(f"  or add \"forwardPorts\": [{args.port}] to devcontainer.json and rebuild.")
        print(f"  Plain docker: run the container with -p {args.port}:{args.port}.")
        print("  (The 172.x container address is not routable from a WSL2 host.)")
        print("\nCtrl-C to stop.", flush=True)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped")


if __name__ == "__main__":
    main()
