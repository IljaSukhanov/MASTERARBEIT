"""Serve only the visualization preview on loopback, reading edits per request."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import runpy

render = runpy.run_path('/Users/iljasukhanov/.codex/plugins/cache/openai-bundled/visualize/1.0.47/skills/visualize/scripts/render.py')['render']
source = Path('/Users/iljasukhanov/.codex/visualizations/2026/10/06/01a110f5-07cd-7b00-a3a6-e21fb89c9d71/eeg-field.html')

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != '/':
            self.send_error(404)
            return
        payload = render(source).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
print(f'http://127.0.0.1:{server.server_port}/', flush=True)
server.serve_forever()
