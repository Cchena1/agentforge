const http = require("http");
const fs = require("fs");
const fsp = require("fs/promises");
const path = require("path");
const { spawn } = require("child_process");
const { URL } = require("url");

const PROJECT_ROOT = __dirname;
const WEB_ROOT = path.join(PROJECT_ROOT, "web");
const FRONTEND_PORT = Number(process.env.FRONTEND_PORT || 3000);
const BACKEND_URL = new URL(process.env.BACKEND_URL || "http://127.0.0.1:8787");
const BACKEND_PYTHON = path.resolve(PROJECT_ROOT, "..", ".codex-python", "python", "python.exe");
const BACKEND_OUT_LOG = path.join(PROJECT_ROOT, "backend.out.log");
const BACKEND_ERR_LOG = path.join(PROJECT_ROOT, "backend.err.log");

let backendProcess = null;

const MIME_TYPES = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".txt": "text/plain; charset=utf-8",
  ".svg": "image/svg+xml",
};

function send(res, statusCode, headers, body) {
  res.writeHead(statusCode, headers);
  res.end(body);
}

async function serveStatic(res, relativePath) {
  const root = path.resolve(WEB_ROOT);
  const resolved = path.resolve(WEB_ROOT, relativePath);
  if (resolved !== root && !resolved.startsWith(root + path.sep)) {
    send(res, 403, { "Content-Type": "text/plain; charset=utf-8" }, "Forbidden");
    return;
  }

  try {
    const data = await fsp.readFile(resolved);
    const type = MIME_TYPES[path.extname(resolved).toLowerCase()] || "application/octet-stream";
    send(res, 200, { "Content-Type": type, "Content-Length": data.length }, data);
  } catch (error) {
    send(res, 404, { "Content-Type": "text/plain; charset=utf-8" }, "Not found");
  }
}

function probeBackend() {
  return new Promise((resolve) => {
    const req = http.get(
      {
        hostname: BACKEND_URL.hostname,
        port: BACKEND_URL.port,
        path: "/health",
        timeout: 500,
      },
      (res) => {
        res.resume();
        resolve(res.statusCode === 200);
      }
    );
    req.on("error", () => resolve(false));
    req.on("timeout", () => {
      req.destroy();
      resolve(false);
    });
  });
}

async function waitForBackendReady(deadlineMs = 10000) {
  const startedAt = Date.now();
  while (Date.now() - startedAt < deadlineMs) {
    if (await probeBackend()) {
      return true;
    }
    await new Promise((resolve) => setTimeout(resolve, 250));
  }
  return false;
}

async function ensureBackend() {
  if (await probeBackend()) {
    return;
  }
  if (!fs.existsSync(BACKEND_PYTHON)) {
    throw new Error(`Backend Python runtime not found at ${BACKEND_PYTHON}`);
  }
  const outHandle = fs.openSync(BACKEND_OUT_LOG, "a");
  const errHandle = fs.openSync(BACKEND_ERR_LOG, "a");
  backendProcess = spawn(BACKEND_PYTHON, ["app.py", "--host", "127.0.0.1", "--port", "8787"], {
    cwd: PROJECT_ROOT,
    stdio: ["ignore", outHandle, errHandle],
    windowsHide: true,
  });
  const ready = await waitForBackendReady();
  if (!ready) {
    throw new Error("Backend did not become ready on port 8787");
  }
}

function proxyApi(req, res) {
  const target = new URL(req.url, BACKEND_URL);
  const proxyReq = http.request(
    target,
    {
      method: req.method,
      headers: {
        ...req.headers,
        host: target.host,
        origin: BACKEND_URL.origin,
      },
    },
    (proxyRes) => {
      const headers = { ...proxyRes.headers };
      if (typeof headers["content-type"] === "string" && headers["content-type"].startsWith("text/")) {
        headers["content-type"] = `${headers["content-type"]}; charset=utf-8`;
      }
      res.writeHead(proxyRes.statusCode || 502, headers);
      proxyRes.pipe(res);
    }
  );

  proxyReq.on("error", (error) => {
    send(res, 502, { "Content-Type": "application/json; charset=utf-8" }, JSON.stringify({ ok: false, error: error.message }));
  });

  req.pipe(proxyReq);
}

async function bootstrap() {
  await ensureBackend();

  const server = http.createServer((req, res) => {
    if (!req.url) {
      send(res, 400, { "Content-Type": "text/plain; charset=utf-8" }, "Bad request");
      return;
    }

    const parsed = new URL(req.url, `http://127.0.0.1:${FRONTEND_PORT}`);
    if (parsed.pathname.startsWith("/api/")) {
      proxyApi(req, res);
      return;
    }
    if (parsed.pathname === "/") {
      serveStatic(res, "index.html");
      return;
    }
    if (parsed.pathname === "/app.js") {
      serveStatic(res, "app.js");
      return;
    }
    if (parsed.pathname === "/styles.css") {
      serveStatic(res, "styles.css");
      return;
    }
    send(res, 404, { "Content-Type": "text/plain; charset=utf-8" }, "Not found");
  });

  server.listen(FRONTEND_PORT, "127.0.0.1", () => {
    console.log(`AI agent-1 frontend running at http://127.0.0.1:${FRONTEND_PORT}`);
    console.log(`Proxying API traffic to ${BACKEND_URL.origin}`);
  });

  const shutdown = () => {
    server.close();
    if (backendProcess && !backendProcess.killed) {
      backendProcess.kill();
    }
  };

  process.on("SIGINT", shutdown);
  process.on("SIGTERM", shutdown);
}

bootstrap().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
