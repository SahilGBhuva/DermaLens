"""Request guard: body-size cap, per-client rate limit, security headers.

Written as plain ASGI middleware so it sees the raw byte stream. That matters
for uploads sent without a Content-Length header (chunked): the cap is enforced
on bytes actually received, so an oversized body is cut off instead of being
spooled to disk first.
"""

from __future__ import annotations

import json
import time
from collections import deque

SECURITY_HEADERS = [
    (b"x-content-type-options", b"nosniff"),
    (b"x-frame-options", b"DENY"),
    (b"referrer-policy", b"no-referrer"),
    # Predictions are derived from medical images: never cache them.
    (b"cache-control", b"no-store"),
]
# The JSON API needs no scripts, styles or frames. /docs (dev only) does.
API_CSP = (b"content-security-policy", b"default-src 'none'; frame-ancestors 'none'")
DOCS_PATHS = ("/docs", "/redoc", "/openapi.json")


class RequestGuard:
    def __init__(
        self,
        app,
        max_body_bytes: int,
        limited_paths: tuple[str, ...],
        rate_limit: int,
        rate_window_seconds: float,
        trust_proxy_headers: bool = False,
        max_tracked_clients: int = 10_000,
    ):
        self.app = app
        self.max_body_bytes = max_body_bytes
        self.limited_paths = limited_paths
        self.rate_limit = rate_limit
        self.rate_window = rate_window_seconds
        self.trust_proxy = trust_proxy_headers
        self.max_tracked = max_tracked_clients
        self.hits: dict[str, deque[float]] = {}

    # ── rate limiting ────────────────────────────────────────────────
    def client_id(self, scope, headers: dict[bytes, bytes]) -> str:
        if self.trust_proxy:
            # Behind a platform proxy (e.g. Render) the right-most entry is the
            # one the proxy appended; entries further left are client-supplied.
            forwarded = headers.get(b"x-forwarded-for", b"").decode("latin-1")
            if forwarded.strip():
                return forwarded.split(",")[-1].strip()
        client = scope.get("client")
        return client[0] if client else "unknown"

    def retry_after(self, client: str) -> int:
        """Seconds until `client` may proceed; 0 records the hit and allows it."""
        now = time.monotonic()
        window = self.hits.setdefault(client, deque())
        while window and now - window[0] > self.rate_window:
            window.popleft()
        if len(window) >= self.rate_limit:
            return max(1, int(self.rate_window - (now - window[0])) + 1)
        window.append(now)
        if len(self.hits) > self.max_tracked:
            # Bound memory: drop clients with no recent activity, then oldest.
            for key in [k for k, v in self.hits.items() if not v or now - v[-1] > self.rate_window]:
                self.hits.pop(key, None)
            while len(self.hits) > self.max_tracked:
                self.hits.pop(next(iter(self.hits)))
        return 0

    # ── ASGI ─────────────────────────────────────────────────────────
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers") or [])
        path = scope.get("path", "")
        state = {"started": False, "too_large": False}

        async def send_with_headers(message):
            if message["type"] == "http.response.start":
                state["started"] = True
                existing = {k.lower() for k, _ in message.get("headers", [])}
                extra = [h for h in SECURITY_HEADERS if h[0] not in existing]
                if not path.startswith(DOCS_PATHS) and API_CSP[0] not in existing:
                    extra.append(API_CSP)
                message = {**message, "headers": list(message.get("headers", [])) + extra}
            await send(message)

        async def reply(status: int, detail: str, extra_headers=()):
            body = json.dumps({"detail": detail}).encode()
            await send_with_headers(
                {
                    "type": "http.response.start",
                    "status": status,
                    "headers": [
                        (b"content-type", b"application/json"),
                        (b"content-length", str(len(body)).encode()),
                        *extra_headers,
                    ],
                }
            )
            await send({"type": "http.response.body", "body": body})

        if scope.get("method") == "POST" and path in self.limited_paths:
            wait = self.retry_after(self.client_id(scope, headers))
            if wait:
                await reply(
                    429,
                    f"Too many analyses from this connection. Try again in {wait} seconds.",
                    [(b"retry-after", str(wait).encode())],
                )
                return

        declared = headers.get(b"content-length", b"")
        if declared.isdigit() and int(declared) > self.max_body_bytes:
            await reply(413, "Image must be under 10 MB.")
            return

        received = 0

        async def capped_receive():
            nonlocal received
            if state["too_large"]:
                return {"type": "http.disconnect"}
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_body_bytes:
                    # Stop reading; the app sees a disconnect and gives up.
                    state["too_large"] = True
                    return {"type": "http.disconnect"}
            return message

        async def guarded_send(message):
            # Once the cap is hit, replace whatever the app says with a 413.
            if state["too_large"]:
                return
            await send_with_headers(message)

        try:
            await self.app(scope, capped_receive, guarded_send)
        except Exception:
            if not state["too_large"]:
                raise
        if state["too_large"] and not state["started"]:
            await reply(413, "Image must be under 10 MB.")
