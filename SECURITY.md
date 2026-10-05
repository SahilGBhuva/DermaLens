# Security

DermaLens is an educational research prototype and not a medical device.

## Reporting issues

Please open a GitHub issue for non-sensitive bugs.

For anything involving private medical images or personal data, do not attach the image to a public GitHub issue.

## Data handling

The repository does not require users to store uploaded images. The default API analyzes an image in memory and does not intentionally persist it.

Production deployments should:
- use HTTPS,
- avoid request logging that captures image payloads,
- define strict CORS origins,
- enforce upload limits,
- and avoid collecting personal health information unless a compliant data-handling plan exists.

## Protections in place

API (`backend/app/main.py`):
- Uploads are limited to 10 MB. The declared request size is checked before parsing, and at most 10 MB + 1 byte is ever read.
- Images over 25 megapixels are rejected from the file header **before decoding**, which blocks decompression bombs (small files that expand to gigabytes in memory). Accepted images are shrunk to at most 1024 px first.
- Only JPEG, PNG and WebP are ever decoded (`Image.open(..., formats=...)`), whatever Content-Type the client claims; TIFF, GIF, BMP, ICO and other formats are refused.
- Request bodies are capped on the bytes actually received (`backend/app/guards.py`), so uploads sent without a Content-Length (chunked) are cut off at 10 MB instead of being spooled to disk.
- Rate limit: 30 analyses per client per 10 minutes (`RATE_LIMIT_REQUESTS`, `RATE_LIMIT_WINDOW_SECONDS`), answered with 429 and `Retry-After`. Behind Render, `TRUST_PROXY_HEADERS=1` uses the right-most `X-Forwarded-For` entry (the one the proxy appends), so a client cannot dodge the limit by spoofing the header.
- Every API response carries `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy: no-referrer`, `Cache-Control: no-store` (predictions derive from medical images) and `Content-Security-Policy: default-src 'none'`.
- Interactive API docs (`/docs`, `/openapi.json`) are disabled when `DERMALENS_ENV=production` (set in the Docker image).
- One analysis runs at a time by default (`INFERENCE_CONCURRENCY`); others wait up to 30 seconds, then get a 503 "busy" response. Load-testing in a 512 MB container showed two concurrent Grad-CAM requests could exceed the memory limit. Inference runs off the event loop, so health checks stay responsive.
- CORS allows only the origins in `CORS_ORIGINS` and no credentials (the API uses no cookies or auth).

Model integrity:
- Weights are loaded with `torch.load(..., weights_only=True)`, which refuses arbitrary pickled code.
- `model_config.json` holds the weights' SHA-256; the API refuses weights that don't match, and only publishes evaluation results measured on the weights it is serving.
- Hosted artifacts are only downloaded over `https://` (`file://` only with `DERMALENS_ALLOW_FILE_URLS=1`, for local tests); `MODEL_SHA256` can pin the download.
- ML scripts also load weights with `weights_only=True`.

Deployment:
- The Docker image runs as an unprivileged user that can only write to `/models`, and installs CPU-only PyTorch.
- The website sends `X-Content-Type-Options`, `X-Frame-Options`/`frame-ancestors`, `Referrer-Policy` and `Permissions-Policy` headers, hides `X-Powered-By`, and in production a Content Security Policy that only allows scripts, styles, fonts and images from the site itself and network requests to the DermaLens API.
- GitHub Actions runs with read-only repository permissions, and every third-party action is pinned to a commit SHA.
- Dependabot (`.github/dependabot.yml`) proposes weekly updates for pip, npm, Docker and GitHub Actions; CodeQL (`.github/workflows/codeql.yml`) scans Python and TypeScript with the security-extended queries.

Verified on 2026-10-05 against the production container (512 MB): docs return 404, headers present, disguised TIFF refused, sixth rapid request from one client gets 429 while other clients are served, spoofed `X-Forwarded-For` does not bypass the limit, runs as `dermalens`, peak memory under the limit.

Dependencies were audited on 2026-10-03 with `pip-audit` (no known vulnerabilities in pinned Python packages) and `npm audit`. One build-time advisory remains in the PostCSS copy bundled with Next.js 15; it affects CSS processing during `next build`, not visitors, and the fix requires upgrading to Next.js 16.

Static analysis: `bandit -r backend/app ml -ll` reports no unresolved medium/high findings (the one `urlopen` call is scheme-checked).

Recommended GitHub settings (Settings → Code security): enable Dependabot alerts and security updates, and protect the `main` branch against force-pushes and deletion.

Re-run the audits with:

```bash
pip install pip-audit bandit && pip-audit -r backend/requirements.txt -r ml/requirements.txt
bandit -r backend/app ml -ll
cd frontend && npm audit --omit=dev
```
