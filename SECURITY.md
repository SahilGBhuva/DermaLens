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
- Only JPEG, PNG and WebP are accepted, and files are verified before use.
- At most two analyses run at once (`INFERENCE_CONCURRENCY`); others wait up to 30 seconds, then get a 503 "busy" response. Inference runs off the event loop, so health checks stay responsive.
- CORS allows only the origins in `CORS_ORIGINS` and no credentials (the API uses no cookies or auth).

Model integrity:
- Weights are loaded with `torch.load(..., weights_only=True)`, which refuses arbitrary pickled code.
- `model_config.json` holds the weights' SHA-256; the API refuses weights that don't match, and only publishes evaluation results measured on the weights it is serving.
- Hosted artifacts are only downloaded over `https://`; `MODEL_SHA256` can pin the download.

Deployment:
- The Docker image runs as an unprivileged user that can only write to `/models`, and installs CPU-only PyTorch.
- The website sends `X-Content-Type-Options`, `X-Frame-Options`/`frame-ancestors`, `Referrer-Policy` and `Permissions-Policy` headers, and hides `X-Powered-By`.
- GitHub Actions runs with read-only repository permissions.

Dependencies were audited on 2026-10-03 with `pip-audit` (no known vulnerabilities in pinned Python packages) and `npm audit`. One build-time advisory remains in the PostCSS copy bundled with Next.js 15; it affects CSS processing during `next build`, not visitors, and the fix requires upgrading to Next.js 16.

Re-run the audits with:

```bash
pip install pip-audit && pip-audit -r backend/requirements.txt -r ml/requirements.txt
cd frontend && npm audit --omit=dev
```
