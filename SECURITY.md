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
