import type { NextConfig } from "next";

const securityHeaders = [
  // Don't let browsers guess content types.
  { key: "X-Content-Type-Options", value: "nosniff" },
  // Don't allow the site to be framed by other sites (clickjacking).
  { key: "X-Frame-Options", value: "DENY" },
  { key: "Content-Security-Policy", value: "frame-ancestors 'none'" },
  // Send only the origin when linking out.
  { key: "Referrer-Policy", value: "strict-origin-when-cross-origin" },
  // The site never needs these browser features.
  { key: "Permissions-Policy", value: "camera=(), microphone=(), geolocation=(), payment=()" },
];

// Content Security Policy for production builds. Scripts and styles may only
// come from this site (inline is required by Next.js without nonces), and the
// page may only talk to the DermaLens API. Development needs eval for hot
// reload, so the policy is applied to production only.
function contentSecurityPolicy() {
  const api = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
  let apiOrigin = "";
  try {
    apiOrigin = new URL(api).origin;
  } catch {
    apiOrigin = "";
  }
  return [
    "default-src 'self'",
    "script-src 'self' 'unsafe-inline'",
    "style-src 'self' 'unsafe-inline'",
    "img-src 'self' data: blob:",
    "font-src 'self'",
    `connect-src 'self' ${apiOrigin}`.trim(),
    "worker-src 'self' blob:",
    "object-src 'none'",
    "base-uri 'self'",
    "form-action 'self'",
    "frame-ancestors 'none'",
  ].join("; ");
}

const nextConfig: NextConfig = {
  poweredByHeader: false,
  async headers() {
    const headers =
      process.env.NODE_ENV === "production"
        ? [
            ...securityHeaders.filter((h) => h.key !== "Content-Security-Policy"),
            { key: "Content-Security-Policy", value: contentSecurityPolicy() },
          ]
        : securityHeaders;
    return [{ source: "/:path*", headers }];
  },
};

export default nextConfig;
