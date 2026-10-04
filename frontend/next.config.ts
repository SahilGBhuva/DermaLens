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

const nextConfig: NextConfig = {
  poweredByHeader: false,
  async headers() {
    return [{ source: "/:path*", headers: securityHeaders }];
  },
};

export default nextConfig;
