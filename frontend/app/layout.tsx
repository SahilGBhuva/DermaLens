import type { Metadata, Viewport } from "next";
import "./globals.css";

// Absolute base for link-preview URLs: set NEXT_PUBLIC_SITE_URL in production;
// on Vercel the production domain is used automatically.
const siteUrl =
  process.env.NEXT_PUBLIC_SITE_URL ??
  (process.env.VERCEL_PROJECT_PRODUCTION_URL
    ? `https://${process.env.VERCEL_PROJECT_PRODUCTION_URL}`
    : "http://localhost:3000");

const apiOrigin = (() => {
  try {
    return new URL(process.env.NEXT_PUBLIC_API_URL ?? "").origin;
  } catch {
    return null;
  }
})();

export const metadata: Metadata = {
  metadataBase: new URL(siteUrl),
  title: {
    default: "DermaLens — See what the model sees",
    template: "%s · DermaLens",
  },
  description:
    "An explainable medical-image ML research experience for inspecting predictions, uncertainty, attention, robustness, and model limitations.",
  applicationName: "DermaLens",
  keywords: [
    "explainable AI",
    "medical imaging",
    "machine learning",
    "robustness",
    "Grad-CAM",
    "skin lesion research",
  ],
  authors: [{ name: "DermaLens" }],
  creator: "DermaLens",
  robots: {
    index: true,
    follow: true,
  },
  openGraph: {
    title: "DermaLens — See what the model sees",
    description:
      "Inspect prediction confidence, uncertainty, model attention, robustness, and limitations.",
    type: "website",
    siteName: "DermaLens",
  },
  twitter: {
    card: "summary_large_image",
    title: "DermaLens — See what the model sees",
    description:
      "Inspect prediction confidence, uncertainty, model attention, robustness, and limitations.",
  },
};

export const viewport: Viewport = {
  themeColor: "#eef4fb",
  colorScheme: "light",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <head>
        {/* Open the connection to the API while the page loads, before the first upload. */}
        {apiOrigin && <link rel="preconnect" href={apiOrigin} crossOrigin="anonymous" />}
      </head>
      <body>{children}</body>
    </html>
  );
}
