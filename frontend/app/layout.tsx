import type { Metadata, Viewport } from "next";
import "./globals.css";

export const metadata: Metadata = {
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
    title: "DermaLens — Interrogate the Model",
    description:
      "Explainable medical-image ML research for inspecting when predictions become fragile.",
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
      <body>{children}</body>
    </html>
  );
}
