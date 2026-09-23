import "./globals.css";

export const metadata = {
  title: "DermaLens — Interrogate the Model",
  description:
    "An explainable medical-image ML research experience for inspecting predictions, uncertainty, attention, robustness, and model limitations.",
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
