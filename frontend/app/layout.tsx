import "./globals.css";

export const metadata = {
  title: "DermaLens",
  description: "Educational ML analysis for skin-lesion images",
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
