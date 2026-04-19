import "./globals.css";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Devreotes Research Explorer",
  description: "GraphRAG UI rebuilt with Next.js",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
