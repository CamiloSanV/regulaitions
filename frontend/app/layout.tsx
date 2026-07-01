import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "regulAItions — AI Compliance Assistant",
  description: "EU AI Act and GDPR compliance agent",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}

