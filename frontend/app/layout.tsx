import type { Metadata } from "next";
import localFont from "next/font/local";
import { Archivo, JetBrains_Mono } from "next/font/google";
import "./globals.css";
import { MotionProvider } from "@/components/shell/motion-provider";
import { SiteFooter } from "@/components/shell/site-footer";
import { SiteNav } from "@/components/shell/site-nav";

const geistSans = localFont({
  src: "./fonts/GeistVF.woff",
  variable: "--font-geist-sans",
  weight: "100 900",
  display: "swap",
});
const archivo = Archivo({
  subsets: ["latin"],
  axes: ["wdth"],
  variable: "--font-display",
  display: "swap",
});
const jetbrains = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-mono",
  display: "swap",
});

export const metadata: Metadata = {
  title: { default: "Apex | F1 race prediction and strategy", template: "%s | Apex" },
  description: "Race winner probabilities and Monte Carlo pit strategy comparison.",
};

// dark is the default; only an explicit "light" choice removes the class (runs before paint)
const themeScript = `try{if(localStorage.getItem('theme')==='light')document.documentElement.classList.remove('dark')}catch(e){}`;

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`dark ${geistSans.variable} ${archivo.variable} ${jetbrains.variable}`}
      suppressHydrationWarning
    >
      <head>
        <script dangerouslySetInnerHTML={{ __html: themeScript }} />
      </head>
      <body>
        <MotionProvider>
          <a
            href="#main"
            className="sr-only focus:not-sr-only focus:fixed focus:left-3 focus:top-3 focus:z-50 focus:rounded-lg focus:bg-signal focus:px-3 focus:py-2 focus:text-signal-fg"
          >
            Skip to content
          </a>
          <SiteNav />
          <main id="main" className="mx-auto max-w-[1200px] px-4 pt-8 sm:px-6 sm:pt-10">
            {children}
          </main>
          <SiteFooter />
        </MotionProvider>
      </body>
    </html>
  );
}
