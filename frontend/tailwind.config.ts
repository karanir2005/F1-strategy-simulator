import type { Config } from "tailwindcss";

// every token is an oklch channel triple in globals.css; <alpha-value> makes bg-x/20 work
const c = (name: string) => `oklch(var(--${name}) / <alpha-value>)`;

const config: Config = {
  darkMode: ["class"],
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ["var(--font-geist-sans)", "ui-sans-serif", "system-ui", "sans-serif"],
        mono: ["var(--font-mono)", "ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
        display: ["var(--font-display)", "var(--font-geist-sans)", "sans-serif"],
      },
      colors: {
        // brand tokens
        bg: c("bg"),
        raised: c("raised"),
        sunken: c("sunken"),
        hover: c("hover"),
        line: { DEFAULT: c("line"), strong: c("line-strong") },
        fg: c("fg"),
        "muted-fg": c("muted-fg"),
        faint: c("faint"),
        signal: { DEFAULT: c("signal"), fg: c("signal-fg"), text: c("signal-text") },
        good: c("good"),
        best: c("best"),
        bad: c("bad"),
        warn: c("warn"),
        tyre: {
          soft: "var(--tyre-soft)",
          medium: "var(--tyre-medium)",
          hard: "var(--tyre-hard)",
          inter: "var(--tyre-inter)",
          wet: "var(--tyre-wet)",
        },
        // shadcn names, kept so the base-ui components in components/ui keep working
        background: c("bg"),
        foreground: c("fg"),
        card: { DEFAULT: c("raised"), foreground: c("fg") },
        popover: { DEFAULT: c("raised"), foreground: c("fg") },
        primary: { DEFAULT: c("signal"), foreground: c("signal-fg") },
        secondary: { DEFAULT: c("sunken"), foreground: c("fg") },
        muted: { DEFAULT: c("sunken"), foreground: c("muted-fg") },
        accent: { DEFAULT: c("hover"), foreground: c("fg") },
        destructive: { DEFAULT: c("bad") },
        border: c("line"),
        input: c("line-strong"),
        ring: c("signal"),
      },
      borderRadius: {
        lg: "var(--radius)",
        md: "calc(var(--radius) - 2px)",
        sm: "2px",
        panel: "10px",
      },
      boxShadow: {
        pop: "var(--shadow-pop)",
      },
    },
  },
  plugins: [require("tailwindcss-animate")],
};
export default config;
