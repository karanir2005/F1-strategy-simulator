import type { Metadata } from "next";

export const metadata: Metadata = { title: "Predict" };

export default function PredictLayout({ children }: { children: React.ReactNode }) {
  return children;
}
