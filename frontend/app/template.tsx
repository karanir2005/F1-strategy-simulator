"use client";

import * as m from "motion/react-m";

/** Remounts on every navigation: enter-only fade (exit animations need AnimatePresence to outlive the child). */
export default function Template({ children }: { children: React.ReactNode }) {
  return (
    <m.div
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.22, ease: [0.22, 1, 0.36, 1] }}
    >
      {children}
    </m.div>
  );
}
