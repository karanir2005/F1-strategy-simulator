"use client";

import { LazyMotion, MotionConfig, domMax } from "motion/react";

/**
 * LazyMotion + domMax: domMax is needed for layout/layoutId (nav underline, row re-sort).
 * Components must use `m.*` from "motion/react-m" (strict throws on `motion.*`).
 * reducedMotion="user" drops transform/layout animation for people who ask for it.
 */
export function MotionProvider({ children }: { children: React.ReactNode }) {
  return (
    <MotionConfig reducedMotion="user">
      <LazyMotion features={domMax} strict>
        {children}
      </LazyMotion>
    </MotionConfig>
  );
}
