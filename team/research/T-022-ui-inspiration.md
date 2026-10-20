# T-022 UI inspiration and design spec

Scope: frontend redesign into `/predict` and `/strategy`. Stack today: Next 14.2, React 18, Tailwind 3.4, base-ui shadcn, Recharts 3.10 (`frontend/package.json`). Current `globals.css` is the untouched shadcn neutral grey oklch theme with no brand colour, no fonts. That is itself the "default AI site" look.

Evidence grade: web search was thin on formula1.com/F1 TV internals (no hex/typeface source found; team hexes below are from memory, **unverified**). Motion facts are from motion.dev docs (fetched 2026-10-19).

## 0. Verified tech facts

- `motion` on npm: **14.4.0**, peer `react ^18 || ^19` (registry.npmjs.org/motion/latest). Works with Next 14 App Router + React 18. Import from `motion/react` (the `framer-motion` package is the same lib, older name). Any file using it needs `"use client"`.
- Bundle (motion.dev/docs/react-reduce-bundle-size): `motion.*` ~43 kB min; `m` + `LazyMotion` ~5 kB initial; `domAnimation` +24 kB (variants, exit, hover/tap); `domMax` +38 kB (adds **layout animations / layoutId**, drag). Import `m` from `motion/react-m`. `LazyMotion strict` throws if `motion.*` is used. Older docs/mirrors quote smaller numbers; treat as 5 to 45 kB range.
- `MotionConfig reducedMotion="user"` auto-disables transform + layout animations, keeps opacity/colour (motion.dev/docs/react-accessibility).
- `AnimatePresence`: direct child needs a stable unique `key`; `mode="wait"` supports one child; AnimatePresence must live OUTSIDE whatever unmounts the child (motion.dev/docs/react-animate-presence).
- WCAG: body text >= 4.5:1, large/UI >= 3:1. Pirelli 2026 compound colours unchanged: soft red, medium yellow, hard white, inter green, wet blue (tyretrade.ie Pirelli 2026 article).

## 1. Design spec

### Direction: "Pit wall instrument"
Dark-first, ink-black surfaces, hairline borders instead of shadows/cards, one signal accent, tabular mono numerals, condensed-to-wide display type. Reference feel: Linear's restraint + Vercel Geist's step-based scale + F1 timing tower density + Polymarket's big-percent-first hierarchy. Information first; decoration only where it encodes data (team colour bar, tyre colour, purple/green timing-screen semantics).

### Palette tokens (CSS variables; map into Tailwind via `hsl`/`oklch` theme extension, keep shadcn names so base-ui components keep working)
Surfaces use a 12-step ink scale (Geist/Radix idea: 1-2 bg, 3-5 component bg, 6-8 borders, 9-10 solid, 11-12 text). Tint everything very slightly cool (hue ~260, chroma 0.01) so it is not dead grey.

| Token | Dark | Light |
|---|---|---|
| `--bg` (= `--background`) | oklch(0.145 0.010 260) | oklch(0.985 0.004 90) |
| `--bg-raised` (= `--card`) | oklch(0.185 0.011 260) | oklch(1 0 0) |
| `--bg-sunken` | oklch(0.115 0.010 260) | oklch(0.955 0.005 90) |
| `--line` (= `--border`) | oklch(1 0 0 / 0.08) | oklch(0.2 0.01 260 / 0.10) |
| `--line-strong` | oklch(1 0 0 / 0.16) | oklch(0.2 0.01 260 / 0.20) |
| `--fg` (= `--foreground`) | oklch(0.965 0.005 260) | oklch(0.18 0.01 260) |
| `--fg-muted` (= `--muted-foreground`) | oklch(0.72 0.01 260) (>=4.5:1 on bg) | oklch(0.45 0.01 260) |
| `--fg-faint` (labels only, large/uppercase) | oklch(0.56 0.01 260) | oklch(0.58 0.01 260) |
| `--accent` (signal, single) | oklch(0.74 0.19 48) orange-flare | oklch(0.62 0.21 40) |
| `--accent-fg` | oklch(0.16 0.02 48) | oklch(0.99 0 0) |
| `--good` (green, "faster/improved") | oklch(0.78 0.19 150) | oklch(0.55 0.16 150) |
| `--best` (purple, "best/overall") | oklch(0.72 0.17 305) | oklch(0.52 0.19 305) |
| `--bad` | oklch(0.70 0.19 25) | oklch(0.55 0.21 25) |
| `--prob-fill` (bars) | `--fg` at 90% with team colour only on the 3px lead strip | same |

Tyre tokens (same both themes; add 1px `--line-strong` ring on hard): `--tyre-soft #FF3B30`, `--tyre-medium #FFD60A`, `--tyre-hard #F2F2F2`, `--tyre-inter #34C759`, `--tyre-wet #2F80FF`.
Team colours: store in a `teamColor` TS map, used ONLY as 3px lead strips, dots, line strokes. Approximate 2025 values from memory, **unverified, re-check for 2026 grid** (Audi and Cadillac entries are new): Mercedes #27F4D2, Ferrari #E8002D, McLaren #FF8000, Red Bull #3671C6, Williams #64C4FF, Aston #229971, Alpine #0093CC, Haas #B6BABD, Racing Bulls #6692FF. Never use team colour as text on dark without a contrast check (Haas grey, Williams light blue fine; Red Bull blue needs lightening for text).
Charts: `--chart-1..5` should be re-pointed to accent, best, good, fg-muted, line-strong; do not leave the grey shadcn defaults.

### Type (all free, `next/font/google`; set `display:"swap"`, expose as CSS vars)
- Display/headings: **Archivo** (variable, has `wdth` 62-125 axis: `axes:["wdth"]`; use ~75 wdth for condensed big numerals/headings, 100 for UI). Alternative: Barlow Condensed (static weights only).
- Body/UI: **Geist** (if the Next 14.2 font list lacks it, use the `geist` npm package; else Inter Tight). **Unverified** that `next/font/google` ships Geist in 14.2.x; check at install.
- Numerals/labels: **JetBrains Mono** (or Geist Mono), always `font-variant-numeric: tabular-nums` on any changing number.
- Scale (px / line-height): 11/16 label (mono, uppercase, +0.08em), 13/20 small, 15/24 body, 18/28 lead, 24/30 h3, 36/38 h2 (Archivo wdth 80, wght 700), 64-112/0.9 hero percentage (Archivo wdth 70-80, wght 800, tabular, -0.02em). Ratio ~1.5 at the top, tight leading on big numbers only.

### Spacing, radius, elevation
- 4px base; layout rhythm 8/12/16/24/40/64. Content max-width 1200, 12-col grid, 24 gutter; predict hero is asymmetric 7/5 (left-aligned, not centred).
- Radius: 2px for data cells/bars, 6px for controls, 10px for panels. No pill buttons except filter chips. Do not put radius 16+ on everything.
- Elevation: none by default. Panels = 1px `--line` border on `--bg-raised`. Only popovers/menus get shadow (`0 8px 30px oklch(0 0 0 / .45)` dark) plus `--line-strong`. A single faint top-edge highlight (inset 0 1px oklch(1 0 0 / .04)) on raised panels is fine. No glass blur except none.
- Background texture: optional 1px grid/dot field at 3-4% opacity, masked to fade out; static, not animated gradient blobs.

### Shared shell
```
[ APEX/ wordmark ]  Predict  Strategy ▔▔▔(layoutId underline)      [US GP · R18 · 24-25 Oct]  [theme]
---------------------------------------------------------------------------------------------
```
Sticky 48px bar, `--line` bottom border, backdrop none (solid `--bg`). Right side: race chip with live status dot (green = fresh, amber = `stale:true`). Footer: model/version line + disclaimer.

### /predict components and wireframe
Components: `RaceHeader`, `HeadlineBlocks` (podium), `ProbabilityTable` (ranked rows), `ModeBadge`, `StalenessBanner`, `EvidencePanel` (walk-forward), `HistoryTable`, `DriverDrawer` (row click: win/podium/top10 bars, exp_pos, p_dnf), `Disclaimer`.
```
UNITED STATES GP · CIRCUIT OF THE AMERICAS · 25 OCT        [PRE-QUALI | GRID ●]  updated 14:02
------------------------------------------------------------------------------------------
 MOST LIKELY WINNER                      │  PODIUM (most likely three)
 ANT  Kimi Antonelli                     │  ┌──────┐
 26.3%   <- 96px Archivo, ticker         │  │ HAM  │ ┌──────┐ ┌──────┐   podium blocks
 ▓▓▓▓▓▓░░░░░░  vs next: +7.1 pts         │  │ 2    │ │ ANT 1│ │ RUS 3│   heights ∝ p_podium
 "1 in 4 races like this"  (freq. framing)│  └──────┘ └──────┘ └──────┘   each shows p_podium%
------------------------------------------------------------------------------------------
 #  DRIVER         WIN              PODIUM        TOP 10     EXP POS   DNF
 1 ▌ANT Mercedes   ▓▓▓▓▓▓ 26.3%     57%           90%        3.1       8%
 2 ▌HAM Ferrari    ▓▓▓▓░░ 18.0%     ...
 ... (22 rows, sortable via header; sort change = row reorder with layout)
 [ Win | Podium | Top 10 ] segmented control sets which column drives the bars
------------------------------------------------------------------------------------------
 HOW MUCH TO TRUST THIS                               HOW IT DID THIS SEASON (walk-forward)
 log-loss 0.079 vs 0.094 baseline  [dumbbell]         Rd  Race     Pole  Actual  P(actual)  podium hit
 16 races, grid mode. Notes list.                     ... mini-strip: ● hit ○ miss, P(actual) bar
 "Probabilities, not picks. A 26% favourite loses 3 in 4."
```
Honest-uncertainty rules: show win % as number with frequency phrasing ("about 1 in 4"); never render winner as a certainty badge; show baseline next to model; label mode (pre-quali vs grid) prominently; stale banner when `stale:true`; optional 100-dot quantile dotplot (each dot = 1% of simulated races, coloured by driver) beside the hero, proven in election-forecast research (CHI 2024 "In Dice We Trust": text summaries and quantile dotplots earned most trust over time); round to whole % above 10, one decimal below; show "<1%" not "0%".

### /strategy components and wireframe
Components: `TrackPicker` (command-palette combobox: type to filter, keyboard nav, track silhouette + laps/pit-loss meta), `OptimalPlanCard` (stint strip + total time), `StintStrip`, `CompareBuilder` (add strategy rows: compound + stint length chips), `ResultsRange` (p10-p90 range plot), `DeltaToBest` lollipop list, `WinProbBar`, `Assumptions` collapsible (what the sim does not model), skeletons.
```
TRACK  [ Circuit of the Americas  ⌄ ⌘K ]    LAPS 56 · PIT LOSS 23.5s · FIELD n=41 races
------------------------------------------------------------------------------------------
 OPTIMAL PLAN                                           1-STOP · 5,412.8 s   (−3.1 s vs best manual)
 [■■■■■■■■■■■■ SOFT 18 ][■■■■■■■■■■■■■■■■■ HARD 38 ]   width ∝ laps, compound colour, lap tick marks
 pit L18                                                alt: 2-stop +1.9 s (within noise) 
------------------------------------------------------------------------------------------
 YOUR STRATEGIES                                            RESULT (p10 ──●p50── p90), delta to best
 A  [S 14][M 20][H 22]  ✎ ×                                 ├────●────┤   +0.0   best  (purple)
 B  [M 28][H 28]        ✎ ×                                 ├─────●──────┤ +4.2   P(beat A) 31%
 C  [S 20][H 36]                                            ├──────●───────┤ +6.8
 [+ add strategy]                                           shared x axis, ticks every 5 s, best-p50 vertical rule
 Draw bars: stint widths are editable by dragging pit boundaries (optional v2).
```
Rules: ranges overlap => say "within noise" instead of declaring a winner; show n runs; sim caveats (one car in isolation, no SC) in a visible assumptions line; `win_probability` language must match the module docstring scope note.

### Motion spec

Global: `<MotionConfig reducedMotion="user">`; wrap in `LazyMotion features={domMax}` only if layoutId/layout used (it is: tab underline, row reorder), else `domAnimation`. Use `m.*` + `strict`. Only animate `opacity`, `transform`, and bar `scaleX` (not width) to stay on the compositor.

| Element | What | Duration / curve | Trigger |
|---|---|---|---|
| Page enter | opacity 0->1, y 6->0 | 220 ms, ease-out `[0.22,1,0.36,1]` | route change (template.tsx, enter only) |
| Nav underline | `layoutId="nav-underline"` | spring stiffness 500, damping 38 | route/tab change |
| Hero % | count 0->value, tabular | 900 ms ease-out (animate() or useSpring stiffness 90, damping 20) | first view, and on data change |
| Ranked bars | scaleX 0->p (origin left) | 600 ms ease-out, stagger 25 ms/row, cap total 600 ms | first view (whileInView, once) |
| Table rows | stagger fade/y 8 | 250 ms, stagger 20 ms | mount; only first ~12 rows animate |
| Re-sort | `layout="position"` on rows | spring 400/40 | segmented control change |
| Podium blocks | height scaleY from bottom | 500 ms, delay 0.1/0.2/0 (2-1-3) | first view |
| Stint strip | segments scaleX sequentially left->right | 350 ms each, stagger 80 ms | plan changes (key on plan hash) |
| Range plot | whisker scaleX from p50 outward | 450 ms ease-out | results arrive |
| Skeleton | shimmer via CSS (not JS) background-position | 1.4 s linear loop | loading; skeleton has same dimensions as final (no layout shift) |
| Hover row | background to `--bg-raised`, team strip widens 3->6 px | 120 ms | pointer |
| Button press | scale 0.98 | spring 600/35 | whileTap |
| Drawer/popover | opacity + y 4 | 160 ms in, 100 ms out | open/close |
Rules: nothing loops except skeletons and the live dot; no animation > 1 s; stagger total <= 600 ms; no scroll-jacking; no parallax; animate on data change only when value really changed (compare before re-triggering).

## 2. framer-motion (motion 14) sketches

Provider (client component in layout):
```tsx
"use client";
import { LazyMotion, MotionConfig, domMax } from "motion/react";
export function Motion({ children }: { children: React.ReactNode }) {
  return (<MotionConfig reducedMotion="user"><LazyMotion features={domMax} strict>{children}</LazyMotion></MotionConfig>);
}
// elsewhere: import * as m from "motion/react-m";  <m.div .../>
```
Page transition. CAUTION: the popular `template.tsx` + `AnimatePresence mode="wait"` snippet (e.g. dev.to nayankyada article) puts AnimatePresence inside template, which itself remounts per navigation, so **exit animations will not run** (AnimatePresence docs: it must outlive the child). Reliable approach is enter-only:
```tsx
// app/template.tsx  (remounts every navigation)
"use client";
import * as m from "motion/react-m";
export default function Template({ children }: { children: React.ReactNode }) {
  return <m.div initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.22, ease: [0.22, 1, 0.36, 1] }}>{children}</m.div>;
}
```
Real exit animations need the FrozenRouter hack (freezes `LayoutRouterContext` from `next/dist/shared/lib/app-router-context.shared-runtime`): internal API, breaks across Next versions, **avoid**. Shared-element feel between pages instead: persistent header with `layoutId` underline in layout.tsx (outside template) plus enter-only content fade. Next 14 has no built-in View Transitions integration (**unverified** for 14.2; `viewTransition` config appeared later).

Stagger:
```tsx
const list = { show: { transition: { staggerChildren: 0.02, delayChildren: 0.05 } } };
const row = { hidden: { opacity: 0, y: 8 }, show: { opacity: 1, y: 0, transition: { duration: 0.25 } } };
<m.ul variants={list} initial="hidden" animate="show">{rows.map(r => <m.li key={r.code} variants={row} />)}</m.ul>
```
Number ticker (no re-render per frame):
```tsx
import { animate, useMotionValue, useTransform, useInView } from "motion/react";
function Ticker({ to, decimals = 1 }: { to: number; decimals?: number }) {
  const mv = useMotionValue(0), ref = useRef<HTMLSpanElement>(null);
  const text = useTransform(mv, v => (v * 100).toFixed(decimals) + "%");
  const inView = useInView(ref, { once: true });
  useEffect(() => { if (inView) { const c = animate(mv, to, { duration: 0.9, ease: [0.22, 1, 0.36, 1] }); return c.stop; } }, [inView, to]);
  return <m.span ref={ref} className="tabular-nums">{text}</m.span>;
}
```
(Render the final value in SSR markup, e.g. `initial` text = formatted `to`, to avoid hydration mismatch and no-JS blank.) `useSpring` is the alternative when value updates repeatedly.

layoutId tab underline: `{active && <m.span layoutId="tab-underline" className="absolute inset-x-0 -bottom-px h-px bg-[var(--accent)]" transition={{ type:"spring", stiffness:500, damping:38 }} />}` inside each tab; scope multiple tab groups with `<LayoutGroup id="nav">`.
Row reorder: sort the array, keep `key={code}`, put `layout="position"` on each row (`layout` alone can distort text). Limit to <= 25 rows.
whileInView: `<m.div initial={{opacity:0,y:12}} whileInView={{opacity:1,y:0}} viewport={{once:true, margin:"-80px"}} />`.

Avoid: `motion.*` everywhere without LazyMotion (43 kB); `layout` on big tables or on elements containing Recharts SVG (Recharts has its own animation: set `isAnimationActive={false}` and animate wrappers instead, or draw the bars/range plots as plain divs/SVG with `m` scaleX, which is cheaper and fully styleable); animating `width/height/left` (use scale/transform); reading `window` during render (hydration); initial animations on SSR content without `initial={false}` where it should be visible at first paint; `AnimatePresence` around lists keyed by index; two stacked animation systems (tw-animate-css + motion) on the same element.

## 3. Sources and takeaways

1. https://motion.dev/docs/react-reduce-bundle-size - m/LazyMotion sizes; layout needs domMax.
2. https://motion.dev/docs/react-animate-presence - key + outlive-the-child rule explains why template.tsx exits fail.
3. https://motion.dev/docs/react-accessibility - `MotionConfig reducedMotion="user"` kills transforms, keeps opacity.
4. https://registry.npmjs.org/motion/latest - motion 14.4.0, React 18/19 peer.
5. https://dev.to/nayankyada/how-i-use-framer-motion-with-nextjs-app-router-without-layout-shift-4378 - template.tsx pattern, `initial={false}`, reserve height to avoid CLS (but its exit claim is doubtful).
6. https://linear.app/now/how-we-redesigned-the-linear-ui - derive all surfaces from base/accent/contrast in perceptual (LCH) space; reduce visual noise; fewer, calmer chrome elements.
7. https://vercel.com/geist/colors - 10-step scale where step ranges have fixed jobs (100-300 bg, 400-600 borders, 900-1000 text); adopt the idea for our ink scale.
8. https://mucollective.northwestern.edu/?p=551 ("In Dice We Trust", CHI 2024) - text summaries and quantile dotplots keep forecast trust over time; use frequency framing.
9. https://statmodeling.stat.columbia.edu/2021/08/11/forecast-displays-that-emphasize-uncertainty/ - 538 ball-swarm / uncertainty-forward displays; no single viz captures all uncertainty, so pair number + dots + text.
10. https://humaninvariant.substack.com/p/novel-interface-designs-for-prediction (and https://avark.agency/learn/prediction-market-design-patterns) - Polymarket/Kalshi share one exchange-like pattern: big % badge, delta chip, timeframe pills, filled line chart; steal big-% hierarchy, avoid the trading-terminal clutter.
11. https://www.tyretrade.ie/index.php/pirelli-presents-2026-f1-tyres/40845 - 2026 compound colours unchanged (red/yellow/white/green/blue).
12. Not fetched (from memory, **unverified**): F1 TV/timing-screen convention purple = overall best, green = personal best/improvement; formula1.com uses a proprietary "Formula1 Display" face, hence Archivo/Barlow Condensed stand-ins; Awwwards/SiteInspire motorsport browsing was not done (low yield expected).

## 4. Don't list (vibecoded tells)

- Purple/indigo-to-blue gradients, gradient text, glowing blurred blobs, animated aurora backgrounds.
- Centred hero + subtitle + two buttons + row of three identical icon cards.
- Uniform grid of equal rounded-2xl cards with soft shadows everywhere; "card in a card in a card".
- Emoji or generic lucide icons as feature illustrations (trophy, flag, rocket); use data glyphs (team strips, tyre dots, sparkbars).
- Glassmorphism/backdrop-blur panels; neon glow borders.
- Pure default shadcn grey theme, Inter-only type, all text same weight; all-caps tracking on every label.
- Fake precision (26.314%), "AI-powered" copy, "Powered by XGBoost" badges, filler stats strips.
- Hero numbers that are not the product's number; confetti/trophy animation on the predicted winner; presenting the top pick as certain.
- Everything fades up on scroll; bouncy springs on data; infinite marquees; parallax.
- Pie/donut charts for win probability; rainbow team colours as fills for entire bars (use thin strips, keep fills neutral).
- Dark mode that is only `#000` with white borders; light mode that is an afterthought (test both).
- Lorem-ish empty states, spinners instead of dimensionally-matched skeletons.
