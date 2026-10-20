/**
 * 2026 team colours. Hexes are the "official" values in FastF1's 2026 plotting table
 * (fastf1/plotting/constants.json), matched to the team names /predict/next actually returns.
 * Used only as thin strips, dots and chart strokes, never as bar fills or text.
 */
const TEAM_COLORS: Record<string, string> = {
  mercedes: "#27f4d2",
  ferrari: "#e80020",
  mclaren: "#ff8000",
  "red bull": "#3671c6",
  williams: "#64c4ff",
  "aston martin": "#229971",
  alpine: "#0093cc",
  "haas": "#b6babd",
  rb: "#6692ff",
  "racing bulls": "#6692ff",
  audi: "#ff2d00",
  // FastF1 lists Cadillac as #444444, which vanishes on the dark theme; lightened for the strip only.
  cadillac: "#8a8f98",
};

const FALLBACK = "#6b7280";

/** API team strings look like "Alpine F1 Team", "RB F1 Team", "Haas F1 Team", "Cadillac F1 Team". */
export function teamColor(team: string): string {
  const t = team.toLowerCase().replace(/ f1 team$/, "").trim();
  if (TEAM_COLORS[t]) return TEAM_COLORS[t];
  const hit = Object.keys(TEAM_COLORS).find((k) => t.startsWith(k));
  return hit ? TEAM_COLORS[hit] : FALLBACK;
}

export function teamShort(team: string): string {
  return team.replace(/ F1 Team$/, "");
}
