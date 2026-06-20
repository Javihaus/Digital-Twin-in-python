// Build the technical deck: "Forecasting battery health you can trust".
// Audience: engineers (not battery specialists). Scientific lean.
// Run: node build_deck.js   (requires pptxgenjs reachable via NODE_PATH or local install)
const path = require("path");
const pptxgen = require("pptxgenjs");

const FIG = path.join(__dirname, "..", "figures");
const OUT = path.join(__dirname, "battery_soh_overview.pptx");

// palette (energy / grid)
const NAVY = "21295C", DEEP = "065A82", TEAL = "1C7293", MINT = "02C39A",
  AMBER = "E8A33D", LIGHT = "F4F7FA", INK = "1B2430", MUTE = "5B6B7B",
  CARD = "FFFFFF", LINEC = "C9D6E2";
const HEAD = "Georgia", BODY = "Calibri";

const pres = new pptxgen();
pres.defineLayout({ name: "W", width: 13.333, height: 7.5 });
pres.layout = "W";
pres.author = "Javier Marin";
pres.title = "Forecasting battery health you can trust";
const W = 13.333, H = 7.5;
const sh = () => ({ type: "outer", color: "000000", blur: 8, offset: 3, angle: 135, opacity: 0.15 });

// equation aspect ratios (width/height)
const EQR = { eq_prior: 10.987, eq_hybrid: 5.848, eq_phs: 10.279, eq_band: 10.624,
  eq_metrics: 5.87, eq_crps: 5.646 };

function accentBar(s) { s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 0.18, h: H, fill: { color: DEEP } }); }
function kicker(s, t) { s.addText(t.toUpperCase(), { x: 0.7, y: 0.45, w: 12, h: 0.4, fontFace: BODY, fontSize: 12.5, bold: true, color: TEAL, charSpacing: 3, margin: 0 }); }
function title(s, t, size) { s.addText(t, { x: 0.7, y: 0.8, w: 12.2, h: 0.95, fontFace: HEAD, fontSize: size || 28, bold: true, color: INK, margin: 0 }); }
function footer(s, n) {
  s.addText("Trustworthy AI for critical systems  ·  J. Marin", { x: 0.7, y: H - 0.42, w: 9, h: 0.3, fontFace: BODY, fontSize: 9, color: MUTE, margin: 0 });
  s.addText(String(n), { x: W - 0.9, y: H - 0.42, w: 0.5, h: 0.3, fontFace: BODY, fontSize: 9, color: MUTE, align: "right", margin: 0 });
}
function card(s, x, y, w, h, fill) { s.addShape(pres.shapes.RECTANGLE, { x, y, w, h, fill: { color: fill || CARD }, line: { color: LINEC, width: 1 }, shadow: sh() }); }
function imgFit(s, file, ow, oh, box, frame) {
  const r = ow / oh; let w = box.w, h = w / r;
  if (h > box.h) { h = box.h; w = h * r; }
  const x = box.x + (box.w - w) / 2, y = box.y + (box.h - h) / 2;
  if (frame !== false) s.addShape(pres.shapes.RECTANGLE, { x: x - 0.08, y: y - 0.08, w: w + 0.16, h: h + 0.16, fill: { color: CARD }, line: { color: "E3E9F0", width: 1 }, shadow: sh() });
  s.addImage({ path: path.join(FIG, file), x, y, w, h });
}
function eqImg(s, name, x, y, w) { const h = w / EQR[name]; s.addImage({ path: path.join(FIG, name + ".png"), x, y, w, h }); return h; }
function box(s, x, y, w, h, fill, line) { s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.06, fill: { color: fill }, line: { color: line || LINEC, width: 1.25 }, shadow: sh() }); }
function arrow(s, x1, y1, x2, y2, color) {
  s.addShape(pres.shapes.LINE, { x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1), h: Math.abs(y2 - y1),
    line: { color: color || MUTE, width: 2, endArrowType: "triangle", beginArrowType: "none" }, flipH: x2 < x1, flipV: y2 < y1 });
}

// ============ 1. TITLE ============
let s = pres.addSlide(); s.background = { color: NAVY };
s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: W, h: 0.25, fill: { color: MINT } });
s.addText("FORECASTING BATTERY HEALTH\nYOU CAN TRUST", { x: 0.9, y: 1.9, w: 11.5, h: 2.0, fontFace: HEAD, fontSize: 40, bold: true, color: "FFFFFF", lineSpacingMultiple: 1.05, margin: 0 });
s.addText("A physics-informed digital twin with calibrated uncertainty for grid-scale energy storage", { x: 0.9, y: 3.85, w: 11.4, h: 0.6, fontFace: BODY, fontSize: 18, color: "CADCFC", margin: 0 });
s.addText([{ text: "Javier Marin", options: { bold: true, color: "FFFFFF" } },
  { text: "   ·   State-of-Health & Remaining-Useful-Life forecasting with calibrated prediction intervals", options: { color: "9FB3C8" } }],
  { x: 0.9, y: 5.15, w: 11.5, h: 0.5, fontFace: BODY, fontSize: 14, margin: 0 });
s.addText("IEEE PES General Meeting 2026  ·  Panel: AI-powered Digital Twins for Grid-Scale Energy Storage", { x: 0.9, y: H - 0.7, w: 11.5, h: 0.4, fontFace: BODY, fontSize: 12, italic: true, color: MINT, margin: 0 });

// ============ 2. PROBLEM & TASK ============
s = pres.addSlide(); s.background = { color: LIGHT }; accentBar(s);
kicker(s, "Problem & task"); title(s, "Forecast capacity fade before it bites — with an error bar");
s.addText([
  { text: "State of Health  ", options: { bold: true } },
  { text: "SoH(n) = Qn / Q0", options: { italic: true } },
  { text: "  — fraction of original capacity at cycle n.  ", options: {} },
  { text: "End of life", options: { bold: true } },
  { text: " conventionally at SoH = 0.80.", options: {} },
], { x: 0.7, y: 1.75, w: 12, h: 0.5, fontFace: BODY, fontSize: 14.5, color: INK, margin: 0 });
card(s, 0.7, 2.4, 7.4, 3.6);
s.addText("The forecasting task", { x: 1.0, y: 2.6, w: 6.8, h: 0.45, fontFace: HEAD, fontSize: 17, bold: true, color: DEEP, margin: 0 });
s.addText([
  { text: "Given SoH for cycles 1…K (early life), predict SoH for cycles K+1…N (the rest of life) — and a calibrated interval for each.", options: { breakLine: true } },
  { text: "", options: { breakLine: true } },
  { text: "This is extrapolation, not interpolation: we never see the test cycles during fitting. It is the regime where naive models quietly fail and overconfident error bars become dangerous.", options: {} },
], { x: 1.0, y: 3.1, w: 6.85, h: 2.7, fontFace: BODY, fontSize: 14, color: INK, lineSpacingMultiple: 1.15, margin: 0 });
const why = [["Predictive maintenance", "Act before a unit drops offline."], ["Safety & warranty", "Stay ahead of thermal/limit risk."], ["No live SoH sensor", "It must be inferred and forecast."]];
why.forEach((p, i) => { const y = 2.4 + i * 1.22; card(s, 8.35, y, 4.25, 1.05);
  s.addShape(pres.shapes.RECTANGLE, { x: 8.35, y, w: 0.1, h: 1.05, fill: { color: TEAL } });
  s.addText(p[0], { x: 8.6, y: y + 0.12, w: 3.9, h: 0.4, fontFace: BODY, fontSize: 14, bold: true, color: DEEP, margin: 0 });
  s.addText(p[1], { x: 8.6, y: y + 0.52, w: 3.9, h: 0.45, fontFace: BODY, fontSize: 12, color: INK, margin: 0 }); });
footer(s, 2);

// ============ 3. ARCHITECTURE (schematic) ============
s = pres.addSlide(); s.background = { color: LIGHT }; accentBar(s);
kicker(s, "Architecture"); title(s, "The hybrid digital twin");
// boxes
box(s, 0.7, 3.0, 2.5, 1.3, "EAF2F6", DEEP);
s.addText("Measured history\nQ(n) per cycle", { x: 0.7, y: 3.0, w: 2.5, h: 1.3, align: "center", valign: "middle", fontFace: BODY, fontSize: 13, bold: true, color: INK, margin: 0 });
box(s, 4.0, 2.05, 3.3, 1.15, "E9F3EF", DEEP);
s.addText([{ text: "Physics prior  fθ(n)", options: { bold: true, color: DEEP, breakLine: true } }, { text: "port-Hamiltonian or empirical law", options: { fontSize: 11, color: MUTE } }], { x: 4.0, y: 2.05, w: 3.3, h: 1.15, align: "center", valign: "middle", fontFace: BODY, fontSize: 13, margin: 0 });
box(s, 4.0, 4.1, 3.3, 1.15, "FBF3E2", AMBER);
s.addText([{ text: "ML residual  gφ(n)", options: { bold: true, color: "8A5A12", breakLine: true } }, { text: "GP on what physics misses", options: { fontSize: 11, color: MUTE } }], { x: 4.0, y: 4.1, w: 3.3, h: 1.15, align: "center", valign: "middle", fontFace: BODY, fontSize: 13, margin: 0 });
box(s, 8.0, 3.0, 2.3, 1.3, "EAF2F6", TEAL);
s.addText([{ text: "Σ  →  SoH(n)", options: { bold: true, color: INK, breakLine: true } }, { text: "+ calibrated σ(h)", options: { fontSize: 11, color: MUTE } }], { x: 8.0, y: 3.0, w: 2.3, h: 1.3, align: "center", valign: "middle", fontFace: BODY, fontSize: 13, margin: 0 });
box(s, 11.0, 3.0, 1.9, 1.3, NAVY, NAVY);
s.addText("RUL to 80%\n+ decision", { x: 11.0, y: 3.0, w: 1.9, h: 1.3, align: "center", valign: "middle", fontFace: BODY, fontSize: 13, bold: true, color: "FFFFFF", margin: 0 });
// arrows
arrow(s, 3.2, 3.4, 4.0, 2.7, MUTE);
arrow(s, 3.2, 3.9, 4.0, 4.6, MUTE);
arrow(s, 7.3, 2.7, 8.0, 3.4, MUTE);
arrow(s, 7.3, 4.6, 8.0, 3.9, MUTE);
arrow(s, 10.3, 3.65, 11.0, 3.65, MUTE);
// feedback loop
s.addShape(pres.shapes.LINE, { x: 1.95, y: 3.65, w: 0, h: 1.95, line: { color: MINT, width: 2 } });
s.addShape(pres.shapes.LINE, { x: 1.95, y: 5.6, w: 7.2, h: 0, line: { color: MINT, width: 2 } });
s.addShape(pres.shapes.LINE, { x: 9.15, y: 4.3, w: 0, h: 1.3, line: { color: MINT, width: 2, beginArrowType: "triangle" } });
s.addText("new measurement → data assimilation (Kalman update)", { x: 2.1, y: 5.62, w: 7.0, h: 0.3, fontFace: BODY, fontSize: 11, italic: true, color: "0E8F70", margin: 0 });
s.addText("The physics block is a port-Hamiltonian model when conservation structure is known (the library's core), or a lighter empirical law for degradation like SoH — same hybrid pattern. The loop folds in new data as it arrives.",
  { x: 0.7, y: 6.2, w: 12, h: 0.7, fontFace: BODY, fontSize: 12.5, italic: true, color: MUTE, margin: 0 });
footer(s, 3);

// ============ 4. WHY HYBRID ============
s = pres.addSlide(); s.background = { color: LIGHT }; accentBar(s);
kicker(s, "Why hybrid"); title(s, "Neither physics nor ML alone is enough");
const hdr = (t) => ({ text: t, options: { bold: true, color: "FFFFFF", fill: { color: DEEP }, align: "center", valign: "middle" } });
const cellc = (t, c) => ({ text: t, options: { align: "center", valign: "middle", color: c || INK, bold: t === "Yes" } });
s.addTable([
  [hdr("Approach"), hdr("Extrapolates?"), hdr("Captures cell detail?"), hdr("Calibrated uncertainty?")],
  [{ text: "Physics only", options: { bold: true, valign: "middle" } }, cellc("Yes", "1B7A3D"), cellc("Limited", MUTE), cellc("No", "B23B3B")],
  [{ text: "ML only", options: { bold: true, valign: "middle" } }, cellc("No", "B23B3B"), cellc("Yes", "1B7A3D"), cellc("Partial", MUTE)],
  [{ text: "Hybrid (this work)", options: { bold: true, color: DEEP, valign: "middle" } }, cellc("Yes", "1B7A3D"), cellc("Yes", "1B7A3D"), cellc("Yes", "1B7A3D")],
], { x: 0.7, y: 1.85, w: 6.7, h: 2.7, fontFace: BODY, fontSize: 13.5, border: { type: "solid", pt: 1, color: "D7E0EA" }, fill: { color: "FFFFFF" }, rowH: [0.7, 0.55, 0.55, 0.6] });
s.addText([
  { text: "Physics", options: { bold: true, color: DEEP } },
  { text: " constrains the forecast to stay physically admissible far ahead; ", options: {} },
  { text: "ML", options: { bold: true, color: "8A5A12" } },
  { text: " corrects what the law misses but cannot extrapolate on its own; the ", options: {} },
  { text: "hybrid", options: { bold: true, color: TEAL } },
  { text: " keeps both — and only it carries a calibrated error bar. The bars at right show it in numbers.", options: {} },
], { x: 0.7, y: 4.7, w: 6.7, h: 1.6, fontFace: BODY, fontSize: 13.5, color: INK, lineSpacingMultiple: 1.15, margin: 0 });
imgFit(s, "02_method_comparison.png", 1400, 900, { x: 7.7, y: 1.9, w: 5.2, h: 4.7 });
footer(s, 4);

// ============ 5. EQUATIONS: MODEL ============
s = pres.addSlide(); s.background = { color: LIGHT }; accentBar(s);
kicker(s, "The model"); title(s, "Structured prior + learned correction");
let y = 1.9;
s.addText("1 · Empirical degradation prior  (SEI film growth ~√n, cycling ~ linear)", { x: 0.7, y, w: 12, h: 0.35, fontFace: BODY, fontSize: 13.5, bold: true, color: DEEP, margin: 0 });
eqImg(s, "eq_prior", 1.2, y + 0.4, 10.9); y += 0.4 + 10.9 / EQR.eq_prior + 0.25;
s.addText("2 · Hybrid forecast = physics prior + damped ML residual   (fθ: a fade law here; a port-Hamiltonian model when structure is known)", { x: 0.7, y, w: 12, h: 0.35, fontFace: BODY, fontSize: 13, bold: true, color: DEEP, margin: 0 });
eqImg(s, "eq_hybrid", 2.6, y + 0.4, 6.0); y += 0.4 + 6.0 / EQR.eq_hybrid + 0.25;
s.addText("3 · Structure-preserving core of the library (passive by construction → no runaway)", { x: 0.7, y, w: 12, h: 0.35, fontFace: BODY, fontSize: 13.5, bold: true, color: DEEP, margin: 0 });
eqImg(s, "eq_phs", 1.4, y + 0.4, 10.5);
footer(s, 5);

// ============ 6. EQUATIONS: UNCERTAINTY ============
s = pres.addSlide(); s.background = { color: LIGHT }; accentBar(s);
kicker(s, "Uncertainty & scoring"); title(s, "Calibrated bands and how we grade them");
y = 1.95;
s.addText("Calibrated interval — split-conformal scale that grows with the horizon h", { x: 0.7, y, w: 12, h: 0.35, fontFace: BODY, fontSize: 13.5, bold: true, color: DEEP, margin: 0 });
eqImg(s, "eq_band", 1.3, y + 0.4, 10.6); y += 0.4 + 10.6 / EQR.eq_band + 0.35;
s.addText("Graded against a naive baseline — skill, and whether the bands mean what they say", { x: 0.7, y, w: 12, h: 0.35, fontFace: BODY, fontSize: 13.5, bold: true, color: DEEP, margin: 0 });
eqImg(s, "eq_metrics", 1.0, y + 0.45, 7.4);
eqImg(s, "eq_crps", 8.7, y + 0.55, 3.9);
s.addText("PICP = coverage · MASE < 1 beats the naive forecast · CRPS scores the whole predictive distribution",
  { x: 0.7, y: y + 0.45 + 7.4 / EQR.eq_metrics + 0.2, w: 12, h: 0.4, fontFace: BODY, fontSize: 12, italic: true, color: MUTE, margin: 0 });
footer(s, 6);

// ============ 7. DATA & CURATION ============
s = pres.addSlide(); s.background = { color: LIGHT }; accentBar(s);
kicker(s, "Data & curation"); title(s, "Rigor starts before the model");
imgFit(s, "00_fleet_overview.png", 1600, 1000, { x: 6.5, y: 1.7, w: 6.4, h: 5.0 });
s.addText([
  { text: "NASA Li-ion dataset (public, Saha & Goebel).", options: { bold: true, breakLine: true } },
  { text: "Of 34 cells, 8 are genuine full-discharge degradation trajectories. The other 26 are reference / impedance / partial-cycle tests — SoH that 'grows', collapses and revives, or starts near zero.", options: { breakLine: true } },
  { text: "", options: { breakLine: true } },
  { text: "Inclusion criteria are explicit:", options: { bold: true, color: DEEP, breakLine: true } },
  { text: "≥ 40 cycles · plausible C0 · monotone-ish fade · no recovery glitches. Two operating temperatures (24 °C, 4 °C) are kept on purpose, to test calibration across conditions.", options: {} },
], { x: 0.7, y: 1.9, w: 5.5, h: 4.7, fontFace: BODY, fontSize: 14, color: INK, lineSpacingMultiple: 1.12, margin: 0 });
footer(s, 7);

// ============ 8. FORECAST ============
s = pres.addSlide(); s.background = { color: LIGHT }; accentBar(s);
kicker(s, "Result · forecast"); title(s, "Forecast the second half of life from the first");
imgFit(s, "01_hero_forecast.png", 1600, 1000, { x: 0.6, y: 1.7, w: 7.8, h: 5.2 });
[["Tracks the true fade", "Trained only left of the split line, the hybrid follows the real decline."],
 ["ML-only drifts off", "Without physics it flattens and misses the trend."],
 ["Band widens with horizon", "Uncertainty grows the further ahead — and keeps containing the truth."]].forEach((p, i) => {
  const yy = 1.95 + i * 1.65; s.addShape(pres.shapes.OVAL, { x: 8.7, y: yy + 0.04, w: 0.26, h: 0.26, fill: { color: MINT } });
  s.addText(p[0], { x: 9.05, y: yy, w: 3.9, h: 0.45, fontFace: HEAD, fontSize: 14.5, bold: true, color: DEEP, margin: 0 });
  s.addText(p[1], { x: 9.05, y: yy + 0.48, w: 3.9, h: 1.05, fontFace: BODY, fontSize: 12.5, color: INK, margin: 0 }); });
footer(s, 8);

// ============ 9. HORIZON ZOOM ============
s = pres.addSlide(); s.background = { color: LIGHT }; accentBar(s);
kicker(s, "Result · by horizon"); title(s, "Short, medium, long term — resolved");
imgFit(s, "10_zoom_horizons.png", 3000, 919, { x: 0.6, y: 2.0, w: 12.1, h: 3.6 });
s.addText([
  { text: "Short term: ", options: { bold: true, color: DEEP } }, { text: "tight band, the correction still informs the mean.   ", options: {} },
  { text: "Medium: ", options: { bold: true, color: DEEP } }, { text: "the physics trend carries; band opens steadily.   ", options: {} },
  { text: "Long: ", options: { bold: true, color: DEEP } }, { text: "wide but earned — the interval owns the growing risk.", options: {} },
], { x: 0.7, y: 5.9, w: 12, h: 0.8, fontFace: BODY, fontSize: 13.5, color: INK, lineSpacingMultiple: 1.15, margin: 0 });
footer(s, 9);

// ============ 10. CALIBRATION + RESIDUALS ============
s = pres.addSlide(); s.background = { color: LIGHT }; accentBar(s);
kicker(s, "Result · is the uncertainty real?"); title(s, "Calibration and predictive-distribution checks");
imgFit(s, "03_calibration.png", 1100, 1100, { x: 0.7, y: 1.9, w: 4.4, h: 4.7 });
imgFit(s, "11_residual_diag.png", 2400, 960, { x: 5.4, y: 2.4, w: 7.4, h: 3.6 });
s.addText([
  { text: "Left: ", options: { bold: true, color: DEEP } },
  { text: "a stated 90% interval contains the truth ~90% of the time across the fleet (ECE = 0.11), sitting just above the diagonal — slightly conservative. ", options: {} },
  { text: "Right: ", options: { bold: true, color: DEEP } },
  { text: "standardized residuals are close to N(0,1), a touch narrower — the bands are trustworthy, with a little room to tighten.", options: {} },
], { x: 0.7, y: 6.2, w: 12, h: 0.9, fontFace: BODY, fontSize: 12.5, color: INK, lineSpacingMultiple: 1.12, margin: 0 });
footer(s, 10);

// ============ 11. RUL ============
s = pres.addSlide(); s.background = { color: LIGHT }; accentBar(s);
kicker(s, "Result · the decision"); title(s, "From curve to maintenance call: Remaining Useful Life");
imgFit(s, "06_rul_hero.png", 1600, 1000, { x: 0.6, y: 1.8, w: 6.3, h: 4.8 });
imgFit(s, "07_rul_parity.png", 1100, 1100, { x: 7.2, y: 1.8, w: 5.0, h: 4.8 });
s.addText("RUL = cycles to 80% SoH. The twin returns a distribution of end-of-life cycles (left); across cells the predicted EOL tracks the true EOL with an interval you can schedule against (right).",
  { x: 0.7, y: 6.55, w: 12, h: 0.6, fontFace: BODY, fontSize: 12.5, italic: true, color: MUTE, margin: 0 });
footer(s, 11);

// ============ 12. BEYOND BATTERIES ============
s = pres.addSlide(); s.background = { color: LIGHT }; accentBar(s);
kicker(s, "Generalization"); title(s, "The same engine, across utilities");
[["Energy storage", "SoH / RUL for grid batteries — shown here."], ["Water networks", "Pump wear, membrane fouling, leak-rate drift."], ["Rotating & static assets", "Transformers, motors, heat exchangers."]].forEach((p, i) => {
  const x = 0.7 + i * 4.05; card(s, x, 2.0, 3.8, 2.3);
  s.addShape(pres.shapes.RECTANGLE, { x, y: 2.0, w: 3.8, h: 0.12, fill: { color: MINT } });
  s.addText(p[0], { x: x + 0.25, y: 2.3, w: 3.3, h: 0.55, fontFace: HEAD, fontSize: 16, bold: true, color: DEEP, margin: 0 });
  s.addText(p[1], { x: x + 0.25, y: 2.9, w: 3.35, h: 1.2, fontFace: BODY, fontSize: 13, color: INK, margin: 0 }); });
card(s, 0.7, 4.7, 11.9, 1.9, NAVY);
s.addText("What makes it deployable", { x: 1.0, y: 4.9, w: 11, h: 0.45, fontFace: HEAD, fontSize: 16, bold: true, color: MINT, margin: 0 });
s.addText([
  { text: "Lightweight & CPU-only", options: { bold: true, color: "FFFFFF" } }, { text: " (laptop, no cluster)    ", options: { color: "CADCFC" } },
  { text: "Auditable & deterministic", options: { bold: true, color: "FFFFFF" } }, { text: " (traceable — EU AI Act-ready)    ", options: { color: "CADCFC" } },
  { text: "Calibrated by default", options: { bold: true, color: "FFFFFF" } }, { text: " (plan against the interval)", options: { color: "CADCFC" } },
], { x: 1.0, y: 5.4, w: 11.4, h: 1.0, fontFace: BODY, fontSize: 13.5, lineSpacingMultiple: 1.2, margin: 0 });
footer(s, 12);

// ============ 13. WHERE THIS SITS (world models) ============
s = pres.addSlide(); s.background = { color: LIGHT }; accentBar(s);
kicker(s, "Broader context"); title(s, "Where this sits: the world-models family");
card(s, 0.7, 1.95, 5.9, 3.7);
s.addShape(pres.shapes.RECTANGLE, { x: 0.7, y: 1.95, w: 5.9, h: 0.12, fill: { color: DEEP } });
s.addText("This work — structured digital twin", { x: 1.0, y: 2.2, w: 5.3, h: 0.5, fontFace: HEAD, fontSize: 16, bold: true, color: DEEP, margin: 0 });
s.addText([
  { text: "State:  observable, physical (SoH)", options: { breakLine: true } },
  { text: "Structure:  known a priori (physics)", options: { breakLine: true } },
  { text: "Uncertainty:  calibrated, first-class", options: { breakLine: true } },
  { text: "Nature:  white-box, auditable, CPU", options: {} },
], { x: 1.0, y: 2.8, w: 5.4, h: 2.6, fontFace: BODY, fontSize: 14, color: INK, lineSpacingMultiple: 1.35, margin: 0 });
card(s, 6.9, 1.95, 5.7, 3.7);
s.addShape(pres.shapes.RECTANGLE, { x: 6.9, y: 1.95, w: 5.7, h: 0.12, fill: { color: MUTE } });
s.addText("ML world models (Dreamer, JEPA, Cosmos)", { x: 7.2, y: 2.2, w: 5.2, h: 0.5, fontFace: HEAD, fontSize: 16, bold: true, color: MUTE, margin: 0 });
s.addText([
  { text: "State:  learned latent representation", options: { breakLine: true } },
  { text: "Structure:  learned from data", options: { breakLine: true } },
  { text: "Uncertainty:  typically weak", options: { breakLine: true } },
  { text: "Nature:  black-box, high-dim, GPU", options: {} },
], { x: 7.2, y: 2.8, w: 5.2, h: 2.6, fontFace: BODY, fontSize: 14, color: INK, lineSpacingMultiple: 1.35, margin: 0 });
card(s, 0.7, 5.85, 11.9, 1.0, NAVY);
s.addText([
  { text: "Same substrate", options: { bold: true, color: MINT } },
  { text: " — state, conservation, dissipation, coupling to outside — at ", options: { color: "CADCFC" } },
  { text: "opposite ends of the observable ↔ latent axis.", options: { bold: true, color: "FFFFFF" } },
  { text: "  Critical-infrastructure decisions need the structured, auditable end.", options: { color: "CADCFC" } },
], { x: 1.0, y: 6.0, w: 11.4, h: 0.7, fontFace: BODY, fontSize: 13.5, valign: "middle", lineSpacingMultiple: 1.1, margin: 0 });
footer(s, 13);

// ============ 14. REFERENCES ============
s = pres.addSlide(); s.background = { color: LIGHT }; accentBar(s);
kicker(s, "Selected references"); title(s, "Where the modelling choices come from");
const refL = [
  ["Data", "Saha & Goebel (2007), NASA Ames Prognostics Data Repository — Li-ion battery data set."],
  ["Degradation physics", "Pinson & Bazant (2013), J. Electrochem. Soc. — SEI growth & √t capacity fade."],
  ["", "Birkl et al. (2017), J. Power Sources — degradation diagnostics for Li-ion cells."],
  ["", "Xu et al. (2018), IEEE Trans. Smart Grid — cell-life degradation modelling."],
  ["Data-driven life", "Severson et al. (2019), Nature Energy — cycle-life prediction before fade."],
  ["GP for SoH / UQ", "Richardson, Osborne & Howey (2017), J. Power Sources — GP regression for SoH."],
  ["", "Rasmussen & Williams (2006), Gaussian Processes for Machine Learning, MIT Press."],
];
const refR = [
  ["Physics-informed ML", "Karniadakis et al. (2021), Nature Reviews Physics — physics-informed ML."],
  ["", "Willard et al. (2023), ACM Computing Surveys — knowledge + ML for science/eng."],
  ["Conformal & calibration", "Vovk, Gammerman & Shafer (2005), Algorithmic Learning in a Random World."],
  ["", "Angelopoulos & Bates (2023) — gentle introduction to conformal prediction."],
  ["", "Kuleshov, Fenner & Ermon (2018), ICML — calibrated regression."],
  ["Scoring & accuracy", "Gneiting & Raftery (2007), JASA — proper scoring rules (CRPS)."],
  ["", "Hyndman & Koehler (2006), Int. J. Forecasting — MASE."],
];
function refCol(arr, x) {
  let yy = 1.85;
  arr.forEach((r) => {
    if (r[0]) { s.addText(r[0], { x, y: yy, w: 5.8, h: 0.3, fontFace: BODY, fontSize: 12, bold: true, color: TEAL, margin: 0 }); yy += 0.32; }
    s.addText(r[1], { x: x + 0.15, y: yy, w: 5.7, h: 0.55, fontFace: BODY, fontSize: 11.5, color: INK, margin: 0 }); yy += 0.55;
  });
}
refCol(refL, 0.7); refCol(refR, 7.0);
s.addText("Port-Hamiltonian core: van der Schaft & Jeltsema (2014), Foundations and Trends in Systems and Control.",
  { x: 0.7, y: 6.7, w: 12, h: 0.35, fontFace: BODY, fontSize: 11, italic: true, color: MUTE, margin: 0 });
footer(s, 14);

// ============ 14. TAKEAWAYS ============
s = pres.addSlide(); s.background = { color: NAVY };
s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: W, h: 0.25, fill: { color: MINT } });
s.addText("WHAT TO TAKE AWAY", { x: 0.9, y: 0.8, w: 11, h: 0.4, fontFace: BODY, fontSize: 13, bold: true, color: MINT, charSpacing: 3, margin: 0 });
s.addText([
  { text: "Forecast the health and the error bar together.", options: { bold: true, color: "FFFFFF", breakLine: true } },
  { text: "In a critical system, a point forecast without calibrated uncertainty is a liability.", options: { color: "CADCFC", breakLine: true } },
  { text: "", options: { breakLine: true } },
  { text: "Structure extrapolates; data-fitting interpolates.", options: { bold: true, color: "FFFFFF", breakLine: true } },
  { text: "Physics + ML beats either alone, and beats the naive baseline by 3-8× on the standard cells.", options: { color: "CADCFC", breakLine: true } },
  { text: "", options: { breakLine: true } },
  { text: "Rigor is the product: curated data, extrapolative evaluation, checked calibration.", options: { bold: true, color: "FFFFFF", breakLine: true } },
  { text: "The same engine carries from batteries to water to rotating assets.", options: { color: "CADCFC" } },
], { x: 0.9, y: 1.55, w: 11.6, h: 3.6, fontFace: BODY, fontSize: 16.5, lineSpacingMultiple: 1.12, margin: 0 });
s.addShape(pres.shapes.LINE, { x: 0.9, y: 5.5, w: 11.5, h: 0, line: { color: "3A4673", width: 1 } });
s.addText([{ text: "Javier Marin", options: { bold: true, color: "FFFFFF" } },
  { text: "  ·  I build the math that tells you when AI is wrong  ·  ", options: { color: "9FB3C8" } },
  { text: "javier@jmarin.info  ·  github.com/Javihaus", options: { color: MINT } }],
  { x: 0.9, y: 5.75, w: 11.6, h: 0.5, fontFace: BODY, fontSize: 13, margin: 0 });

pres.writeFile({ fileName: OUT }).then(() => console.log("WROTE", OUT));
