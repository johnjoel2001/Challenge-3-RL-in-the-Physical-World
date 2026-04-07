/**
 * Scenario generation — ported from Python command_center.py
 * Generates geographic-scale drone interception scenarios over the Persian Gulf.
 */
import seedrandom from 'seedrandom';

// ── GEOGRAPHIC CONSTANTS ──
export const IRAN_LAUNCH_SITES = [
  { lat: 27.18, lon: 56.27, label: 'Bandar Abbas' },
  { lat: 26.55, lon: 54.35, label: 'Bandar Lengeh' },
  { lat: 28.97, lon: 50.84, label: 'Bushehr' },
  { lat: 25.64, lon: 57.78, label: 'Jask' },
  { lat: 27.50, lon: 56.90, label: 'Minab' },
  { lat: 27.19, lon: 56.22, label: 'Bandar-e Shahid Rajaee' },
  { lat: 30.28, lon: 48.30, label: 'Abadan' },
];

export const AIRBASES = {
  'Al Dhafra AB, UAE': { lat: 24.2481, lon: 54.5472 },
  'Camp Arifjan, Kuwait': { lat: 29.3417, lon: 47.9775 },
  'Prince Sultan AB, KSA': { lat: 24.0627, lon: 47.5802 },
};

const PHASE_RF = 0.50;
const PHASE_YOLO = 0.55;
const PHASE_KILL = 0.92;
export const TOTAL_FRAMES = 150;

function lerp(a, b, t) {
  return a + (b - a) * t;
}

/** Seeded normal distribution via Box-Muller */
function normalRandom(rng, mean = 0, std = 1) {
  const u1 = rng();
  const u2 = rng();
  const z = Math.sqrt(-2 * Math.log(u1 || 1e-10)) * Math.cos(2 * Math.PI * u2);
  return mean + z * std;
}

/**
 * Generate domain randomization parameters (mirrors core/domain_randomization.py).
 */
export function generateDomainRand(seed = 42) {
  const rng = seedrandom(seed + 7777);
  const uniform = (lo, hi) => lo + rng() * (hi - lo);
  const randInt = (lo, hi) => Math.floor(lo + rng() * (hi - lo));
  return {
    drone_mass: +uniform(0.7, 1.5).toFixed(3),
    max_force: +uniform(3.5, 7.0).toFixed(2),
    drag_coeff: +uniform(0.1, 0.6).toFixed(3),
    evader_speed: +uniform(1.0, 3.5).toFixed(2),
    num_obstacles: randInt(2, 9),
    obs_noise_std: +uniform(0, 0.05).toFixed(4),
    action_delay: randInt(0, 4),
    gravity: +uniform(9.75, 9.85).toFixed(3),
  };
}

/**
 * Generate a full scenario: adversary random-walk path from Iran to base,
 * interceptor smooth pursuit, phase transitions, telemetry.
 */
export function generateScenario(baseName, seed = 42) {
  const rng = seedrandom(seed);
  const uniform = (lo, hi) => lo + rng() * (hi - lo);
  const normal = (mean, std) => normalRandom(rng, mean, std);
  const randInt = (lo, hi) => Math.floor(lo + rng() * (hi - lo));

  const base = AIRBASES[baseName];
  const baseLat = base.lat;
  const baseLon = base.lon;

  const site = IRAN_LAUNCH_SITES[randInt(0, IRAN_LAUNCH_SITES.length)];
  const iranLat = site.lat;
  const iranLon = site.lon;
  const iranLabel = site.label;

  const rfT = PHASE_RF + uniform(-0.08, 0.05);
  const yoloT = PHASE_YOLO + uniform(-0.05, 0.05);
  const killT = PHASE_KILL + uniform(-0.04, 0.03);
  const driftScale = uniform(0.02, 0.06);

  // Random walk path for adversary
  let advLatC = iranLat;
  let advLonC = iranLon;
  const advPath = [];

  for (let i = 0; i < TOTAL_FRAMES; i++) {
    const t = i / (TOTAL_FRAMES - 1);
    const pull = 0.6 + 0.4 * t;
    const tgtLat = lerp(iranLat, baseLat, t);
    const tgtLon = lerp(iranLon, baseLon, t);
    advLatC += (tgtLat - advLatC) * pull * 0.15 + normal(0, driftScale) * (1 - 0.5 * t);
    advLonC += (tgtLon - advLonC) * pull * 0.15 + normal(0, driftScale) * (1 - 0.5 * t);
    advPath.push([advLatC, advLonC]);
  }

  // Converge last frames near base
  for (let i = Math.max(0, TOTAL_FRAMES - 8); i < TOTAL_FRAMES; i++) {
    const bl = (i - (TOTAL_FRAMES - 8)) / 8.0;
    const [la, lo] = advPath[i];
    advPath[i] = [lerp(la, baseLat + 0.015, bl), lerp(lo, baseLon + 0.015, bl)];
  }

  // Pre-compute the interception point (adversary position at kill frame)
  const killFrame = Math.min(TOTAL_FRAMES - 1, Math.floor(killT * (TOTAL_FRAMES - 1)));
  const [interceptLat, interceptLon] = advPath[killFrame];

  // Bezier control point: 40% along base→kill, with small perpendicular offset
  // This creates a gentle curve WITHOUT overshooting past the interception point
  const fracLat = baseLat + 0.4 * (interceptLat - baseLat);
  const fracLon = baseLon + 0.4 * (interceptLon - baseLon);
  // Small perpendicular offset for a natural pursuit arc (not a straight line)
  const dLat0 = interceptLat - baseLat;
  const dLon0 = interceptLon - baseLon;
  const perpLat = -dLon0;  // perpendicular direction
  const perpLon = dLat0;
  const perpScale = 0.06;  // small offset — visible curve, no overshoot
  const ctrlLat = fracLat + perpLat * perpScale;
  const ctrlLon = fracLon + perpLon * perpScale;

  const frames = [];
  for (let i = 0; i < TOTAL_FRAMES; i++) {
    const t = i / (TOTAL_FRAMES - 1);

    // After kill, freeze adversary at interception point
    let advLat, advLon;
    if (t >= killT) {
      advLat = interceptLat;
      advLon = interceptLon;
    } else {
      [advLat, advLon] = advPath[i];
    }

    const dLat = advLat - baseLat;
    const dLon = advLon - baseLon;
    const distKm = Math.sqrt(
      (dLat * 111) ** 2 + (dLon * 111 * Math.cos((baseLat * Math.PI) / 180)) ** 2
    );

    let phase, pn;
    if (t < rfT) { phase = 'CROSSING'; pn = 1; }
    else if (t < yoloT) { phase = 'RF DETECTED'; pn = 2; }
    else if (t < killT) { phase = 'PURSUING'; pn = 3; }
    else { phase = 'INTERCEPTED'; pn = 4; }

    // Interceptor: quadratic Bezier from base → ctrl (near adversary mid-pursuit) → kill point
    // Guarantees smooth forward-only motion with a natural pursuit curve
    let intLat, intLon;
    if (t < yoloT) {
      intLat = baseLat;
      intLon = baseLon;
    } else if (t <= killT) {
      const pt = (t - yoloT) / (killT - yoloT);
      // Quadratic Bezier: B(pt) = (1-pt)²·P0 + 2(1-pt)·pt·P1 + pt²·P2
      const a = (1 - pt) * (1 - pt);
      const b = 2 * (1 - pt) * pt;
      const c = pt * pt;
      intLat = a * baseLat + b * ctrlLat + c * interceptLat;
      intLon = a * baseLon + b * ctrlLon + c * interceptLon;
    } else {
      intLat = interceptLat;
      intLon = interceptLon;
    }

    let yc;
    if (pn <= 1) {
      yc = 0.0;
    } else if (pn === 2) {
      yc = +uniform(0.25, 0.50).toFixed(2);
    } else {
      yc = +Math.min(0.99, Math.max(0.50, 0.55 + (t - yoloT) * 3.0 + normal(0, 0.03))).toFixed(2);
    }

    const advAlt = 500 + normal(0, 20);
    const intAlt = t < yoloT ? 100 : lerp(100, advAlt, Math.min(1, (t - yoloT) / (killT - yoloT)));
    const brg = ((Math.atan2(dLon, dLat) * 180) / Math.PI + 360) % 360;

    frames.push({
      step: i + 1, t,
      advLat, advLon, advAlt,
      intLat, intLon, intAlt,
      distKm: +distKm.toFixed(1),
      bearing: +brg.toFixed(1),
      phase, pn, yc,
    });
  }

  return { frames, iranLabel, iranLat, iranLon };
}