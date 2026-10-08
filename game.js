import {
  FilesetResolver,
  HandLandmarker,
  FaceLandmarker,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

const WASM_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
const HAND_MODEL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";
const FACE_MODEL =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";

const EMOJI_FONT = '"Apple Color Emoji","Segoe UI Emoji","Noto Color Emoji","Twemoji Mozilla",sans-serif';

// drink: sipped (level bar, no bite marks). soft: bitten, but slurpy instead of crunchy.
const FOODS = [
  { id: "burger", emoji: "🍔", bites: 4, colors: ["#c8873a", "#f2c14e", "#6aa84f", "#d9472b", "#7a4a21"] },
  { id: "sandwich", emoji: "🥪", bites: 4, colors: ["#f3d9a4", "#e8b86d", "#7cc36b", "#f6e27a", "#d96b5f"] },
  { id: "pizza", emoji: "🍕", bites: 4, colors: ["#f6c453", "#e8a33d", "#d7402b", "#fff3c4", "#6aa84f"] },
  { id: "coke", emoji: "🥤", bites: 5, drink: true, colors: ["#ffffff", "#f8e9e0", "#d7b9a8", "#9fd3ff"] },
  // No emoji for a milk pouch, so this one is drawn by hand (emoji is used for the score).
  { id: "nandini", emoji: "🥛", draw: drawNandiniPacket, bites: 4, drink: true, words: ["MOO!", "SLURP!", "GLUG", "AHH!"], colors: ["#ffffff", "#f4f8ff", "#d6e6ff"] },
];

// A Nandini-style blue milk pouch: crimped top seal, bulging sides, white wave band.
function drawNandiniPacket(g, sz, S) {
  const cx = S / 2, cy = S / 2 + sz * 0.03;
  const w = sz * 0.62, h = sz * 0.9;
  const x = cx - w / 2, y = cy - h / 2;

  const body = new Path2D();
  body.moveTo(x + w * 0.05, y + h * 0.1);
  body.lineTo(x + w * 0.95, y + h * 0.1);
  body.quadraticCurveTo(x + w * 1.07, y + h * 0.55, x + w * 0.95, y + h * 0.95);
  body.quadraticCurveTo(cx, y + h * 1.02, x + w * 0.05, y + h * 0.95);
  body.quadraticCurveTo(x - w * 0.07, y + h * 0.55, x + w * 0.05, y + h * 0.1);
  const grad = g.createLinearGradient(x, 0, x + w, 0);
  grad.addColorStop(0, "#1b54b0");
  grad.addColorStop(0.4, "#3f8ff0");
  grad.addColorStop(1, "#163f8c");
  g.fillStyle = grad;
  g.fill(body); // the drop shadow (set by the caller) applies to the pouch outline only
  g.shadowColor = "transparent";

  g.save();
  g.clip(body);
  // White wave band across the middle.
  g.fillStyle = "#fff";
  g.beginPath();
  g.moveTo(x - 2, y + h * 0.5);
  g.bezierCurveTo(x + w * 0.3, y + h * 0.42, x + w * 0.7, y + h * 0.58, x + w + 2, y + h * 0.48);
  g.lineTo(x + w + 2, y + h * 0.66);
  g.bezierCurveTo(x + w * 0.7, y + h * 0.76, x + w * 0.3, y + h * 0.6, x - 2, y + h * 0.68);
  g.closePath();
  g.fill();
  // Shine.
  g.fillStyle = "rgba(255,255,255,0.22)";
  g.beginPath();
  g.ellipse(x + w * 0.22, y + h * 0.36, w * 0.06, h * 0.2, -0.1, 0, Math.PI * 2);
  g.fill();
  g.restore();

  // Crimped top seal.
  g.fillStyle = "#163f8c";
  g.fillRect(x + w * 0.05, y, w * 0.9, h * 0.11);
  g.strokeStyle = "rgba(255,255,255,0.45)";
  g.lineWidth = Math.max(1, sz * 0.008);
  for (let i = 1; i < 12; i++) {
    const lx = x + w * 0.05 + (w * 0.9 * i) / 12;
    g.beginPath();
    g.moveTo(lx, y + h * 0.01);
    g.lineTo(lx, y + h * 0.1);
    g.stroke();
  }

  g.textAlign = "center";
  g.textBaseline = "middle";
  g.fillStyle = "#fff";
  g.font = `italic 900 ${sz * 0.15}px Georgia, "Times New Roman", serif`;
  g.fillText("Nandini", cx, y + h * 0.3);
  g.fillStyle = "#1b54b0";
  g.font = `800 ${sz * 0.1}px system-ui, sans-serif`;
  g.fillText("MILK", cx, y + h * 0.58);
  g.font = `${sz * 0.17}px ${EMOJI_FONT}`;
  g.fillText("🐄", cx, y + h * 0.82);
}

const HELD_SCALE = 1.4; // snacks grow a bit when picked up so they read well next to your face
const HOVER_GRAB = 0.35; // seconds your hand must rest on a snack to pick it up

const BITE_WORDS = ["CHOMP!", "NOM", "MUNCH!", "YUM!", "CRUNCH!"];
const SIP_WORDS = ["SIP SIP!", "SLURP!", "GLUG", "SIP!", "AHH!"];

const canvas = document.getElementById("stage");
const ctx = canvas.getContext("2d");
const video = document.getElementById("cam");
const overlay = document.getElementById("overlay");
const hud = document.getElementById("hud");
const startBtn = document.getElementById("start");
const msgEl = document.getElementById("msg");
const statusEl = document.getElementById("status");

let W = 0, H = 0, dpr = 1;
let vr = { s: 1, ox: 0, oy: 0, vw: 1, vh: 1 };
let handLm = null, faceLm = null;

const SIDES = ["left", "right"];

const state = {
  // The full menu on both sides of the screen, so players on either side can reach every snack.
  items: SIDES.flatMap((side) => FOODS.map((def, i) => ({
    def, i, side,
    x: 0, y: 0, home: { x: 0, y: 0 }, size: 100,
    state: "shelf", scale: 1, rot: 0, squash: 0,
    taken: 0, bites: [], biteAngle: null,
    holder: null, nearTime: 0, primed: false, lastBite: 0, respawnAt: 0,
    sprite: null, vx: 0,
  }))),
  hands: new Map(),
  faces: [], // everyone in view; each snack goes to the mouth nearest the hand holding it
  particles: [],
  texts: [],
  score: Object.fromEntries(FOODS.map((f) => [f.id, 0])),
  shake: 0,
  debug: false,
  size: 100,
  lastVideoTime: -1,
  lastDetect: 0,
  lastHandSeen: 0,
  lastFrame: performance.now(),
};

// ---------- math helpers ----------
const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
const mid = (a, b) => ({ x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 });
const avg = (pts) => ({
  x: pts.reduce((s, p) => s + p.x, 0) / pts.length,
  y: pts.reduce((s, p) => s + p.y, 0) / pts.length,
});
const lerp = (a, b, t) => a + (b - a) * t;
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const ease = (dt, speed) => 1 - Math.exp(-dt * speed);
const pick = (arr) => arr[Math.floor(Math.random() * arr.length)];

// Landmark (normalized, un-mirrored video coords) -> mirrored screen coords.
const toScreen = (lm) => ({ x: vr.ox + (1 - lm.x) * vr.vw * vr.s, y: vr.oy + lm.y * vr.vh * vr.s });

function updateVideoRect() {
  const vw = video.videoWidth || 1280, vh = video.videoHeight || 720;
  const s = Math.max(W / vw, H / vh);
  vr = { s, vw, vh, ox: (W - vw * s) / 2, oy: (H - vh * s) / 2 };
}

// ---------- layout & sprites ----------
function layout() {
  dpr = Math.min(window.devicePixelRatio || 1, 1.5);
  W = window.innerWidth;
  H = window.innerHeight;
  canvas.width = Math.round(W * dpr);
  canvas.height = Math.round(H * dpr);
  canvas.style.width = W + "px";
  canvas.style.height = H + "px";
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  const n = FOODS.length;
  // One column down each side, sized to fit between the top buttons and the bottom hint.
  const avail = H - 2 * (W < 600 ? 130 : 80);
  const size = Math.round(Math.max(44, Math.min(clamp(Math.min(W, H) * 0.15, 54, 130), avail / (n * 1.22))));
  state.size = size;
  const cell = size * 1.22;
  const top = H / 2 - (cell * (n - 1)) / 2;
  // Not flush with the edge: a hand reaching the very edge of the camera view gets cut off and lost.
  const left = Math.max(14 + size * 0.65, W * 0.09);
  // The menu strips (plus some slack): any part of a hand inside one can grab from it.
  const y0 = top - size * 0.9, y1 = top + (n - 1) * cell + size * 0.9;
  state.menus = {
    left: { x0: -Infinity, x1: left + size * 1.9, y0, y1 },
    right: { x0: W - left - size * 1.9, x1: Infinity, y0, y1 },
  };
  for (const it of state.items) {
    it.size = size;
    it.home = { x: it.side === "left" ? left : W - left, y: top + it.i * cell };
    if (it.state === "shelf" || it.state === "spawning") { it.x = it.home.x; it.y = it.home.y; }
    rebuildSprite(it);
  }
  updateVideoRect();
}

function rebuildSprite(it) {
  const S = it.size * 1.25;
  const q = dpr * HELD_SCALE;
  const c = it.sprite || document.createElement("canvas");
  c.width = c.height = Math.ceil(S * q);
  const g = c.getContext("2d");
  g.setTransform(q, 0, 0, q, 0, 0);
  g.font = `${it.size}px ${EMOJI_FONT}`;
  g.textAlign = "center";
  g.textBaseline = "middle";
  g.shadowColor = "rgba(0,0,0,0.35)";
  g.shadowBlur = 10 * q;
  g.shadowOffsetY = 5 * q;
  if (it.def.draw) it.def.draw(g, it.size, S);
  else g.fillText(it.def.emoji, S / 2, S / 2 + it.size * 0.06);
  g.shadowColor = "transparent";
  for (const b of it.bites) cutBite(g, it, b);
  it.sprite = c;
}

// Punch a bite-shaped hole into the sprite, deeper with each bite.
function cutBite(g, it, { angle, k }) {
  const sz = it.size, c = (sz * 1.25) / 2;
  const d = sz * 0.5 - k * sz * 0.16;
  const cx = c + Math.cos(angle) * d, cy = c + Math.sin(angle) * d;
  const px = -Math.sin(angle), py = Math.cos(angle);
  g.save();
  g.globalCompositeOperation = "destination-out";
  for (let i = -2; i <= 2; i++) {
    const r = sz * (0.17 - Math.abs(i) * 0.025);
    g.beginPath();
    g.arc(cx + px * i * sz * 0.1, cy + py * i * sz * 0.1, r, 0, Math.PI * 2);
    g.fill();
  }
  g.restore();
}

// ---------- sound ----------
const sfx = (() => {
  let ac = null, noise = null, gawkSrc = null;
  const clips = {};

  function init() {
    try {
      ac = new (window.AudioContext || window.webkitAudioContext)();
      noise = ac.createBuffer(1, ac.sampleRate, ac.sampleRate);
      const d = noise.getChannelData(0);
      for (let i = 0; i < d.length; i++) d[i] = Math.random() * 2 - 1;
    } catch { ac = null; return; }
    // Sound clips next to index.html; built-in synth sounds are used if one is missing.
    for (const name of ["fahhh", "gawk"]) {
      fetch(`${name}.mp3`)
        .then((r) => (r.ok ? r.arrayBuffer() : null))
        .then((buf) => buf && ac.decodeAudioData(buf))
        .then((buf) => { if (buf) clips[name] = { buf, gain: peakGain(buf) }; })
        .catch(() => {});
    }
  }

  // The clips are recorded at very different volumes (fahhh.mp3 peaks at about a third of
  // gawk.mp3 and got lost under the crunch), so scale each one to the same peak level.
  function peakGain(buf) {
    let peak = 0;
    for (let c = 0; c < buf.numberOfChannels; c++) {
      for (const v of buf.getChannelData(c)) if (Math.abs(v) > peak) peak = Math.abs(v);
    }
    return peak > 0 ? 0.95 / peak : 1;
  }

  // Play a clip, optionally only its first `maxSec` seconds with a short fade-out.
  function playClip(clip, t, maxSec = 0) {
    const src = ac.createBufferSource();
    src.buffer = clip.buf;
    const g = ac.createGain();
    g.gain.value = clip.gain;
    src.connect(g).connect(ac.destination);
    if (maxSec && clip.buf.duration > maxSec) {
      g.gain.setValueAtTime(clip.gain, t + maxSec - 0.4);
      g.gain.linearRampToValueAtTime(0.0001, t + maxSec);
      src.start(t, 0, maxSec);
    } else {
      src.start(t);
    }
    return src;
  }

  // Browsers keep audio muted until the first tap/click/keypress on the page.
  const unlocked = () => ac?.state === "running";
  const unlock = () => { if (ac && ac.state !== "running") ac.resume(); };

  // A synthesized "FAHHH": breathy F, then a buzzy "ahh" voice whose pitch sags.
  function fahVoice(t) {
    burst(t, 0.16, 3800, 0.9, 0.4, 2800);
    const v = t + 0.1, dur = 1.15;
    const o = ac.createOscillator();
    o.type = "sawtooth";
    o.frequency.setValueAtTime(240, v);
    o.frequency.linearRampToValueAtTime(270, v + 0.15);
    o.frequency.exponentialRampToValueAtTime(130, v + dur);
    const vib = ac.createOscillator();
    vib.frequency.value = 5.5;
    const vibGain = ac.createGain();
    vibGain.gain.value = 7;
    vib.connect(vibGain).connect(o.frequency);
    const out = ac.createGain();
    out.gain.setValueAtTime(0.0001, v);
    out.gain.exponentialRampToValueAtTime(0.9, v + 0.05);
    out.gain.setValueAtTime(0.9, v + dur * 0.55);
    out.gain.exponentialRampToValueAtTime(0.0001, v + dur);
    // Vowel formants for "ah".
    for (const [f, q, g] of [[750, 6, 3], [1150, 8, 2], [2500, 10, 0.8]]) {
      const bp = ac.createBiquadFilter();
      bp.type = "bandpass";
      bp.frequency.value = f;
      bp.Q.value = q;
      const gg = ac.createGain();
      gg.gain.value = g;
      o.connect(bp).connect(gg).connect(out);
    }
    out.connect(ac.destination);
    o.start(v);
    vib.start(v);
    o.stop(v + dur + 0.05);
    vib.stop(v + dur + 0.05);
  }

  function burst(t, dur, freq, q, gain, freqEnd = freq) {
    const src = ac.createBufferSource();
    src.buffer = noise;
    const f = ac.createBiquadFilter();
    f.type = "bandpass";
    f.Q.value = q;
    f.frequency.setValueAtTime(freq, t);
    f.frequency.exponentialRampToValueAtTime(freqEnd, t + dur);
    const g = ac.createGain();
    g.gain.setValueAtTime(0.0001, t);
    g.gain.exponentialRampToValueAtTime(gain, t + 0.01);
    g.gain.exponentialRampToValueAtTime(0.0001, t + dur);
    src.connect(f).connect(g).connect(ac.destination);
    src.start(t, Math.random() * 0.5);
    src.stop(t + dur + 0.05);
  }

  function tone(t, dur, f0, f1, gain, type = "sine") {
    const o = ac.createOscillator();
    o.type = type;
    o.frequency.setValueAtTime(f0, t);
    o.frequency.exponentialRampToValueAtTime(f1, t + dur);
    const g = ac.createGain();
    g.gain.setValueAtTime(0.0001, t);
    g.gain.exponentialRampToValueAtTime(gain, t + 0.01);
    g.gain.exponentialRampToValueAtTime(0.0001, t + dur);
    o.connect(g).connect(ac.destination);
    o.start(t);
    o.stop(t + dur + 0.05);
  }

  // Skip sounds while muted, otherwise they'd all fire at once when audio unlocks.
  const play = (fn) => () => { if (unlocked()) fn(ac.currentTime); };

  return {
    init,
    unlock,
    unlocked,
    available: () => !!ac,
    onStateChange: (fn) => ac?.addEventListener("statechange", fn),
    crunch: play((t) => {
      for (let i = 0; i < 4; i++) burst(t + i * 0.045 + Math.random() * 0.02, 0.07, 1500 + Math.random() * 2000, 1.2, 0.5);
    }),
    slurp: play((t) => {
      burst(t, 0.35, 350, 4, 0.6, 1400);
      burst(t + 0.12, 0.25, 500, 6, 0.4, 1800);
    }),
    // Two quick "sip sip" pulls through a straw.
    sip: play((t) => {
      for (const d of [0, 0.17]) {
        burst(t + d, 0.11, 2600, 3, 0.55, 1300);
        tone(t + d, 0.09, 900, 1500, 0.06, "triangle");
      }
    }),
    pick: play((t) => tone(t, 0.09, 500, 900, 0.15)),
    // Finishing food.
    fahhh: play((t) => (clips.fahhh ? playClip(clips.fahhh, t) : fahVoice(t))),
    // Finishing a drink: first 3 seconds of the gawk clip (it's long), never two at once.
    gawk: play((t) => {
      if (!clips.gawk) return fahVoice(t);
      try { gawkSrc?.stop(); } catch {}
      gawkSrc = playClip(clips.gawk, t, 3);
    }),
  };
})();

// ---------- tracking ----------
let nextHandId = 1;

// Pair this frame's detections ({at, radius}) with tracks from earlier frames, closest pairs first.
// Matching one detection at a time lets a newcomer's hand steal the identity (and the snack) of
// another player's hand nearby.
function matchTracks(dets, tracks, trackPos) {
  const pairs = [];
  dets.forEach((d, i) => {
    for (const t of tracks) {
      const dd = dist(trackPos(t), d.at);
      if (dd < d.radius) pairs.push({ i, t, dd });
    }
  });
  pairs.sort((a, b) => a.dd - b.dd);
  const out = dets.map(() => null), taken = new Set();
  for (const { i, t } of pairs) {
    if (out[i] || taken.has(t)) continue;
    out[i] = t;
    taken.add(t);
  }
  return out;
}

function processHands(res, now) {
  const dets = res.landmarks.map((lms) => {
    const p = lms.map(toScreen);
    const palm = dist(p[0], p[9]) || 1;
    return {
      p,
      pinch: dist(p[4], p[8]) / palm,
      fist: [8, 12, 16, 20].reduce((s, t) => s + dist(p[t], p[0]), 0) / 4 / palm,
      at: avg([p[0], p[5], p[9], p[13], p[17]]),
      radius: Math.max(palm * 4, 220),
    };
  });
  // Match to the closest hand from previous frames (handedness labels flip too often to use as IDs).
  const matched = matchTracks(dets, [...state.hands.values()], (h) => h.center);
  dets.forEach(({ p, pinch, fist, at: center }, i) => {
    let h = matched[i];
    const fresh = !h;
    if (!h) {
      h = { id: nextHandId++, closed: false, holding: null, x: 0, y: 0, hoverItem: null, hoverTime: 0, cooldown: 0 };
      state.hands.set(h.id, h);
    }

    // Hysteresis so the grab doesn't flicker at the threshold.
    const isPinch = pinch < (h.closed ? 0.45 : 0.3);
    const isFist = fist < (h.closed ? 1.5 : 1.3);
    h.closed = isPinch || isFist;
    const target = isPinch ? mid(p[4], p[8]) : center;
    h.tx = target.x;
    h.ty = target.y;
    if (fresh) { h.x = h.tx; h.y = h.ty; }
    h.center = center;
    // Any of these touching a snack counts as "on" it.
    h.probes = [center, mid(p[4], p[8]), p[4], p[8], p[12], p[16], p[20]];
    h.pts = p;
    h.pinch = pinch;
    h.fist = fist;
    h.seen = now;
  });
  if (res.landmarks.length) state.lastHandSeen = now;
}

function processFace(res, now) {
  const dets = (res.faceLandmarks || []).map((lms, i) => {
    const up = toScreen(lms[13]), lo = toScreen(lms[14]);
    const l = toScreen(lms[61]), r = toScreen(lms[291]);
    const mouthW = dist(l, r) || 1;
    return {
      up, lo, l, r, mouthW,
      ratio: dist(up, lo) / mouthW,
      jaw: res.faceBlendshapes?.[i]?.categories?.find((c) => c.categoryName === "jawOpen")?.score ?? 0,
      at: mid(up, lo),
      radius: Math.max(mouthW * 3, 150),
    };
  });
  // Match to the closest face from previous frames so each person keeps their own state.
  const matched = matchTracks(dets, state.faces, (f) => f);
  const used = new Set();
  dets.forEach(({ up, lo, l, r, mouthW, ratio, jaw, at: c }, i) => {
    let f = matched[i];
    if (!f) {
      f = { x: c.x, y: c.y, open: false };
      state.faces.push(f);
    }
    used.add(f);
    f.open = f.open ? jaw > 0.12 || ratio > 0.18 : jaw > 0.25 || ratio > 0.3;
    f.tx = c.x;
    f.ty = c.y;
    f.mouthW = mouthW;
    f.ratio = ratio;
    f.jaw = jaw;
    f.seen = now;
    f.pts = state.debug ? [up, lo, l, r] : null;
  });
  // A hand often covers the mouth while eating, so remember unseen faces for a bit.
  state.faces = state.faces.filter((f) => used.has(f) || now - f.seen < 1500);
}

function nearestFace(p) {
  let best = null, bd = Infinity;
  for (const f of state.faces) {
    const d = dist(f, p);
    if (d < bd) { bd = d; best = f; }
  }
  return best;
}

// ---------- game logic ----------
function grab(h, it) {
  it.state = "held";
  it.holder = h;
  it.leftHome = false;
  h.holding = it;
  h.hoverItem = null;
  h.hoverTime = 0;
  sfx.pick();
}

function release(h) {
  const it = h.holding;
  if (it && it.state === "held") {
    it.state = "returning";
    it.holder = null;
  }
  h.holding = null;
}

function mouthAnchor(it, f) {
  // Hold the snack just below the mouth so you can still see your mouth chomp.
  return { x: f.x, y: f.y + it.size * 0.32 };
}

function bite(it, now, f) {
  it.taken++;
  it.lastBite = now;
  it.nearTime = 0;
  it.primed = false;
  it.squash = 1;
  state.shake = 7;

  const mouth = f ? { x: f.x, y: f.y } : { x: it.x, y: it.y - it.size * 0.4 };
  if (!it.def.drink) {
    if (it.biteAngle === null) it.biteAngle = Math.atan2(mouth.y - it.y, mouth.x - it.x) - it.rot;
    it.bites.push({ angle: it.biteAngle, k: it.taken - 1 });
    rebuildSprite(it);
  }
  const last = it.taken >= it.def.bites;
  // The last bite plays only the finishing clip (fahhh/gawk) so the crunch doesn't drown it out.
  if (!last) {
    if (it.def.drink) sfx.sip();
    else if (it.def.soft) sfx.slurp();
    else sfx.crunch();
  }
  burst(mouth.x, mouth.y, it.def, 14, it.def.drink);
  popText(mouth.x, mouth.y - it.size * 0.5, pick(it.def.words || (it.def.drink ? SIP_WORDS : BITE_WORDS)), "#fff");

  if (last) finish(it, now, mouth);
}

function finish(it, now, mouth) {
  if (it.holder) it.holder.holding = null;
  it.holder = null;
  it.state = "gone";
  it.respawnAt = now + 1300;
  state.score[it.def.id]++;
  bumpScore(it.def.id);
  burst(mouth.x, mouth.y, it.def, 30, it.def.drink);
  popText(mouth.x, mouth.y - it.size * 0.9, `+1 ${it.def.emoji}`, "#ffd84d", 1.4);
  if (it.def.drink) sfx.gawk();
  else sfx.fahhh();
}

function update(dt, now) {
  // Hands: smooth cursors, drop stale hands, grab.
  // Grabbing is sticky: once a snack is in your hand it stays there until you eat it,
  // put it back on the menu, or your hand leaves the camera for a while.
  for (const [id, h] of state.hands) {
    if (now - h.seen > (h.holding ? 1500 : 700)) {
      release(h);
      state.hands.delete(id);
      continue;
    }
    const k = ease(dt, 25);
    h.x = lerp(h.x, h.tx, k);
    h.y = lerp(h.y, h.ty, k);

    // Which menu strip (if any) the hand is reaching into.
    let inMenu = [], menuSide = null;
    for (const side of SIDES) {
      const m = state.menus[side];
      const pts = h.probes.filter((p) => p.x > m.x0 && p.x < m.x1 && p.y > m.y0 && p.y < m.y1);
      if (pts.length > inMenu.length) { inMenu = pts; menuSide = side; }
    }
    // After putting a snack back, the hand has to leave the menu before it grabs again.
    if (!inMenu.length) h.needLeave = false;
    if (h.holding || now < h.cooldown || h.needLeave) {
      h.hoverItem = null;
      h.hoverTime = 0;
      continue;
    }
    const free = state.items.filter((it) => it.state === "shelf" || it.state === "returning");
    let best = null, bd = Infinity;
    for (const it of free) {
      const d = Math.min(...h.probes.map((p) => dist(it, p)));
      if (d < it.size * (h.closed ? 1.1 : 0.9) && d < bd) { bd = d; best = it; }
    }
    // Not touching one exactly but some of the hand is in the menu strip: take the snack at that height.
    if (!best && inMenu.length) {
      const y = inMenu.reduce((sum, p) => sum + p.y, 0) / inMenu.length;
      for (const it of free) {
        if (it.side !== menuSide) continue;
        const d = Math.abs(it.home.y - y);
        if (d < bd) { bd = d; best = it; }
      }
    }
    if (!best) {
      h.hoverItem = null;
      h.hoverTime = 0;
      continue;
    }
    if (h.hoverItem !== best) { h.hoverItem = best; h.hoverTime = 0; }
    h.hoverTime += dt;
    // Pinch/fist grabs instantly; just holding your hand over a snack grabs it after a moment.
    if (h.closed || h.hoverTime >= HOVER_GRAB) grab(h, best);
  }

  for (const f of state.faces) {
    const k = ease(dt, 30);
    f.x = lerp(f.x, f.tx, k);
    f.y = lerp(f.y, f.ty, k);
  }

  const t = now / 1000;
  for (const it of state.items) {
    it.squash = Math.max(0, it.squash - dt * 4);
    it.near = false;
    switch (it.state) {
      case "shelf":
        it.x = it.home.x;
        it.y = it.home.y + Math.sin(t * 2 + it.i) * 4;
        it.rot = Math.sin(t * 1.5 + it.i) * 0.06;
        it.scale = lerp(it.scale, 1, ease(dt, 10));
        break;

      case "returning": {
        const k = ease(dt, 8);
        it.x = lerp(it.x, it.home.x, k);
        it.y = lerp(it.y, it.home.y, k);
        it.rot = lerp(it.rot, 0, k);
        it.scale = lerp(it.scale, 1, k);
        if (dist(it, it.home) < 3) it.state = "shelf";
        break;
      }

      case "held": {
        const h = it.holder;
        const f = (it.face = nearestFace(h));
        let tx = h.x, ty = h.y;
        let nearR = Infinity;
        if (f) {
          const a = mouthAnchor(it, f);
          nearR = Math.max(f.mouthW * 1.4, it.size * 0.75);
          // Magnet toward the mouth: pulls harder the closer your hand gets, then snaps on.
          const zone = Math.max(it.size * 2.4, f.mouthW * 4);
          const dm = Math.hypot(tx - a.x, ty - a.y);
          if (dm < zone) {
            const m = clamp((1 - dm / zone) * 1.7, 0, 1);
            tx = lerp(tx, a.x, m);
            ty = lerp(ty, a.y, m);
          }
        }
        // Bringing it back to its spot on the menu puts it down.
        if (dist(it, it.home) > it.size * 2) it.leftHome = true;
        if (it.leftHome && dist({ x: h.x, y: h.y }, it.home) < it.size * 0.5) {
          h.cooldown = now + 600;
          h.needLeave = true;
          release(h);
          break;
        }
        const k = ease(dt, 22);
        const px = it.x;
        it.x = lerp(it.x, tx, k);
        it.y = lerp(it.y, ty, k);
        it.vx = lerp(it.vx, (it.x - px) / Math.max(dt, 0.001), 0.2);
        it.rot = lerp(it.rot, clamp(it.vx * 0.0006, -0.4, 0.4), ease(dt, 10));
        it.scale = lerp(it.scale, HELD_SCALE, ease(dt, 10));

        if (f && dist(it, mouthAnchor(it, f)) < nearR) {
          it.near = true;
          it.nearTime += dt;
          if (f.open) it.primed = true;
          const ready = now - it.lastBite > 300;
          // Bite on a chomp (open -> close), or automatically if the mouth isn't being read.
          if (ready && ((it.primed && !f.open) || it.nearTime > 1.5)) bite(it, now, f);
        } else {
          it.nearTime = 0;
          it.primed = false;
        }
        break;
      }

      case "gone":
        if (now > it.respawnAt) {
          it.state = "spawning";
          it.taken = 0;
          it.bites = [];
          it.biteAngle = null;
          it.scale = 0;
          it.x = it.home.x;
          it.y = it.home.y;
          rebuildSprite(it);
        }
        break;

      case "spawning":
        it.scale = lerp(it.scale, 1, ease(dt, 7));
        it.x = it.home.x;
        it.y = it.home.y;
        if (it.scale > 0.97) it.state = "shelf";
        break;
    }
  }

  for (const p of state.particles) {
    p.life -= dt;
    p.vy += p.g * dt;
    p.x += p.vx * dt;
    p.y += p.vy * dt;
  }
  state.particles = state.particles.filter((p) => p.life > 0);

  for (const tx of state.texts) {
    tx.life -= dt;
    tx.y -= 60 * dt;
  }
  state.texts = state.texts.filter((tx) => tx.life > 0);

  state.shake = Math.max(0, state.shake - dt * 40);
}

function burst(x, y, def, n, bubbles) {
  for (let i = 0; i < n; i++) {
    const a = Math.random() * Math.PI * 2;
    const sp = 80 + Math.random() * 260;
    state.particles.push({
      x, y,
      vx: Math.cos(a) * sp,
      vy: Math.sin(a) * sp - (bubbles ? 120 : 160),
      g: bubbles ? -200 : 900,
      life: 0.6 + Math.random() * 0.5,
      max: 1.1,
      r: bubbles ? 3 + Math.random() * 6 : 2 + Math.random() * 5,
      color: pick(def.colors),
      bubble: bubbles,
    });
  }
}

function popText(x, y, text, color, scale = 1) {
  state.texts.push({ x, y, text, color, scale, life: 0.9, max: 0.9 });
}

// ---------- drawing ----------
function draw(now) {
  ctx.save();
  if (state.shake > 0) ctx.translate((Math.random() - 0.5) * state.shake, (Math.random() - 0.5) * state.shake);

  // The camera is the <video> element underneath; the canvas only holds the game layer.
  ctx.clearRect(-20, -20, W + 40, H + 40);

  drawShelf();
  for (const it of state.items) if (it.state !== "held") drawItem(it);
  drawMouthGuide(now);
  for (const it of state.items) if (it.state === "held") drawItem(it);
  drawParticles();
  drawHands();
  drawTexts();
  if (state.debug) drawDebug();

  ctx.restore();
}

function roundRect(x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

function drawShelf() {
  const s = state.size;
  for (const side of SIDES) {
    const items = state.items.filter((it) => it.side === side);
    const xs = items.map((it) => it.home.x), ys = items.map((it) => it.home.y);
    const x = Math.min(...xs) - s * 0.62, y = Math.min(...ys) - s * 0.66;
    const w = Math.max(...xs) - Math.min(...xs) + s * 1.24, h = Math.max(...ys) - Math.min(...ys) + s * 1.32;
    ctx.fillStyle = "rgba(0,0,0,0.3)";
    roundRect(x, y, w, h, 18);
    ctx.fill();
  }
  for (const it of state.items) {
    ctx.fillStyle = "rgba(255,255,255,0.22)";
    ctx.beginPath();
    ctx.ellipse(it.home.x, it.home.y + s * 0.45, s * 0.45, s * 0.1, 0, 0, Math.PI * 2);
    ctx.fill();
  }
}

function drawItem(it) {
  if (it.state === "gone") return;
  const S = it.size * 1.25 * it.scale;
  const sq = it.squash * 0.18;
  ctx.save();
  ctx.translate(it.x, it.y);
  ctx.rotate(it.rot);
  ctx.scale(1 + sq, 1 - sq);
  ctx.drawImage(it.sprite, -S / 2, -S / 2, S, S);
  ctx.restore();

  // Drinks show how much is left instead of bite marks.
  if (it.def.drink && it.taken > 0) {
    const left = 1 - it.taken / it.def.bites;
    const bw = 8, bh = it.size * 0.7;
    const bx = it.x + S * 0.42, by = it.y - bh / 2;
    ctx.fillStyle = "rgba(0,0,0,0.5)";
    roundRect(bx, by, bw, bh, 4);
    ctx.fill();
    ctx.fillStyle = "#7a3b1d";
    roundRect(bx + 1, by + 1 + (bh - 2) * (1 - left), bw - 2, (bh - 2) * left, 3);
    ctx.fill();
  }
}

function drawMouthGuide(now) {
  for (const held of state.items) {
    if (held.state === "held" && held.face && state.faces.includes(held.face)) drawGuide(held, held.face, now);
  }
}

function drawGuide(held, f, now) {
  const pulse = 1 + Math.sin(now / 150) * 0.08;
  const r = Math.max(f.mouthW * 0.75, 26) * pulse;
  ctx.save();
  ctx.lineWidth = 4;
  ctx.setLineDash(held.near ? [] : [8, 8]);
  ctx.strokeStyle = held.near ? "rgba(90,255,120,0.9)" : "rgba(255,255,255,0.7)";
  ctx.beginPath();
  ctx.arc(f.x, f.y, r, 0, Math.PI * 2);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.font = "bold 16px system-ui, sans-serif";
  ctx.textAlign = "center";
  const label = held.near ? (f.open ? "now close! 😬" : "open wide! 😮") : "bring it here 👄";
  ctx.lineWidth = 4;
  ctx.strokeStyle = "rgba(0,0,0,0.7)";
  ctx.strokeText(label, f.x, f.y - r - 10);
  ctx.fillStyle = "#fff";
  ctx.fillText(label, f.x, f.y - r - 10);
  ctx.restore();
}

function drawParticles() {
  for (const p of state.particles) {
    ctx.globalAlpha = clamp(p.life / 0.4, 0, 1);
    ctx.beginPath();
    ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
    if (p.bubble) {
      ctx.strokeStyle = p.color;
      ctx.lineWidth = 2;
      ctx.stroke();
    } else {
      ctx.fillStyle = p.color;
      ctx.fill();
    }
  }
  ctx.globalAlpha = 1;
}

function drawHands() {
  const r = state.size * 0.28;
  for (const h of state.hands.values()) {
    // Fill-up ring while your hand rests on a snack.
    if (h.hoverItem && !h.holding) {
      const it = h.hoverItem;
      const p = clamp(h.hoverTime / HOVER_GRAB, 0, 1);
      ctx.save();
      ctx.lineWidth = 6;
      ctx.lineCap = "round";
      ctx.strokeStyle = "rgba(0,0,0,0.35)";
      ctx.beginPath();
      ctx.arc(it.x, it.y, it.size * 0.62, 0, Math.PI * 2);
      ctx.stroke();
      ctx.strokeStyle = "#ffd84d";
      ctx.beginPath();
      ctx.arc(it.x, it.y, it.size * 0.62, -Math.PI / 2, -Math.PI / 2 + p * Math.PI * 2);
      ctx.stroke();
      ctx.restore();
    }
    ctx.save();
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.arc(h.x, h.y, h.holding ? r * 0.6 : r, 0, Math.PI * 2);
    if (h.closed) {
      ctx.fillStyle = "rgba(255,216,77,0.35)";
      ctx.fill();
      ctx.strokeStyle = "#ffd84d";
    } else {
      ctx.setLineDash([6, 6]);
      ctx.strokeStyle = "rgba(255,255,255,0.85)";
    }
    ctx.stroke();
    ctx.restore();
  }
}

function drawTexts() {
  ctx.save();
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  for (const tx of state.texts) {
    const p = 1 - tx.life / tx.max;
    const sc = tx.scale * (p < 0.15 ? p / 0.15 : 1);
    ctx.globalAlpha = clamp(tx.life / 0.3, 0, 1);
    ctx.font = `900 ${Math.round(34 * sc)}px system-ui, sans-serif`;
    ctx.lineWidth = 6;
    ctx.strokeStyle = "rgba(0,0,0,0.75)";
    ctx.strokeText(tx.text, tx.x, tx.y);
    ctx.fillStyle = tx.color;
    ctx.fillText(tx.text, tx.x, tx.y);
  }
  ctx.restore();
}

function drawDebug() {
  ctx.save();
  ctx.font = "12px monospace";
  for (const h of state.hands.values()) {
    ctx.fillStyle = "#0ff";
    for (const p of h.pts || []) ctx.fillRect(p.x - 2, p.y - 2, 4, 4);
    ctx.fillStyle = "#fff";
    ctx.fillText(`pinch ${h.pinch?.toFixed(2)} fist ${h.fist?.toFixed(2)}`, h.x + 20, h.y);
  }
  for (const f of state.faces) {
    ctx.fillStyle = "#f0f";
    for (const p of f.pts || []) ctx.fillRect(p.x - 3, p.y - 3, 6, 6);
    ctx.fillStyle = "#fff";
    ctx.fillText(`jaw ${f.jaw.toFixed(2)} ratio ${f.ratio.toFixed(2)} ${f.open ? "OPEN" : "closed"}`, f.x + 30, f.y);
  }
  ctx.restore();
}

// ---------- HUD ----------
const scoreEl = document.getElementById("score");
function renderScore(bumpId) {
  const total = Object.values(state.score).reduce((a, b) => a + b, 0);
  scoreEl.innerHTML =
    `<span id="sc-total">😋 <b>${total}</b></span>` +
    FOODS.filter((f) => state.score[f.id] > 0)
      .map((f) => `<span id="sc-${f.id}">${f.emoji} <b>${state.score[f.id]}</b></span>`)
      .join("");
  if (bumpId) for (const id of ["sc-total", `sc-${bumpId}`]) document.getElementById(id)?.classList.add("bump");
}

function bumpScore(id) {
  renderScore(id);
}

let lastStatus = 0;
function updateStatus(now) {
  if (now - lastStatus < 200) return;
  lastStatus = now;
  let text = "";
  if (!state.faces.length) text = "🙂 Look at the camera so I can find your mouth";
  else if (now - state.lastHandSeen > 1000) text = "✋ Show your hand to the camera";
  if (statusEl.textContent !== text) statusEl.textContent = text;
}

// ---------- main loop ----------
const params = new URLSearchParams(location.search);
// How many people can play at once (faces/mouths tracked, and both hands each). ?players=6 for more.
const PLAYERS = clamp(parseInt(params.get("players"), 10) || 4, 1, 8);
try { localStorage.removeItem("snack-attack-cpu"); } catch {}
let delegate = params.get("cpu") ? "CPU" : "GPU";
let fileset = null;
let swapping = false;
const diag = { gpuSamples: [], handMs: 0, faceMs: 0, frames: 0, handFrames: 0, faceFrames: 0, fps: 0, fpsCount: 0, fpsAt: 0, startedAt: 0, error: "" };

function loop() {
  const now = performance.now();
  const dt = Math.min(0.05, (now - state.lastFrame) / 1000);
  state.lastFrame = now;

  if (handLm && !swapping && video.readyState >= 2 && video.currentTime !== state.lastVideoTime) {
    state.lastVideoTime = video.currentTime;
    updateVideoRect();
    // MediaPipe needs strictly increasing timestamps.
    const ts = Math.max(now, state.lastDetect + 1);
    state.lastDetect = ts;
    try {
      // Hands and faces take turns so each camera frame only pays for one model (the screen
      // stays smooth because hands, faces and snacks are interpolated between updates).
      //   holding a snack: hands and faces alternate (mouth chomps matter now)
      //   otherwise:       hands 2 of every 3 frames, faces 1 of 3 (just keeping track of mouths)
      const holding = state.items.some((it) => it.state === "held");
      const doFace = holding ? diag.frames % 2 === 1 : diag.frames % 3 === 2;
      let t0 = performance.now();
      if (!doFace) {
        const hr = handLm.detectForVideo(video, ts);
        const handMs = performance.now() - t0;
        diag.handMs = lerp(diag.handMs, handMs, 0.1);
        if (now - diag.startedAt > 2000 && diag.gpuSamples.length < 15) diag.gpuSamples.push(handMs);
        processHands(hr, now);
        if (hr.landmarks.length) diag.handFrames++;
      } else {
        t0 = performance.now();
        const fr = faceLm.detectForVideo(video, ts);
        diag.faceMs = lerp(diag.faceMs, performance.now() - t0, 0.1);
        processFace(fr, now);
        if (fr.faceLandmarks?.length) diag.faceFrames++;
      }
      diag.frames++;
      diag.fpsCount++;
    } catch (e) {
      console.error(e);
      diag.error = e.message || String(e);
      if (delegate === "GPU") switchToCPU("GPU tracking crashed");
    }
    // Some browsers/drivers run "GPU" tracking in software (very slow) or silently find nothing.
    // Fall back to CPU only when it's clearly broken, judged after warm-up.
    if (delegate === "GPU" && !swapping) {
      const s = diag.gpuSamples;
      const median = s.length === 15 ? [...s].sort((a, b) => a - b)[7] : 0;
      if (median > 150) switchToCPU(`GPU tracking too slow (${median.toFixed(0)} ms/frame)`);
      else if (diag.frames >= 90 && now - diag.startedAt > 6000 && diag.handFrames + diag.faceFrames === 0) {
        switchToCPU("GPU tracking found nothing");
      }
    }
  }
  if (now - diag.fpsAt > 1000) {
    diag.fps = diag.fpsCount;
    diag.fpsCount = 0;
    diag.fpsAt = now;
  }

  update(dt, now);
  draw(now);
  updateStatus(now);
  updateDiag();
  requestAnimationFrame(loop);
}

async function loadModels() {
  const opts = (modelAssetPath, extra) => ({
    baseOptions: { modelAssetPath, delegate },
    runningMode: "VIDEO",
    ...extra,
  });
  // Created one after another: building two GPU graphs at once can fail on some browsers.
  // The caps are for everyone in view combined, and MediaPipe stops looking for new hands/faces
  // once that many are tracked, so leave room for every player (and both hands each).
  const hand = await HandLandmarker.createFromOptions(fileset, opts(HAND_MODEL, {
    numHands: PLAYERS * 2,
    minHandDetectionConfidence: 0.3,
    minHandPresenceConfidence: 0.3,
    minTrackingConfidence: 0.3,
  }));
  const face = await FaceLandmarker.createFromOptions(fileset, opts(FACE_MODEL, {
    numFaces: PLAYERS,
    outputFaceBlendshapes: true,
    minFaceDetectionConfidence: 0.3,
    minFacePresenceConfidence: 0.3,
    minTrackingConfidence: 0.3,
  }));
  handLm?.close?.();
  faceLm?.close?.();
  handLm = hand;
  faceLm = face;
  diag.frames = diag.handFrames = diag.faceFrames = 0;
  diag.gpuSamples = [];
  diag.startedAt = performance.now();
}

async function switchToCPU(reason) {
  if (swapping) return;
  swapping = true;
  console.warn(`${reason}, switching to CPU`);
  toast("Switching tracking to CPU mode…", 0);
  delegate = "CPU";
  try {
    await loadModels();
    toast("CPU mode on ✔");
  } catch (e) {
    diag.error = e.message || String(e);
    toast("Tracking failed to load: " + diag.error);
  }
  swapping = false;
}

const toastEl = document.getElementById("toast");
let toastTimer = 0;
function toast(text, ms = 3000) {
  toastEl.textContent = text;
  toastEl.hidden = false;
  clearTimeout(toastTimer);
  if (ms) toastTimer = setTimeout(() => (toastEl.hidden = true), ms);
}

const diagEl = document.getElementById("diag");
let lastDiag = 0;
function updateDiag() {
  const now = performance.now();
  if (!state.debug) {
    if (!diagEl.hidden) diagEl.hidden = true;
    return;
  }
  if (now - lastDiag < 250) return;
  lastDiag = now;
  diagEl.hidden = false;
  diagEl.textContent =
    `mode ${delegate} · tracking ${diag.fps} fps · cam ${video.videoWidth}×${video.videoHeight}\n` +
    `hand ${diag.handMs.toFixed(1)} ms · face ${diag.faceMs.toFixed(1)} ms per frame\n` +
    `hands now ${state.hands.size}/${PLAYERS * 2} · faces ${state.faces.length}/${PLAYERS}` +
    (state.faces.length ? ` (mouths ${state.faces.map((f) => (f.open ? "OPEN" : "closed")).join(", ")})` : "") +
    `\nframes ${diag.frames} · with hands ${diag.handFrames} · with face ${diag.faceFrames}` +
    (diag.error ? `\nerror: ${diag.error}` : "");
}

async function start() {
  startBtn.hidden = true;
  overlay.hidden = false;

  if (!navigator.mediaDevices?.getUserMedia) {
    return fail("Camera needs HTTPS (or localhost). Open this page over https.");
  }

  sfx.unlock(); // allowed without a tap on sites you've used before; otherwise the first tap turns sound on
  msgEl.textContent = "Starting camera… (allow camera access if asked)";
  try {
    video.srcObject = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30 } },
      audio: false,
    });
    await video.play();
    sfx.unlock(); // some browsers allow sound once the camera is live
  } catch (e) {
    return fail(`Camera blocked: ${e.message}`);
  }

  // Show the game right away; tracking finishes loading behind it.
  layout();
  overlay.hidden = true;
  hud.hidden = false;
  toast(delegate === "CPU" ? "Loading hand & face tracking (CPU mode)…" : "Loading hand & face tracking…", 0);
  state.lastFrame = performance.now();
  if (!looping) { looping = true; requestAnimationFrame(loop); }

  try {
    fileset ||= await FilesetResolver.forVisionTasks(WASM_URL);
    try {
      await loadModels();
    } catch (e) {
      if (delegate !== "GPU") throw e;
      console.warn("GPU delegate failed, falling back to CPU", e);
      delegate = "CPU";
      await loadModels();
    }
  } catch (e) {
    console.error(e);
    return fail(`Couldn't load tracking models: ${e.message}`);
  }
  toast("Ready! Grab a snack ✋", 2000);
}

let looping = false;

function fail(text) {
  overlay.hidden = false;
  msgEl.textContent = text;
  startBtn.hidden = false;
}

startBtn.addEventListener("click", start);
for (const ev of ["pointerdown", "keydown", "touchstart", "click"]) window.addEventListener(ev, sfx.unlock, { passive: true });
sfx.init();
// The game is played hands-free, so nobody taps by default and audio stays locked.
// Show a button saying so until the browser lets sound play.
const soundBtn = document.getElementById("btn-sound");
const syncSoundBtn = () => { soundBtn.hidden = !sfx.available() || sfx.unlocked(); };
soundBtn.addEventListener("click", sfx.unlock);
sfx.onStateChange(syncSoundBtn);
syncSoundBtn();
window.addEventListener("resize", layout);
function resetScore() {
  for (const id in state.score) state.score[id] = 0;
  renderScore();
}
renderScore();
window.addEventListener("keydown", (e) => {
  if (e.key === "d" || e.key === "D") state.debug = !state.debug;
  if (e.key === "r" || e.key === "R") resetScore();
});
document.getElementById("btn-reset").addEventListener("click", resetScore);

layout();
start();
