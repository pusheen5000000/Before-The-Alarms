// ============================================================
// Canopy Guardian — frontend demo logic
//
// This file drives the whole page: decoding an uploaded clip,
// drawing the waveform + detection timeline, and running a
// PLACEHOLDER in-browser detector (see runInference below).
//
// Swap-in point for the real model:
//   Replace runInference(audioBuffer) with a call to your
//   inference endpoint (e.g. fetch('/api/detect', { body: wav }))
//   that returns the same shape: an array of
//   { startTime, endTime, label, confidence } events.
// ============================================================

const dropzone = document.getElementById("dropzone");
const dropzoneLabel = document.getElementById("dropzoneLabel");
const fileInput = document.getElementById("fileInput");
const player = document.getElementById("player");

const waveformBlock = document.getElementById("waveformBlock");
const timelineBlock = document.getElementById("timelineBlock");
const waveformCanvas = document.getElementById("waveformCanvas");
const timelineCanvas = document.getElementById("timelineCanvas");
const timelineAxis = document.getElementById("timelineAxis");
const clipMeta = document.getElementById("clipMeta");

const statusDot = document.getElementById("statusDot");
const statusText = document.getElementById("statusText");

const summaryCard = document.getElementById("summaryCard");
const eventList = document.getElementById("eventList");
const eventCount = document.getElementById("eventCount");

let audioCtx = null;
let currentBuffer = null;
let currentEvents = [];
let currentDuration = 0;
let rafId = null;

// ---------------- file intake ----------------

["dragenter", "dragover"].forEach(evt =>
  dropzone.addEventListener(evt, e => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  })
);
["dragleave", "drop"].forEach(evt =>
  dropzone.addEventListener(evt, e => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
  })
);
dropzone.addEventListener("drop", e => {
  const file = e.dataTransfer.files?.[0];
  if (file) handleFile(file);
});
dropzone.addEventListener("click", () => fileInput.click());
dropzone.addEventListener("keydown", e => {
  if (e.key === "Enter" || e.key === " ") fileInput.click();
});
fileInput.addEventListener("change", () => {
  const file = fileInput.files?.[0];
  if (file) handleFile(file);
});

async function handleFile(file) {
  setStatus("processing");
  dropzoneLabel.textContent = file.name;

  player.src = URL.createObjectURL(file);
  waveformBlock.hidden = false;
  timelineBlock.hidden = false;

  const arrayBuffer = await file.arrayBuffer();
  audioCtx = audioCtx || new (window.AudioContext || window.webkitAudioContext)();
  const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);

  currentBuffer = audioBuffer;
  currentDuration = audioBuffer.duration;

  clipMeta.textContent = `${audioBuffer.sampleRate} Hz · ${formatTime(audioBuffer.duration)}`;

  drawWaveform(audioBuffer);

  const events = runInference(audioBuffer);
  currentEvents = events;

  drawTimeline(audioBuffer, events);
  renderTimelineAxis(audioBuffer.duration);
  renderEventList(events);
  renderSummary(events);
  setStatus(events.length ? "alert" : "listening");
}

// ---------------- detector (placeholder heuristic) ----------------
//
// Windows the clip, runs an FFT per window, and flags windows where
// energy concentrates in the 100-600 Hz band typical of chainsaw
// engine + chain harmonics, above a loudness floor. This is a stand-in
// for a trained backbone (YAMNet/PANNs embeddings -> classifier head)
// — real judgement calls (multi-class, learned thresholds, per-SNR
// calibration) belong in that trained model, not here.

function runInference(audioBuffer) {
  const channel = audioBuffer.getChannelData(0);
  const sampleRate = audioBuffer.sampleRate;

  const windowSize = 2048;
  const hop = 1024; // 50% overlap
  const lowHz = 100;
  const highHz = 600;
  const binWidth = sampleRate / windowSize;
  const lowBin = Math.max(1, Math.floor(lowHz / binWidth));
  const highBin = Math.min(windowSize / 2 - 1, Math.ceil(highHz / binWidth));

  const NOISE_FLOOR = 0.015;   // min RMS to consider a window non-silent
  const BAND_THRESHOLD = 0.42; // fraction of spectral energy in target band

  const candidates = []; // { t, confidence }

  for (let start = 0; start + windowSize <= channel.length; start += hop) {
    const frame = new Float64Array(windowSize);
    let sumSq = 0;
    for (let i = 0; i < windowSize; i++) {
      const hann = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (windowSize - 1));
      const sample = channel[start + i] * hann;
      frame[i] = sample;
      sumSq += sample * sample;
    }
    const rms = Math.sqrt(sumSq / windowSize);
    if (rms < NOISE_FLOOR) continue;

    const magnitudes = fftMagnitude(frame);
    let totalEnergy = 0;
    let bandEnergy = 0;
    for (let b = 1; b < magnitudes.length; b++) {
      const e = magnitudes[b] * magnitudes[b];
      totalEnergy += e;
      if (b >= lowBin && b <= highBin) bandEnergy += e;
    }
    if (totalEnergy === 0) continue;
    const bandRatio = bandEnergy / totalEnergy;

    if (bandRatio > BAND_THRESHOLD) {
      const confidence = Math.min(0.99, bandRatio * (0.6 + Math.min(rms * 4, 0.4)));
      candidates.push({ t: start / sampleRate, confidence });
    }
  }

  // merge adjacent flagged windows into discrete events
  const events = [];
  let open = null;
  const gapTolerance = (hop / sampleRate) * 1.5;

  for (const c of candidates) {
    if (open && c.t - open.endTime <= gapTolerance) {
      open.endTime = c.t + windowSize / sampleRate;
      open.peak = Math.max(open.peak, c.confidence);
    } else {
      if (open) events.push(open);
      open = { startTime: c.t, endTime: c.t + windowSize / sampleRate, peak: c.confidence };
    }
  }
  if (open) events.push(open);

  return events
    .filter(e => e.endTime - e.startTime >= 0.15) // drop single-frame flickers
    .map(e => ({
      startTime: e.startTime,
      endTime: e.endTime,
      confidence: e.peak,
      label: "possible chainsaw"
    }));
}

// Minimal radix-2 Cooley-Tukey FFT, returns magnitude of the first N/2 bins.
function fftMagnitude(real) {
  const n = real.length;
  const re = Float64Array.from(real);
  const im = new Float64Array(n);

  // bit-reversal permutation
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      [re[i], re[j]] = [re[j], re[i]];
      [im[i], im[j]] = [im[j], im[i]];
    }
  }

  for (let len = 2; len <= n; len <<= 1) {
    const ang = (-2 * Math.PI) / len;
    const wRe = Math.cos(ang), wIm = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let curRe = 1, curIm = 0;
      for (let k = 0; k < len / 2; k++) {
        const uRe = re[i + k], uIm = im[i + k];
        const vRe = re[i + k + len / 2] * curRe - im[i + k + len / 2] * curIm;
        const vIm = re[i + k + len / 2] * curIm + im[i + k + len / 2] * curRe;
        re[i + k] = uRe + vRe;
        im[i + k] = uIm + vIm;
        re[i + k + len / 2] = uRe - vRe;
        im[i + k + len / 2] = uIm - vIm;
        const nextRe = curRe * wRe - curIm * wIm;
        const nextIm = curRe * wIm + curIm * wRe;
        curRe = nextRe; curIm = nextIm;
      }
    }
  }

  const half = n / 2;
  const mags = new Float64Array(half);
  for (let i = 0; i < half; i++) mags[i] = Math.sqrt(re[i] * re[i] + im[i] * im[i]);
  return mags;
}

// ---------------- rendering ----------------

function drawWaveform(audioBuffer) {
  const canvas = waveformCanvas;
  const dpr = window.devicePixelRatio || 1;
  const width = canvas.clientWidth;
  const height = canvas.clientHeight || 90;
  canvas.width = width * dpr;
  canvas.height = height * dpr;
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, width, height);

  const data = audioBuffer.getChannelData(0);
  const step = Math.ceil(data.length / width);
  const mid = height / 2;

  ctx.fillStyle = "rgba(116, 176, 131, 0.18)";
  ctx.strokeStyle = "#74b083";
  ctx.lineWidth = 1;
  ctx.beginPath();

  for (let x = 0; x < width; x++) {
    let min = 1, max = -1;
    const start = x * step;
    for (let i = 0; i < step && start + i < data.length; i++) {
      const v = data[start + i];
      if (v < min) min = v;
      if (v > max) max = v;
    }
    ctx.moveTo(x, mid + min * mid * 0.9);
    ctx.lineTo(x, mid + max * mid * 0.9);
  }
  ctx.stroke();

  canvas._playheadHeight = height;
}

function drawTimeline(audioBuffer, events) {
  const canvas = timelineCanvas;
  const dpr = window.devicePixelRatio || 1;
  const width = canvas.clientWidth;
  const height = canvas.clientHeight || 110;
  canvas.width = width * dpr;
  canvas.height = height * dpr;
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, width, height);

  const duration = audioBuffer.duration;
  const data = audioBuffer.getChannelData(0);
  const mid = height * 0.62;
  const step = Math.ceil(data.length / width);

  // amplitude envelope, strip-chart style
  ctx.strokeStyle = "#3c5240";
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let x = 0; x < width; x++) {
    let sum = 0, n = 0;
    const start = x * step;
    for (let i = 0; i < step && start + i < data.length; i++) {
      sum += Math.abs(data[start + i]);
      n++;
    }
    const avg = n ? sum / n : 0;
    const y = mid - avg * mid * 1.6;
    if (x === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();

  // baseline
  ctx.strokeStyle = "#263626";
  ctx.beginPath();
  ctx.moveTo(0, mid);
  ctx.lineTo(width, mid);
  ctx.stroke();

  // flare markers per event
  events.forEach(evt => {
    const cx = (evt.startTime / duration) * width;
    const high = evt.confidence > 0.75;
    ctx.fillStyle = high ? "#d9583f" : "#e8934a";
    ctx.beginPath();
    ctx.moveTo(cx, 6);
    ctx.lineTo(cx - 5, 16);
    ctx.lineTo(cx + 5, 16);
    ctx.closePath();
    ctx.fill();
  });

  canvas._playheadHeight = height;
}

function renderTimelineAxis(duration) {
  timelineAxis.innerHTML = "";
  const steps = 5;
  for (let i = 0; i <= steps; i++) {
    const span = document.createElement("span");
    span.textContent = formatTime((duration / steps) * i);
    timelineAxis.appendChild(span);
  }
}

function renderEventList(events) {
  eventList.innerHTML = "";
  eventCount.textContent = `${events.length} event${events.length === 1 ? "" : "s"}`;

  if (events.length === 0) {
    const li = document.createElement("li");
    li.className = "empty-state";
    li.textContent = "No threat-band energy detected in this clip.";
    eventList.appendChild(li);
    return;
  }

  events.forEach(evt => {
    const li = document.createElement("li");
    li.className = "event-row";

    const flare = document.createElement("span");
    flare.className = `flare-icon ${evt.confidence > 0.75 ? "high" : "mid"}`;

    const time = document.createElement("span");
    time.className = "event-time";
    time.textContent = formatTime(evt.startTime);

    const label = document.createElement("span");
    label.className = "event-label";
    label.textContent = evt.label;

    const conf = document.createElement("span");
    conf.className = "event-confidence";
    conf.textContent = evt.confidence.toFixed(2);

    li.append(flare, time, label, conf);
    eventList.appendChild(li);
  });
}

function renderSummary(events) {
  if (events.length === 0) {
    summaryCard.hidden = true;
    return;
  }
  const peak = events.reduce((m, e) => Math.max(m, e.confidence), 0);
  summaryCard.hidden = false;
  summaryCard.innerHTML = `
    <p class="summary-title">Chainsaw detected</p>
    <p class="summary-body">${events.length} event${events.length === 1 ? "" : "s"} flagged · peak confidence ${peak.toFixed(2)}</p>
  `;
}

function setStatus(state) {
  statusDot.classList.remove("listening", "alert");
  if (state === "processing") {
    statusText.textContent = "NODE 04 · ANALYZING";
  } else if (state === "alert") {
    statusDot.classList.add("alert");
    statusText.textContent = "NODE 04 · THREAT FLAGGED";
  } else {
    statusDot.classList.add("listening");
    statusText.textContent = "NODE 04 · LISTENING";
  }
}

// ---------------- playhead sync ----------------

player.addEventListener("play", () => {
  cancelAnimationFrame(rafId);
  const tick = () => {
    drawPlayheads();
    if (!player.paused && !player.ended) rafId = requestAnimationFrame(tick);
  };
  tick();
});
player.addEventListener("pause", () => cancelAnimationFrame(rafId));
player.addEventListener("seeked", drawPlayheads);

function drawPlayheads() {
  if (!currentBuffer || !currentDuration) return;
  const fraction = player.currentTime / currentDuration;

  [waveformCanvas, timelineCanvas].forEach(canvas => {
    const dpr = window.devicePixelRatio || 1;
    const ctx = canvas.getContext("2d");
    const width = canvas.width / dpr;
    const height = canvas.height / dpr;
    // redraw base layer then overlay playhead
    if (canvas === waveformCanvas) drawWaveform(currentBuffer);
    else drawTimeline(currentBuffer, currentEvents);

    const x = fraction * width;
    ctx.strokeStyle = "#e9e4d6";
    ctx.globalAlpha = 0.55;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, height);
    ctx.stroke();
    ctx.globalAlpha = 1;
  });
}

function formatTime(sec) {
  if (!isFinite(sec)) return "0:00";
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}
