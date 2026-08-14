/* ===========================================================================
 * Eco-Ear front end
 *
 * Captures raw microphone PCM at 16 kHz, keeps a rolling 2-second window, and
 * posts a WAV of that window to /predict twice a second. The capture core
 * (ring buffer + WAV encoder) is deliberately unchanged from the original
 * implementation -- it matches what the server's preprocess_file expects.
 * =========================================================================== */

(() => {
  'use strict';

  // --- audio contract, must stay in step with app.py (SR / CLIP_SECS) -------
  const TARGET_SAMPLE_RATE = 16000;
  const CLIP_SECS = 2.0;
  const TARGET_SAMPLES = TARGET_SAMPLE_RATE * CLIP_SECS; // 32,000 samples
  const POST_INTERVAL_MS = 500;
  const MIN_SAMPLES_BEFORE_POST = 4000; // ~0.25 s of audio
  const NEAR_SILENT_PEAK = 0.02;
  const MAX_LOG_ENTRIES = 60;

  const FALLBACK_CLASSES = ['background', 'gunshot', 'chainsaw', 'firework', 'vehicle'];

  // --------------------------------------------------------------- elements --
  const el = {};
  [
    'statusPill', 'statusText', 'banner', 'bannerText', 'liveStatus', 'liveAlert',
    'clipHint', 'scope', 'scopeCanvas', 'levelMeter', 'levelFill', 'levelValue',
    'levelWarn', 'startBtn', 'stopBtn', 'latencyHint', 'verdict', 'verdictArc',
    'verdictPct', 'verdictLabel', 'verdictSub', 'verdictTags', 'meters',
    'thresholdText', 'statClips', 'statThreats', 'statUptime', 'clearLogBtn',
    'logEmpty', 'log', 'drop', 'fileInput', 'fileStatus',
  ].forEach((id) => { el[id] = document.getElementById(id); });

  // ------------------------------------------------------------------ state --
  let classes = FALLBACK_CLASSES.slice();
  let threatClasses = classes.filter((c) => c !== 'background');
  let threshold = 0.10;

  let audioStream = null;
  let audioContext = null;
  let scriptProcessor = null;
  let mediaStreamSource = null;
  let analyser = null;
  let mute = null; // zero-gain sink, see startRecording()
  let streamInterval = null;
  let rafId = null;
  let uptimeInterval = null;

  let isRecording = false;
  let isUploading = false;
  let pendingBlob = null; // only ever the freshest clip

  let clipCount = 0;
  let threatCount = 0;
  let sessionStartedAt = 0;
  let lastAnnouncedThreat = '';

  // Rolling PCM window
  const pcmRingBuffer = new Float32Array(TARGET_SAMPLES);
  let bufferWriteIndex = 0;
  let totalSamplesRecorded = 0;

  // Canvas
  const canvasCtx = el.scopeCanvas.getContext('2d');
  let waveformData = null;

  const meterRefs = new Map(); // class name -> { row, fill, pct }

  // ================================================================ helpers ==

  const clamp01 = (n) => Math.min(1, Math.max(0, n));
  const pct = (n) => `${Math.round(clamp01(n) * 100)}%`;

  function colorFor(name) {
    const styles = getComputedStyle(document.documentElement);
    const value = styles.getPropertyValue(`--class-${name}`).trim();
    return value || styles.getPropertyValue('--class-background').trim() || '#64748b';
  }

  function setStatus(text, state = 'idle') {
    el.statusText.textContent = text;
    el.statusPill.dataset.state = state;
    // Only narrate meaningful transitions; the 500 ms upload churn would
    // otherwise flood a screen reader with "Analysing..." forever.
    if (state !== 'busy') el.liveStatus.textContent = text;
  }

  function showBanner(message) {
    el.bannerText.textContent = message;
    el.banner.dataset.visible = 'true';
  }

  function hideBanner() {
    el.banner.dataset.visible = 'false';
    el.bannerText.textContent = '';
  }

  function formatClock(ms) {
    const total = Math.floor(ms / 1000);
    const mins = Math.floor(total / 60);
    const secs = total % 60;
    return `${mins}:${String(secs).padStart(2, '0')}`;
  }

  // ============================================================ model config ==

  async function loadMeta() {
    try {
      const res = await fetch('/meta', { headers: { Accept: 'application/json' } });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const meta = await res.json();

      if (Array.isArray(meta.classes) && meta.classes.length) classes = meta.classes;
      threatClasses = Array.isArray(meta.threat_classes) && meta.threat_classes.length
        ? meta.threat_classes
        : classes.filter((c) => c !== 'background');
      if (typeof meta.threshold === 'number') threshold = meta.threshold;

      if (meta.clip_secs && meta.sample_rate) {
        el.clipHint.textContent =
          `${Number(meta.clip_secs).toFixed(1)} s window \u00b7 ${meta.sample_rate / 1000} kHz \u00b7 ${meta.model || 'model'}`;
      }
    } catch (error) {
      // Not fatal: the fallback class list is the same one the server ships.
      console.warn('Could not load /meta, using built-in defaults:', error);
    }

    el.thresholdText.textContent = threshold.toFixed(2);
    buildMeters();
  }

  function buildMeters() {
    el.meters.innerHTML = '';
    meterRefs.clear();

    classes.forEach((name) => {
      const row = document.createElement('li');
      row.className = 'meter';
      row.dataset.winner = 'false';
      row.style.setProperty('--meter-color', colorFor(name));

      const head = document.createElement('div');
      head.className = 'meter__row';

      const label = document.createElement('span');
      label.className = 'meter__name';
      label.textContent = name;

      const spacer = document.createElement('span');
      spacer.className = 'meter__spacer';

      const value = document.createElement('span');
      value.className = 'meter__pct';
      value.textContent = '--';

      head.append(label, spacer, value);

      const track = document.createElement('div');
      track.className = 'meter__track';
      track.setAttribute('role', 'meter');
      track.setAttribute('aria-label', `${name} confidence`);
      track.setAttribute('aria-valuemin', '0');
      track.setAttribute('aria-valuemax', '100');
      track.setAttribute('aria-valuenow', '0');

      const fill = document.createElement('div');
      fill.className = 'meter__fill';
      track.append(fill);

      row.append(head, track);
      el.meters.append(row);

      meterRefs.set(name, { row, fill, pctEl: value, track });
    });
  }

  // ================================================================ verdict ==

  const ARC_LENGTH = 314.16; // 2 * pi * r, r = 50 in the SVG

  function renderPrediction(data, source) {
    const label = String(data.class || 'unknown');
    const confidence = clamp01(Number(data.confidence) || 0);
    const isThreat = threatClasses.includes(label);
    const color = colorFor(label);

    el.verdict.dataset.threat = String(isThreat);
    el.verdict.style.borderLeftColor = color;
    el.verdictLabel.textContent = label;
    el.verdictLabel.style.color = color;
    el.verdictPct.textContent = pct(confidence);
    el.verdictArc.style.stroke = color;
    el.verdictArc.style.strokeDashoffset = String(ARC_LENGTH * (1 - confidence));

    el.verdictSub.textContent = isThreat
      ? `Threat class cleared the ${threshold.toFixed(2)} threshold.`
      : 'No threat class cleared the threshold.';

    renderTags(data.top_labels);
    renderScores(data.scores, label);

    clipCount += 1;
    el.statClips.textContent = String(clipCount);

    if (isThreat) {
      threatCount += 1;
      el.statThreats.textContent = String(threatCount);
      addLogEntry(label, confidence, source, color);

      // Repeated identical detections at 2 Hz would make the assertive live
      // region unusable, so only announce a change of class.
      if (label !== lastAnnouncedThreat) {
        el.liveAlert.textContent = `${label} detected, ${pct(confidence)} confidence.`;
        lastAnnouncedThreat = label;
      }
    } else {
      lastAnnouncedThreat = '';
    }
  }

  function renderTags(topLabels) {
    el.verdictTags.innerHTML = '';
    if (!Array.isArray(topLabels)) return;

    topLabels.slice(0, 4).forEach((entry) => {
      if (!entry || !entry.label) return;
      const li = document.createElement('li');
      li.textContent = `${entry.label} ${pct(entry.score)}`;
      el.verdictTags.append(li);
    });
  }

  function renderScores(scores, winner) {
    if (!scores || typeof scores !== 'object') return;

    meterRefs.forEach((ref, name) => {
      const raw = Number(scores[name]);
      const value = Number.isFinite(raw) ? clamp01(raw) : 0;
      ref.fill.style.width = pct(value);
      ref.pctEl.textContent = pct(value);
      ref.track.setAttribute('aria-valuenow', String(Math.round(value * 100)));
      ref.row.dataset.winner = String(name === winner);
    });
  }

  function resetVerdict() {
    el.verdict.dataset.threat = 'false';
    el.verdict.style.borderLeftColor = colorFor('background');
    el.verdictLabel.textContent = 'Awaiting audio';
    el.verdictLabel.style.color = '';
    el.verdictPct.textContent = '--';
    el.verdictArc.style.strokeDashoffset = String(ARC_LENGTH);
    el.verdictSub.textContent = 'Start the monitor or drop an audio file to classify.';
    el.verdictTags.innerHTML = '';
    lastAnnouncedThreat = '';
  }

  // ============================================================== threat log ==

  function addLogEntry(label, confidence, source, color) {
    const li = document.createElement('li');
    li.className = 'log__item log__item--new';
    li.style.setProperty('--event-color', color);

    const time = document.createElement('span');
    time.className = 'log__time';
    time.textContent = new Date().toLocaleTimeString([], { hour12: false });

    const name = document.createElement('span');
    name.className = 'log__label';
    name.textContent = label;

    const src = document.createElement('span');
    src.className = 'log__src';
    src.textContent = source;

    const conf = document.createElement('span');
    conf.className = 'log__conf';
    conf.textContent = pct(confidence);

    li.append(time, name, src, conf);
    el.log.prepend(li);
    el.logEmpty.style.display = 'none';

    while (el.log.children.length > MAX_LOG_ENTRIES) {
      el.log.lastElementChild.remove();
    }
  }

  el.clearLogBtn.addEventListener('click', () => {
    el.log.innerHTML = '';
    el.logEmpty.style.display = '';
    threatCount = 0;
    el.statThreats.textContent = '0';
  });

  // ============================================================= visualizer ==

  function resizeCanvas() {
    const ratio = window.devicePixelRatio || 1;
    const rect = el.scopeCanvas.getBoundingClientRect();
    if (!rect.width) return;
    el.scopeCanvas.width = Math.floor(rect.width * ratio);
    el.scopeCanvas.height = Math.floor(rect.height * ratio);
    canvasCtx.setTransform(ratio, 0, 0, ratio, 0, 0);
  }

  window.addEventListener('resize', resizeCanvas);

  function drawFrame() {
    if (!isRecording || !analyser || !waveformData) return;

    analyser.getFloatTimeDomainData(waveformData);

    const width = el.scopeCanvas.clientWidth;
    const height = el.scopeCanvas.clientHeight;

    canvasCtx.clearRect(0, 0, width, height);

    // midline
    canvasCtx.strokeStyle = 'rgba(148, 163, 184, 0.22)';
    canvasCtx.lineWidth = 1;
    canvasCtx.beginPath();
    canvasCtx.moveTo(0, height / 2);
    canvasCtx.lineTo(width, height / 2);
    canvasCtx.stroke();

    // waveform
    canvasCtx.strokeStyle = getComputedStyle(document.documentElement)
      .getPropertyValue('--accent').trim() || '#34d399';
    canvasCtx.lineWidth = 1.6;
    canvasCtx.lineJoin = 'round';
    canvasCtx.beginPath();

    const step = width / waveformData.length;
    let peak = 0;
    for (let i = 0; i < waveformData.length; i++) {
      const sample = waveformData[i];
      const abs = Math.abs(sample);
      if (abs > peak) peak = abs;
      const y = (height / 2) - sample * (height / 2) * 0.92;
      if (i === 0) canvasCtx.moveTo(0, y);
      else canvasCtx.lineTo(i * step, y);
    }
    canvasCtx.stroke();

    updateLevel(peak);
    rafId = requestAnimationFrame(drawFrame);
  }

  function updateLevel(peak) {
    const shown = Math.round(clamp01(peak) * 100);
    el.levelFill.style.width = `${shown}%`;
    el.levelValue.textContent = `${shown}%`;
    el.levelMeter.setAttribute('aria-valuenow', String(shown));
  }

  // ============================================================== recording ==

  async function startRecording() {
    if (isRecording) return;

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      showBanner('This browser does not expose getUserMedia. Microphone capture is unavailable.');
      setStatus('Unsupported', 'error');
      return;
    }

    hideBanner();
    setStatus('Requesting microphone...', 'busy');

    try {
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      audioContext = new AudioCtx({ sampleRate: TARGET_SAMPLE_RATE });
      if (audioContext.state === 'suspended') await audioContext.resume();

      // Echo cancellation / noise suppression / AGC are tuned for voice calls:
      // they actively suppress the transient, non-speech sounds this app needs
      // to detect (gunshots, chainsaws, fireworks), and echoCancellation in
      // particular will null out audio played through the same machine's
      // speakers, mistaking it for acoustic echo. Turn them all off.
      audioStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          sampleRate: TARGET_SAMPLE_RATE,
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false,
        },
      });

      mediaStreamSource = audioContext.createMediaStreamSource(audioStream);

      analyser = audioContext.createAnalyser();
      analyser.fftSize = 2048;
      analyser.smoothingTimeConstant = 0.6;
      waveformData = new Float32Array(analyser.fftSize);
      mediaStreamSource.connect(analyser);

      scriptProcessor = audioContext.createScriptProcessor(2048, 1, 1);

      pcmRingBuffer.fill(0);
      bufferWriteIndex = 0;
      totalSamplesRecorded = 0;

      scriptProcessor.onaudioprocess = (event) => {
        if (!isRecording) return;
        const input = event.inputBuffer.getChannelData(0);
        for (let i = 0; i < input.length; i++) {
          pcmRingBuffer[bufferWriteIndex] = input[i];
          bufferWriteIndex = (bufferWriteIndex + 1) % TARGET_SAMPLES;
          totalSamplesRecorded++;
        }
      };

      // A ScriptProcessorNode only runs while it's connected to the graph, but
      // wiring it straight to destination pipes the microphone back out of the
      // speakers -- audible echo, and a feedback loop on an open mic. Route it
      // through a silent gain node instead: the node still gets pulled, no
      // audio reaches the output.
      mute = audioContext.createGain();
      mute.gain.value = 0;
      mediaStreamSource.connect(scriptProcessor);
      scriptProcessor.connect(mute);
      mute.connect(audioContext.destination);

      isRecording = true;
      sessionStartedAt = Date.now();

      streamInterval = setInterval(postWindow, POST_INTERVAL_MS);
      uptimeInterval = setInterval(() => {
        el.statUptime.textContent = formatClock(Date.now() - sessionStartedAt);
      }, 1000);

      el.startBtn.disabled = true;
      el.stopBtn.disabled = false;
      el.scope.dataset.active = 'true';
      resizeCanvas();
      rafId = requestAnimationFrame(drawFrame);

      setStatus('Listening', 'listening');
    } catch (error) {
      console.error('startRecording failed:', error);
      const denied = error && (error.name === 'NotAllowedError' || error.name === 'SecurityError');
      showBanner(denied
        ? 'Microphone permission was denied. Allow access in your browser, then start again.'
        : `Could not open the microphone: ${error.message || error}`);
      setStatus('Microphone error', 'error');
      cleanupAudio();
    }
  }

  function stopRecording() {
    if (!isRecording) return;
    isRecording = false;

    clearInterval(streamInterval);
    streamInterval = null;
    clearInterval(uptimeInterval);
    uptimeInterval = null;
    if (rafId) cancelAnimationFrame(rafId);
    rafId = null;

    cleanupAudio();

    el.startBtn.disabled = false;
    el.stopBtn.disabled = true;
    el.scope.dataset.active = 'false';
    el.levelWarn.textContent = '';
    updateLevel(0);
    canvasCtx.clearRect(0, 0, el.scopeCanvas.clientWidth, el.scopeCanvas.clientHeight);
    setStatus('Stopped', 'idle');
  }

  function cleanupAudio() {
    if (scriptProcessor) {
      scriptProcessor.onaudioprocess = null;
      try { scriptProcessor.disconnect(); } catch { /* already detached */ }
      scriptProcessor = null;
    }
    if (mute) {
      try { mute.disconnect(); } catch { /* already detached */ }
      mute = null;
    }
    if (analyser) {
      try { analyser.disconnect(); } catch { /* already detached */ }
      analyser = null;
    }
    if (mediaStreamSource) {
      try { mediaStreamSource.disconnect(); } catch { /* already detached */ }
      mediaStreamSource = null;
    }
    if (audioStream) {
      audioStream.getTracks().forEach((track) => track.stop());
      audioStream = null;
    }
    if (audioContext) {
      audioContext.close().catch(() => { /* already closing */ });
      audioContext = null;
    }
    waveformData = null;
  }

  /* Unrolls the circular PCM buffer into a sequential 2-second Float32Array. */
  function getOrderedBufferSnapshot() {
    const snapshot = new Float32Array(TARGET_SAMPLES);
    if (totalSamplesRecorded < TARGET_SAMPLES) {
      snapshot.set(pcmRingBuffer.subarray(0, bufferWriteIndex), 0);
    } else {
      const tailLength = TARGET_SAMPLES - bufferWriteIndex;
      snapshot.set(pcmRingBuffer.subarray(bufferWriteIndex), 0);
      snapshot.set(pcmRingBuffer.subarray(0, bufferWriteIndex), tailLength);
    }
    return snapshot;
  }

  function postWindow() {
    if (!isRecording || totalSamplesRecorded < MIN_SAMPLES_BEFORE_POST) return;

    const snapshot = getOrderedBufferSnapshot();

    let peak = 0;
    for (let i = 0; i < snapshot.length; i++) {
      const abs = Math.abs(snapshot[i]);
      if (abs > peak) peak = abs;
    }

    // A near-silent clip can only ever come back as background. Say so rather
    // than letting the user wonder why nothing is ever detected.
    el.levelWarn.textContent = peak < NEAR_SILENT_PEAK
      ? 'Input is near-silent. Check the selected microphone or input gain.'
      : '';

    enqueueUpload(encodeWAV(snapshot, TARGET_SAMPLE_RATE));
  }

  // ================================================================= upload ==

  function enqueueUpload(blob) {
    // Keep only the newest clip: a stale 2-second window is worthless for a
    // live monitor, so drop anything that queued up behind a slow request.
    pendingBlob = blob;
    drainQueue();
  }

  async function drainQueue() {
    if (isUploading || !pendingBlob) return;

    const blob = pendingBlob;
    pendingBlob = null;
    isUploading = true;

    try {
      await predict(blob, 'clip.wav', 'mic');
    } catch (error) {
      console.error('Live upload failed:', error);
    } finally {
      isUploading = false;
      if (pendingBlob) drainQueue();
    }
  }

  async function predict(blob, filename, source) {
    const startedAt = performance.now();
    const form = new FormData();
    form.append('file', blob, filename);

    const response = await fetch('/predict', { method: 'POST', body: form });
    const text = await response.text();

    let data = null;
    try { data = JSON.parse(text); } catch { /* handled below */ }

    if (!response.ok) {
      const message = (data && data.error) || text || response.statusText;
      throw new Error(message);
    }
    if (!data) throw new Error('Server returned a non-JSON response.');
    if (data.error) throw new Error(data.error);

    el.latencyHint.textContent = `${Math.round(performance.now() - startedAt)} ms \u00b7 ${source}`;
    hideBanner();
    renderPrediction(data, source);

    if (source === 'mic' && isRecording) setStatus('Listening', 'listening');
    return data;
  }

  // ============================================================ file upload ==

  async function classifyFile(file) {
    if (!file) return;

    el.fileStatus.dataset.kind = 'info';
    el.fileStatus.textContent = `Analysing ${file.name}...`;
    setStatus('Analysing file', 'busy');

    try {
      await predict(file, file.name, 'file');
      el.fileStatus.textContent = `Done: ${file.name}`;
    } catch (error) {
      console.error('File classification failed:', error);
      el.fileStatus.dataset.kind = 'error';
      el.fileStatus.textContent = `Failed: ${error.message || error}`;
      showBanner(`Could not classify ${file.name}: ${error.message || error}`);
    } finally {
      setStatus(isRecording ? 'Listening' : 'Idle', isRecording ? 'listening' : 'idle');
    }
  }

  el.fileInput.addEventListener('change', () => {
    const file = el.fileInput.files && el.fileInput.files[0];
    if (file) classifyFile(file);
    el.fileInput.value = ''; // allow re-picking the same file
  });

  ['dragenter', 'dragover'].forEach((type) => {
    el.drop.addEventListener(type, (event) => {
      event.preventDefault();
      el.drop.dataset.drag = 'true';
    });
  });

  ['dragleave', 'dragend'].forEach((type) => {
    el.drop.addEventListener(type, () => { el.drop.dataset.drag = 'false'; });
  });

  el.drop.addEventListener('drop', (event) => {
    event.preventDefault();
    el.drop.dataset.drag = 'false';
    const file = event.dataTransfer && event.dataTransfer.files && event.dataTransfer.files[0];
    if (file) classifyFile(file);
  });

  // The dropzone is a <label for="fileInput">, so a click already opens the
  // picker. Without this, a keyboard Enter on the label would do nothing.
  el.drop.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      el.fileInput.click();
    }
  });
  el.drop.tabIndex = 0;

  // =================================================== 16-bit PCM WAV writer ==

  function encodeWAV(samples, sampleRate) {
    const buffer = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(buffer);

    writeString(view, 0, 'RIFF');
    view.setUint32(4, 36 + samples.length * 2, true);
    writeString(view, 8, 'WAVE');
    writeString(view, 12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);              // PCM
    view.setUint16(22, 1, true);              // mono
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true); // byte rate
    view.setUint16(32, 2, true);              // block align
    view.setUint16(34, 16, true);             // bits per sample
    writeString(view, 36, 'data');
    view.setUint32(40, samples.length * 2, true);

    let offset = 44;
    for (let i = 0; i < samples.length; i++, offset += 2) {
      const s = Math.max(-1, Math.min(1, samples[i]));
      view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    }

    return new Blob([view], { type: 'audio/wav' });
  }

  function writeString(view, offset, string) {
    for (let i = 0; i < string.length; i++) {
      view.setUint8(offset + i, string.charCodeAt(i));
    }
  }

  // =================================================================== init ==

  el.startBtn.addEventListener('click', startRecording);
  el.stopBtn.addEventListener('click', stopRecording);

  window.addEventListener('beforeunload', () => {
    if (isRecording) stopRecording();
  });

  resizeCanvas();
  resetVerdict();
  updateLevel(0);
  setStatus('Idle', 'idle');
  loadMeta();
})();
