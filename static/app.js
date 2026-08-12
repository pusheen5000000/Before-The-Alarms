let startBtn = document.getElementById('startBtn');
let stopBtn = document.getElementById('stopBtn');
let statusDiv = document.getElementById('status');
let predDiv = document.getElementById('prediction');

let audioStream = null;
let audioContext = null;
let scriptProcessor = null;
let mediaStreamSource = null;
let streamInterval = null;

let isRecording = false;
let isUploading = false;
let uploadQueue = [];

// Audio settings matching PyTorch model expectations
const TARGET_SAMPLE_RATE = 16000;
const CLIP_SECS = 2.0;
const TARGET_SAMPLES = TARGET_SAMPLE_RATE * CLIP_SECS; // 32,000 samples for 2s

// Fixed-size rolling PCM buffer for the last 2 seconds
let pcmRingBuffer = new Float32Array(TARGET_SAMPLES);
let bufferWriteIndex = 0;
let totalSamplesRecorded = 0;

/*
 * Start recording directly from microphone PCM stream.
 */
async function startRecording() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    statusDiv.textContent = 'getUserMedia is not supported in this browser.';
    return;
  }

  if (isRecording) return;

  try {
    // Initialize AudioContext at target sample rate (16kHz)
    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    audioContext = new AudioCtx({ sampleRate: TARGET_SAMPLE_RATE });

    if (audioContext.state === 'suspended') {
      await audioContext.resume();
    }

    // Get microphone stream
    audioStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        sampleRate: TARGET_SAMPLE_RATE,
        echoCancellation: true,
        noiseSuppression: true
      }
    });

    mediaStreamSource = audioContext.createMediaStreamSource(audioStream);

    // Create a processor node to tap raw audio frames (buffer size 2048)
    scriptProcessor = audioContext.createScriptProcessor(2048, 1, 1);

    // Reset buffer tracking
    pcmRingBuffer.fill(0);
    bufferWriteIndex = 0;
    totalSamplesRecorded = 0;

    // Capture incoming raw PCM samples directly from microphone
    scriptProcessor.onaudioprocess = (event) => {
      if (!isRecording) return;

      const inputData = event.inputBuffer.getChannelData(0);
      for (let i = 0; i < inputData.length; i++) {
        pcmRingBuffer[bufferWriteIndex] = inputData[i];
        bufferWriteIndex = (bufferWriteIndex + 1) % TARGET_SAMPLES;
        totalSamplesRecorded++;
      }
    };

    // Connect node graph
    mediaStreamSource.connect(scriptProcessor);
    scriptProcessor.connect(audioContext.destination);

    // Send 2-second tail segment to Flask every 500ms
    streamInterval = setInterval(() => {
      if (!isRecording) return;
      if (totalSamplesRecorded < 4000) return; // Wait until at least ~0.25s recorded

      const snapshot = getOrderedBufferSnapshot();
      const wavBlob = encodeWAV(snapshot, audioContext.sampleRate);
      enqueueUpload(wavBlob);
    }, 500);

    isRecording = true;
    startBtn.disabled = true;
    stopBtn.disabled = false;
    statusDiv.textContent = 'Recording...';
    console.log('PCM Stream Recording started');

  } catch (error) {
    console.error('startRecording error:', error);
    statusDiv.textContent = 'Microphone access denied or error: ' + (error.message || error);
    cleanupAudio();
  }
}

/*
 * Unrolls the circular PCM ring buffer into a sequential 2-second Float32Array.
 */
function getOrderedBufferSnapshot() {
  const snapshot = new Float32Array(TARGET_SAMPLES);
  if (totalSamplesRecorded < TARGET_SAMPLES) {
    // If less than 2 seconds recorded so far, copy from index 0
    snapshot.set(pcmRingBuffer.subarray(0, bufferWriteIndex), 0);
  } else {
    // Copy oldest part from writeIndex to end, then newest part from 0 to writeIndex
    const tailLength = TARGET_SAMPLES - bufferWriteIndex;
    snapshot.set(pcmRingBuffer.subarray(bufferWriteIndex), 0);
    snapshot.set(pcmRingBuffer.subarray(0, bufferWriteIndex), tailLength);
  }
  return snapshot;
}

/*
 * Stop recording and cleanup audio nodes.
 */
function stopRecording() {
  if (!isRecording) return;

  console.log('Stopping recording...');
  isRecording = false;

  if (streamInterval) {
    clearInterval(streamInterval);
    streamInterval = null;
  }

  cleanupAudio();

  startBtn.disabled = false;
  stopBtn.disabled = true;
  statusDiv.textContent = 'Stopped';
}

function cleanupAudio() {
  if (scriptProcessor) {
    scriptProcessor.disconnect();
    scriptProcessor.onaudioprocess = null;
    scriptProcessor = null;
  }

  if (mediaStreamSource) {
    mediaStreamSource.disconnect();
    mediaStreamSource = null;
  }

  if (audioStream) {
    audioStream.getTracks().forEach(track => track.stop());
    audioStream = null;
  }

  if (audioContext) {
    audioContext.close();
    audioContext = null;
  }
}

/*
 * Upload one recorded audio clip to Flask.
 */
async function uploadAudioChunk(blob) {
  if (!blob || blob.size === 0) return;

  if (isUploading) {
    return;
  }

  isUploading = true;

  try {
    statusDiv.textContent = 'Uploading clip...';

    const filename = 'clip.wav';
    const form = new FormData();
    form.append('file', blob, filename);

    const response = await fetch('/predict', {
      method: 'POST',
      body: form
    });

    const text = await response.text();
    const contentType = response.headers.get('content-type') || '';

    let data = null;
    if (contentType.includes('application/json')) {
      try {
        data = JSON.parse(text);
      } catch (error) {
        console.error('Could not parse JSON response:', text);
      }
    }

    if (!response.ok) {
      const message = data?.error || text || response.statusText;
      console.error('Prediction request failed:', response.status, message);
      statusDiv.textContent = 'Server error: ' + message;
      return;
    }

    if (!data) {
      try {
        data = JSON.parse(text);
      } catch (error) {
        console.error('Server returned invalid JSON:', text);
        statusDiv.textContent = 'Server returned invalid JSON.';
        return;
      }
    }

    if (data.class !== undefined && data.confidence !== undefined) {
      const confidence = Math.round(data.confidence * 100);
      predDiv.textContent = `Prediction: ${data.class} (confidence ${confidence}%)`;
      console.log('Prediction:', data);
    } else if (data.error) {
      console.error('Server returned an error:', data.error);
      statusDiv.textContent = 'Prediction error: ' + data.error;
    }

    if (isRecording) {
      statusDiv.textContent = 'Recording...';
    } else {
      statusDiv.textContent = 'Idle';
    }

  } catch (error) {
    console.error('Upload failed:', error);
    statusDiv.textContent = 'Upload failed: ' + (error.message || error);
  } finally {
    isUploading = false;
  }
}

/*
 * Upload Queue Management
 */
function enqueueUpload(blob) {
  // Drop older pending payloads if client gets behind to ensure zero latency
  if (uploadQueue.length > 0) {
    uploadQueue = [];
  }
  uploadQueue.push(blob);
  processQueue();
}

async function processQueue() {
  if (isUploading || uploadQueue.length === 0) return;

  const next = uploadQueue.shift();
  try {
    await uploadAudioChunk(next);
  } catch (err) {
    console.error('Queued upload failed:', err);
  } finally {
    if (uploadQueue.length > 0) processQueue();
  }
}

/*
 * Direct 16-bit PCM WAV Encoder
 */
function encodeWAV(samples, sampleRate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);

  writeString(view, 0, 'RIFF');
  view.setUint32(4, 36 + samples.length * 2, true);
  writeString(view, 8, 'WAVE');
  writeString(view, 12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); // PCM format
  view.setUint16(22, 1, true); // Mono channel
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeString(view, 36, 'data');
  view.setUint32(40, samples.length * 2, true);

  let offset = 44;
  for (let i = 0; i < samples.length; i++, offset += 2) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
  }

  return new Blob([view], { type: 'audio/wav' });
}

function writeString(view, offset, string) {
  for (let i = 0; i < string.length; i++) {
    view.setUint8(offset + i, string.charCodeAt(i));
  }
}

// Event Listeners
if (startBtn) {
  startBtn.addEventListener('click', startRecording);
}

if (stopBtn) {
  stopBtn.addEventListener('click', stopRecording);
}