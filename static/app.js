let startBtn = document.getElementById('startBtn');
let stopBtn = document.getElementById('stopBtn');
let statusDiv = document.getElementById('status');
let predDiv = document.getElementById('prediction');

let audioStream = null;
let mediaRecorder = null;
let isRecording = false;
let isUploading = false;
let chunks = [];
let uploadQueue = [];
const AudioCtx = window.AudioContext || window.webkitAudioContext;
const audioContext = AudioCtx ? new AudioCtx() : null;

/*
 * Find a MIME type supported by the current browser.
 *
 * Chrome/Edge normally support audio/webm;codecs=opus.
 * Firefox may also support audio/ogg;codecs=opus.
 */
function getSupportedMimeType() {
  const types = [
    'audio/webm;codecs=opus',
    'audio/webm',
    'audio/ogg;codecs=opus',
    'audio/ogg'
  ];

  for (const type of types) {
    if (MediaRecorder.isTypeSupported(type)) {
      return type;
    }
  }

  return '';
}

/*
 * Start recording from the microphone.
 *
 * The recorder stays running continuously and emits a data chunk
 * every 2 seconds instead of repeatedly stopping and restarting.
 */
async function startRecording() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    statusDiv.textContent =
      'getUserMedia is not supported in this browser.';
    return;
  }

  if (typeof MediaRecorder === 'undefined') {
    statusDiv.textContent =
      'MediaRecorder is not available in this browser.';
    return;
  }

  if (isRecording) {
    return;
  }

  try {
    // Request microphone access once.
    if (!audioStream) {
      audioStream = await navigator.mediaDevices.getUserMedia({
        audio: true
      });
    }

    const mimeType = getSupportedMimeType();

    if (!mimeType) {
      statusDiv.textContent =
        'This browser does not support a compatible audio recording format.';
      return;
    }

    console.log('Using MediaRecorder MIME type:', mimeType);

    mediaRecorder = new MediaRecorder(audioStream, {
      mimeType: mimeType
    });

    /*
     * Each time a 2-second chunk becomes available,
     * upload it to the Flask server.
     */
    mediaRecorder.ondataavailable = (event) => {
      if (!event.data || event.data.size === 0) return;

      console.log('Received audio chunk:', event.data.size, 'bytes', event.data.type);

      // Keep adding to the cumulative buffer
      chunks.push(event.data);

      // Create a complete, playable WebM blob containing all chunks recorded so far
      const currentMimeType = getSupportedMimeType() || event.data.type;
      const cumulativeBlob = new Blob(chunks, { type: currentMimeType });

      if (audioContext) {
        convertChunkToWav(cumulativeBlob)
          .then(wavBlob => {
            if (wavBlob) {
              enqueueUpload(wavBlob);
            } else {
              enqueueUpload(cumulativeBlob);
            }
          })
          .catch(err => {
            console.error('Chunk conversion failed:', err);
            enqueueUpload(cumulativeBlob);
          });
      } else {
        enqueueUpload(cumulativeBlob);
      }
    };

    mediaRecorder.onerror = (event) => {
      console.error('MediaRecorder error:', event.error);

      statusDiv.textContent =
        'Recording error: ' +
        (event.error?.message || 'Unknown recording error');
    };

    mediaRecorder.onstart = () => {
      console.log('MediaRecorder started');
      statusDiv.textContent = 'Recording...';
    };

    mediaRecorder.onstop = async () => {
      console.log('MediaRecorder stopped');

      if (chunks.length === 0) return;

      try {
        const mimeType = getSupportedMimeType() || (chunks[0] && chunks[0].type) || '';
        const finalBlob = new Blob(chunks, { type: mimeType });
        chunks = [];

        // Try to convert the final assembled blob and enqueue it.
        if (audioContext) {
          try {
            const wav = await convertChunkToWav(finalBlob);
            if (wav) enqueueUpload(wav);
          } catch (err) {
            console.error('Final conversion failed, uploading raw blob instead:', err);
            enqueueUpload(finalBlob);
          }
        } else {
          enqueueUpload(finalBlob);
        }
      } catch (err) {
        console.error('Failed to upload final blob:', err);
      }
    };

    /*
     * Start recording continuously.
     *
     * The 2000 argument tells MediaRecorder to emit
     * a dataavailable event approximately every 2 seconds.
     */
    mediaRecorder.start(2000);

    isRecording = true;

    startBtn.disabled = true;
    stopBtn.disabled = false;

    statusDiv.textContent = 'Recording...';

  } catch (error) {
    console.error('startRecording error:', error);

    statusDiv.textContent =
      'Microphone access denied or error: ' +
      (error.message || error);

    // Clean up if microphone initialization failed.
    if (audioStream) {
      audioStream.getTracks().forEach(track => track.stop());
      audioStream = null;
    }

    mediaRecorder = null;
    isRecording = false;

    startBtn.disabled = false;
    stopBtn.disabled = true;
  }
}

/*
 * Stop recording.
 */
function stopRecording() {
  if (!isRecording) {
    return;
  }

  console.log('Stopping recording...');

  isRecording = false;

  if (mediaRecorder && mediaRecorder.state !== 'inactive') {
    /*
     * Calling stop() causes one final dataavailable event,
     * so the last partial chunk will also be uploaded.
     */
    mediaRecorder.stop();
  }

  startBtn.disabled = false;
  stopBtn.disabled = true;

  statusDiv.textContent = 'Stopped';
}

/*
 * Upload one recorded audio chunk to Flask.
 */
async function uploadAudioChunk(blob) {
  if (!blob || blob.size === 0) {
    return;
  }

  /*
   * Don't start another upload if the previous request is still
   * being processed.
   */
  if (isUploading) {
    console.warn('Previous upload still running; skipping this chunk.');
    return;
  }

  isUploading = true;

  try {
    statusDiv.textContent = 'Uploading clip...';

    let filename = 'clip.bin';
    if (blob.type.includes('ogg')) {
      filename = 'clip.ogg';
    } else if (blob.type.includes('webm')) {
      filename = 'clip.webm';
    } else if (blob.type.includes('wav') || blob.type.includes('wave')) {
      filename = 'clip.wav';
    }

    const form = new FormData();
    form.append('file', blob, filename);

    console.log(
      'Uploading:',
      filename,
      blob.type,
      blob.size,
      'bytes'
    );

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
        console.error(
          'Could not parse JSON response:',
          text
        );
      }
    }

    /*
     * Handle HTTP errors.
     */
    if (!response.ok) {
      const message =
        data?.error ||
        text ||
        response.statusText;

      console.error(
        'Prediction request failed:',
        response.status,
        message
      );

      statusDiv.textContent =
        'Server error: ' + message;

      return;
    }

    /*
     * Handle a successful response.
     */
    if (!data) {
      try {
        data = JSON.parse(text);
      } catch (error) {
        console.error(
          'Server returned invalid JSON:',
          text
        );

        statusDiv.textContent =
          'Server returned invalid JSON.';

        return;
      }
    }

    /*
     * Display the prediction.
     */
    if (data.class !== undefined && data.confidence !== undefined) {
      const confidence = Math.round(data.confidence * 100);

      predDiv.textContent =
        `Prediction: ${data.class} ` +
        `(confidence ${confidence}%)`;

      console.log('Prediction:', data);
    } else if (data.error) {
      console.error(
        'Server returned an error:',
        data.error
      );

      statusDiv.textContent =
        'Prediction error: ' + data.error;
    } else {
      console.warn(
        'Unexpected server response:',
        data
      );

      statusDiv.textContent =
        'Unexpected server response.';
    }

    /*
     * Only show Idle if recording has actually stopped.
     */
    if (isRecording) {
      statusDiv.textContent = 'Recording...';
    } else {
      statusDiv.textContent = 'Idle';
    }

  } catch (error) {
    console.error('Upload failed:', error);

    statusDiv.textContent =
      'Upload failed: ' +
      (error.message || error);

  } finally {
    isUploading = false;
  }
}

/*
 * Upload queue and audio conversion helpers
 */
function enqueueUpload(blob) {
  uploadQueue.push(blob);
  processQueue();
}

async function processQueue() {
  if (isUploading) return;
  if (uploadQueue.length === 0) return;

  const next = uploadQueue.shift();
  try {
    await uploadAudioChunk(next);
  } catch (err) {
    console.error('Queued upload failed:', err);
  } finally {
    if (uploadQueue.length > 0) processQueue();
  }
}

async function convertChunkToWav(blob) {
  if (!audioContext) return null;

  const arrayBuffer = await blob.arrayBuffer();
  let decoded;
  try {
    decoded = await audioContext.decodeAudioData(arrayBuffer);
  } catch (err) {
    console.warn('decodeAudioData failed for chunk:', err);
    return null;
  }

  const SR = 16000;
  const clipSecs = 2.0; // Send the most recent 2 seconds
  const totalSamples = decoded.length;
  const maxSamples = Math.floor(clipSecs * decoded.sampleRate);
  
  // Determine start frame for the last 2 seconds
  const startSample = Math.max(0, totalSamples - maxSamples);
  const frameCount = totalSamples - startSample;

  const offlineCtx = new OfflineAudioContext(1, Math.ceil((frameCount / decoded.sampleRate) * SR), SR);

  // Mix channels into a mono buffer for only the tail portion
  const mono = offlineCtx.createBuffer(1, frameCount, decoded.sampleRate);
  const channelCount = decoded.numberOfChannels;
  for (let i = 0; i < frameCount; i++) {
    let sum = 0;
    for (let ch = 0; ch < channelCount; ch++) {
      sum += decoded.getChannelData(ch)[startSample + i];
    }
    mono.getChannelData(0)[i] = sum / channelCount;
  }

  const src = offlineCtx.createBufferSource();
  src.buffer = mono;
  src.connect(offlineCtx.destination);
  src.start(0);

  const rendered = await offlineCtx.startRendering();
  return audioBufferToWavBlob(rendered);
}

function audioBufferToWavBlob(buffer) {
  const numChannels = buffer.numberOfChannels;
  const sampleRate = buffer.sampleRate;
  const bitDepth = 16;

  const samples = buffer.getChannelData(0);
  const bufferLength = samples.length * (bitDepth / 8);
  const wavBuffer = new ArrayBuffer(44 + bufferLength);
  const view = new DataView(wavBuffer);

  writeString(view, 0, 'RIFF');
  view.setUint32(4, 36 + bufferLength, true);
  writeString(view, 8, 'WAVE');
  writeString(view, 12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * numChannels * (bitDepth / 8), true);
  view.setUint16(32, numChannels * (bitDepth / 8), true);
  view.setUint16(34, bitDepth, true);
  writeString(view, 36, 'data');
  view.setUint32(40, bufferLength, true);

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

// Event Listeners
if (startBtn) {
  startBtn.addEventListener('click', startRecording);
}

if (stopBtn) {
  stopBtn.addEventListener('click', stopRecording);
}