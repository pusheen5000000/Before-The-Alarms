from flask import Flask, request, jsonify, send_from_directory
from werkzeug.exceptions import HTTPException
import os
import tempfile
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa

# Simple CNN matching the training network
class SimpleCNN(nn.Module):
    def __init__(self, n_mels=64, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        
        # Flatten all dimensions except batch (shape becomes 1x64 instead of 64x1)
        x = torch.flatten(x, 1)
        
        x = self.fc(x)
        return x

MODEL_PATH = "pytorch_threat_model.pt"
SR = 16000
CLIP_SECS = 2.0
N_MELS = 64
BACKGROUND_THRESHOLD = float(os.getenv("BACKGROUND_THRESHOLD", "0.55"))

app = Flask(__name__, static_folder="static")


@app.errorhandler(Exception)
def handle_exception(e):
    import traceback
    # If it's an HTTP exception (404, 405, etc), return its code and message without a stack trace
    if isinstance(e, HTTPException):
        return jsonify({"error": e.description}), e.code

    tb = traceback.format_exc()
    app.logger.exception(e)
    return jsonify({"error": str(e), "trace": tb}), 500


@app.route('/.well-known/<path:subpath>', methods=['GET'])
def well_known(subpath):
    # Respond to browser probes (e.g. Chrome DevTools) with a no-content response
    return ('', 204)


def load_model(path):
    if not os.path.exists(path):
        return None, None
    data = torch.load(path, map_location=torch.device("cpu"))
    classes = data.get("classes")
    num_classes = len(classes)
    model = SimpleCNN(n_mels=N_MELS, num_classes=num_classes)
    model.load_state_dict(data["model_state_dict"])
    model.eval()
    return model, classes


def preprocess_file(path, sr=SR, clip_secs=CLIP_SECS, n_mels=N_MELS):
    y, _ = librosa.load(path, sr=sr, mono=True)
    clip_len = int(sr * clip_secs)
    if len(y) < clip_len:
        y = np.pad(y, (0, clip_len - len(y)), mode="constant")
    elif len(y) > clip_len:
        y = y[:clip_len]
    melspec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=1024, hop_length=512, n_mels=n_mels, power=2.0
    )
    log_mel = librosa.power_to_db(melspec, ref=np.max)
    log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-6)
    # shape (n_mels, time) -> (1, 1, n_mels, time)
    arr = np.expand_dims(np.expand_dims(log_mel.astype(np.float32), axis=0), axis=0)
    return torch.from_numpy(arr)


model, CLASSES = load_model(MODEL_PATH)


@app.route("/", methods=["GET"])
def index():
    return send_from_directory("static", "index.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    global model, CLASSES
    if request.method == "GET":
        return jsonify({"info": "Send a POST multipart/form-data with field 'file' to get a prediction."})
    if model is None:
        return jsonify({"error": "Model file not found. Train the model first (pytorch_threat_model.pt)."}), 400
    if "file" not in request.files:
        return jsonify({"error": "No file field in request"}), 400
    f = request.files["file"]
    # NamedTemporaryFile on Windows cannot be reopened; use mkstemp and close the fd.
    filename = f.filename or ""
    ext = os.path.splitext(filename)[1] or ".webm"
    fd, tmp_path = tempfile.mkstemp(suffix=ext)
    os.close(fd)
    try:
        f.save(tmp_path)
        # log basic info for debugging
        try:
            st = os.stat(tmp_path)
            app.logger.info(f"Saved upload to {tmp_path} ({st.st_size} bytes)")
        except Exception:
            pass
        try:
            tensor = preprocess_file(tmp_path)
        except Exception as e:
            # Try server-side ffmpeg conversion as a fallback (browser may upload webm/ogg)
            import traceback, subprocess
            tb = traceback.format_exc()
            app.logger.warning("preprocess_file failed, attempting ffmpeg conversion:\n" + tb)
            converted = tmp_path + ".converted.wav"
            try:
                # call ffmpeg to convert input to 16kHz mono WAV
                cmd = [
                    "ffmpeg",
                    "-y",
                    "-i",
                    tmp_path,
                    "-ar",
                    str(SR),
                    "-ac",
                    "1",
                    converted,
                ]
                subprocess.check_output(cmd, stderr=subprocess.STDOUT)
                app.logger.info(f"ffmpeg conversion succeeded: {converted}")
                tensor = preprocess_file(converted)
            except FileNotFoundError:
                # ffmpeg not installed
                app.logger.error("ffmpeg not found; install ffmpeg to enable server-side conversion")
                return jsonify({"error": f"Failed to process audio: {e}", "trace": tb, "hint": "Install ffmpeg for server-side conversion"}), 500
            except subprocess.CalledProcessError as cpe:
                out = cpe.output.decode(errors='ignore') if isinstance(cpe.output, bytes) else str(cpe.output)
                app.logger.error(f"ffmpeg conversion failed: {out}")
                return jsonify({"error": f"Failed to process audio: {e}", "trace": tb, "ffmpeg_output": out}), 500
            finally:
                try:
                    if os.path.exists(converted):
                        os.remove(converted)
                except Exception:
                    pass
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
    try:
        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            idx = int(np.argmax(probs))
            class_name = CLASSES[idx]
            confidence = float(probs[idx])
            background_idx = CLASSES.index("background") if "background" in CLASSES else None

            # If the top prediction isn't background, but confidence is too low to trust:
            if background_idx is not None and class_name != "background" and confidence < BACKGROUND_THRESHOLD:
                class_name = "background"
                # Set a logical confidence for fallback instead of raw low background probability
                confidence = float(np.max([probs[background_idx], 1.0 - confidence]))
                
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        app.logger.error("Model inference failed:\n" + tb)
        return jsonify({"error": f"Model inference failed: {e}", "trace": tb}), 500
    return jsonify({"class": class_name, "confidence": confidence, "probs": probs.tolist()})


if __name__ == "__main__":
    print("Loaded model:", MODEL_PATH, "->", "present" if model is not None else "MISSING")
    app.run(host="0.0.0.0", port=5000)
