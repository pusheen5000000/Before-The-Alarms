from flask import Flask, request, jsonify, send_from_directory
from werkzeug.exceptions import HTTPException
import os
import csv
import tempfile
import numpy as np
import librosa
import tensorflow as tf
import tensorflow_hub as hub

YAMNET_URL = "https://tfhub.dev/google/yamnet/1"
SR = 16000
CLIP_SECS = 2.0
BACKGROUND_THRESHOLD = float(os.getenv("BACKGROUND_THRESHOLD", "0.10"))

PROJECT_CLASSES = ["background", "gunshot", "chainsaw", "firework", "vehicle"]
THREAT_KEYWORDS = [
    "gunshot", "gunfire", "machine gun", "artillery", "cap gun",
    "chainsaw",
    "firework", "fire cracker",
    "vehicle", "car", "engine", "motor vehicle", "truck", "vehicle horn", "car horn",
    "explosion",
    "siren",
    "glass",
]

PROJECT_KEYWORD_MAP = {
    "gunshot": ["gunshot", "gunfire", "machine gun", "artillery", "cap gun", "explosion"],
    "chainsaw": ["chainsaw"],
    "firework": ["firework", "fire cracker", "firecracker"],
    "vehicle": ["vehicle", "car", "truck", "motor vehicle", "engine", "horn", "siren"],
}


def load_yamnet():
    print("Loading YAMNet...")
    model = hub.load(YAMNET_URL)
    class_map_path = model.class_map_path().numpy().decode("utf-8")
    class_names = []
    with tf.io.gfile.GFile(class_map_path) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            class_names.append(row[2])
    return model, class_names


def matching_indices(class_names):
    matches = {}
    for idx, name in enumerate(class_names):
        lname = name.lower()
        for kw in THREAT_KEYWORDS:
            if kw in lname:
                matches.setdefault(kw, []).append((idx, name))
    return matches


yamnet_model, YAMNET_CLASSES = load_yamnet()
YAMNET_RELEVANT = matching_indices(YAMNET_CLASSES)

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


def preprocess_file(path, sr=SR, clip_secs=CLIP_SECS):
    y, _ = librosa.load(path, sr=sr, mono=True)
    clip_len = int(sr * clip_secs)
    if len(y) < clip_len:
        y = np.pad(y, (0, clip_len - len(y)), mode="constant")
    elif len(y) > clip_len:
        y = y[:clip_len]
    return y.astype(np.float32)


def project_score_vector(mean_scores, class_names):
    # Only the actual threat classes get scored here. YAMNet's per-class
    # scores are independent sigmoids, not a softmax, so they don't sum to
    # 1 -- background must NOT be derived as "1 - sum(other scores)", since
    # that formula inflates background almost every time (see classify_audio).
    score_map = {name: 0.0 for name in PROJECT_CLASSES if name != "background"}

    for idx, name in enumerate(class_names):
        lname = name.lower()
        score = float(mean_scores[idx])
        if score <= 0:
            continue

        for project_label, keywords in PROJECT_KEYWORD_MAP.items():
            if any(keyword in lname for keyword in keywords):
                score_map[project_label] += score
                break

    return score_map


def classify_audio(audio):
    scores, _, _ = yamnet_model(audio)
    mean_scores = scores.numpy().mean(axis=0)
    score_map = project_score_vector(mean_scores, YAMNET_CLASSES)

    best_label = max(score_map, key=score_map.get)
    best_score = score_map[best_label]

    # Nothing cleared the bar -> background. Otherwise, whichever threat
    # class scored highest wins, regardless of how the other classes scored.
    if best_score < BACKGROUND_THRESHOLD:
        return "background", float(1.0 - best_score), mean_scores

    return best_label, float(best_score), mean_scores


@app.route("/", methods=["GET"])
def index():
    return send_from_directory("static", "index.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return jsonify({"info": "Send a POST multipart/form-data with field 'file' to get a prediction."})
    if "file" not in request.files:
        return jsonify({"error": "No file field in request"}), 400

    f = request.files["file"]
    filename = f.filename or ""
    ext = os.path.splitext(filename)[1] or ".webm"
    fd, tmp_path = tempfile.mkstemp(suffix=ext)
    os.close(fd)

    try:
        f.save(tmp_path)
        try:
            audio = preprocess_file(tmp_path)
        except Exception as e:
            import traceback, subprocess
            tb = traceback.format_exc()
            app.logger.warning("preprocess_file failed, attempting ffmpeg conversion:\n" + tb)
            converted = tmp_path + ".converted.wav"
            try:
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
                audio = preprocess_file(converted)
            except FileNotFoundError:
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
        class_name, confidence, scores = classify_audio(audio)
        probs = scores.astype(float)
        top_idx = int(np.argmax(probs))
        if class_name.lower() == "background":
            top_idx = int(np.argmax(probs))
        return jsonify({
            "class": class_name,
            "confidence": float(confidence),
            "probs": probs.tolist(),
            "top_index": top_idx,
        })
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        app.logger.error("YAMNet inference failed:\n" + tb)
        return jsonify({"error": f"YAMNet inference failed: {e}", "trace": tb}), 500


if __name__ == "__main__":
    print("Loaded YAMNet model from:", YAMNET_URL)
    app.run(host="0.0.0.0", port=5000)