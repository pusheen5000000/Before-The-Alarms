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
BACKGROUND_THRESHOLD = float(os.getenv("BACKGROUND_THRESHOLD", "0.20"))

PROJECT_CLASSES = ["background", "gunshot", "chainsaw", "firework", "vehicle"]
THREAT_KEYWORDS = [
    "gunshot", "gunfire", "machine gun", "artillery", "cap gun",
    "chainsaw",
    "firework", "fire cracker",
    "vehicle", "car", "motor vehicle", "truck", "vehicle horn", "car horn",
    "light engine", "medium engine", "heavy engine", "engine starting",
    "engine knocking", "idling",
    "explosion",
    "siren",
    "glass",
]

PROJECT_KEYWORD_MAP = {
    "gunshot": ["gunshot", "gunfire", "machine gun", "artillery", "cap gun", "explosion"],
    "chainsaw": ["chainsaw"],
    "firework": ["firework", "firecracker", "fire cracker", "fireworks"],
    "vehicle": [
        "vehicle", "car", "truck", "motor vehicle",
        "light engine", "medium engine", "heavy engine",
        "engine starting", "engine knocking", "idling",
        "horn", "siren",
    ],
}

# Explicit YAMNet label index → project class mapping, built once at startup.
# This avoids the old substring matching that caused false hits (e.g. "car"
# matching "Carnatic music", "Scary music", "Shuffling cards"; "horn" matching
# "French horn"). Each YAMNet label is examined once, and only mapped if the
# keyword matches as a *complete word or phrase segment* within the label.
# The mapping is populated by build_label_index() after YAMNet loads.
YAMNET_LABEL_INDEX = {}   # {int yamnet_idx: str project_class}

# Labels that should NEVER map to vehicle even if they contain matching words.
# These prevent cross-contamination from compound AudioSet names.
VEHICLE_EXCLUDE = frozenset([
    "carnatic music", "scary music", "shuffling cards",
    "french horn", "english horn", "foghorn",
    "railroad car, train wagon", "train horn",
    "boat, water vehicle",  # fires on amplified mic noise (waterfall/wave)
])

DISAMBIGUATION_RULES = [
    # (class_to_promote, min_raw_yamnet_score_for_any_anchor, anchor_keywords, classes_to_demote)
    # "explosion" and "fireworks" are key — they fire reliably on gunshots even
    # after speaker/room coloring, where "gunshot"/"gunfire" confidence drops.
    # A real vehicle NEVER triggers any of these labels above 0.02.
    ("gunshot", 0.02, ["gunshot", "gunfire", "machine gun", "artillery", "explosion", "cap gun", "fireworks", "firecracker"], ["vehicle"]),
    ("gunshot", 0.04, ["gunshot", "gunfire", "machine gun", "explosion"], ["firework"]),
]

# Precomputed at startup after YAMNet loads: maps each disambiguation rule's
# anchor keywords to their YAMNet label indices for O(1) lookup instead of
# scanning all 521 labels with regex every inference.
ANCHOR_INDICES = {}  # populated by build_anchor_indices()

# Maximum number of YAMNet labels to include per class in noisy-OR.
# Vehicle has 23 matching labels; even tiny residual scores from unrelated audio
# compound to inflate the vehicle score when many labels weakly fire. Capping to
# the top-K strongest-firing labels per class prevents this accumulation while
# still letting genuinely vehicular audio (where multiple vehicle-specific labels
# fire strongly together) score high.
CLASS_TOP_K = {
    "gunshot": 6,     # only 6 labels, keep all
    "chainsaw": 3,
    "firework": 4,
    "vehicle": 5,     # only the 5 strongest vehicle labels count
}


def _word_match(keyword, label_lower):
    """Check if keyword appears as a complete word/phrase in the label.
    'car' should match 'Car' and 'Car alarm' but not 'Carnatic' or 'Scary'."""
    import re
    # Escape keyword for regex, look for word boundaries
    pattern = r'\b' + re.escape(keyword) + r'\b'
    return bool(re.search(pattern, label_lower))


def build_label_index(class_names):
    """Map each YAMNet label to at most one project class using word-boundary
    matching. Called once after YAMNet loads."""
    index = {}
    for idx, name in enumerate(class_names):
        lname = name.lower()
        # Check exclusion list first
        if lname in VEHICLE_EXCLUDE:
            continue
        for project_label, keywords in PROJECT_KEYWORD_MAP.items():
            if any(_word_match(kw, lname) for kw in keywords):
                index[idx] = project_label
                break
    return index


def build_anchor_indices(class_names):
    """Precompute which YAMNet label indices correspond to each disambiguation
    rule's anchor keywords. This lets the hot path do a direct index lookup
    instead of scanning all 521 labels with regex."""
    anchors = {}
    for rule_idx, (promote, min_score, keywords, demote) in enumerate(DISAMBIGUATION_RULES):
        indices = []
        for idx, name in enumerate(class_names):
            lname = name.lower()
            if any(_word_match(kw, lname) for kw in keywords):
                indices.append(idx)
        anchors[rule_idx] = indices
    return anchors


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
YAMNET_LABEL_INDEX = build_label_index(YAMNET_CLASSES)
ANCHOR_INDICES = build_anchor_indices(YAMNET_CLASSES)

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

    y = normalize_peak(y)

    return y.astype(np.float32)


# RMS energy floor: clips below this are mic noise / room tone. Returning them
# un-normalized (or skipping inference entirely) prevents YAMNet from
# interpreting amplified hiss as "Water" / "Waterfall" / "White noise".
# Typical idle mic RMS is 0.001-0.01; a real acoustic event is 0.02+.
RMS_GATE = float(os.getenv("RMS_GATE", "0.015"))


def normalize_peak(y, target_peak=0.9):
    """Normalize loud clips down and boost genuinely present signals up.
    Clips with RMS below RMS_GATE are returned as-is (they're just noise)."""
    rms = float(np.sqrt(np.mean(y ** 2))) if len(y) else 0.0
    if rms < RMS_GATE:
        return y
    peak = float(np.max(np.abs(y))) if len(y) else 0.0
    if peak < 1e-4:
        return y
    gain = target_peak / peak
    return y * gain


def project_score_vector(mean_scores, class_names):
    # Gather per-class scores from YAMNet labels.
    class_scores = {name: [] for name in PROJECT_CLASSES if name != "background"}

    # Per-label floor: YAMNet outputs below this are model noise, not
    # meaningful detections. Without this, white noise / ambient mic hiss
    # produces dozens of labels at 0.02-0.05 that compound via noisy-OR
    # into a false vehicle detection.
    LABEL_FLOOR = 0.08

    for idx, project_label in YAMNET_LABEL_INDEX.items():
        score = min(max(float(mean_scores[idx]), 0.0), 1.0)
        if score >= LABEL_FLOOR:
            class_scores[project_label].append(score)

    # Noisy-OR of only the top-K strongest-firing labels per class.
    # This prevents classes with many matching labels (vehicle=23) from
    # inflating on weak residual signals, while still allowing genuinely
    # strong multi-label detections to score high.
    score_map = {}
    for label, scores_list in class_scores.items():
        k = CLASS_TOP_K.get(label, 6)
        top_scores = sorted(scores_list, reverse=True)[:k]
        complement = 1.0
        for s in top_scores:
            complement *= (1.0 - s)
        score_map[label] = 1.0 - complement

    # --- Disambiguation ---
    # The core issue: gunshots played through speakers/room/mic produce
    # sustained low-frequency energy that lights up "Vehicle", "Motor vehicle",
    # "Heavy engine" etc at modest levels. Even with top-K capping, vehicle can
    # beat gunshot because the smeared transient lowers YAMNet's "Gunshot"
    # confidence. But a real vehicle NEVER produces any "Gunshot/gunfire" signal
    # at all -- so if we see even a modest gunshot anchor, the vehicle label is
    # a false positive from acoustic coloring.
    #
    # Strategy: if ANY gunshot-specific anchor label fires above a low floor
    # (0.02), AND vehicle currently leads, force gunshot to win. The floor is
    # deliberately very low because speakers + room reverb heavily attenuate
    # the sharp transient that YAMNet expects for a confident "Gunshot".
    best = max(score_map, key=score_map.get)
    for rule_idx, (promote, min_anchor, _keywords, demote_list) in enumerate(DISAMBIGUATION_RULES):
        if best not in demote_list:
            continue
        # Check precomputed anchor indices directly (no regex in hot path)
        anchor_hit = any(
            float(mean_scores[idx]) >= min_anchor
            for idx in ANCHOR_INDICES.get(rule_idx, [])
        )
        if anchor_hit:
            margin = 0.01
            needed = score_map[best] + margin
            if score_map[promote] < needed:
                score_map[promote] = min(needed, 0.99)
            best = promote

    return score_map


def classify_audio(audio):
    # Fast-path: if the clip is just mic noise, skip inference entirely.
    rms = float(np.sqrt(np.mean(audio ** 2))) if len(audio) else 0.0
    if rms < RMS_GATE:
        empty_scores = {c: 0.0 for c in PROJECT_CLASSES if c != "background"}
        empty_scores["background"] = 1.0
        empty_mean = np.zeros(len(YAMNET_CLASSES), dtype=np.float32)
        return "background", 1.0, empty_mean, empty_scores

    scores, _, _ = yamnet_model(audio)
    mean_scores = scores.numpy().mean(axis=0)
    score_map = project_score_vector(mean_scores, YAMNET_CLASSES)

    best_label = max(score_map, key=score_map.get)
    best_score = score_map[best_label]

    # "background" isn't scored directly -- it's the residual confidence that
    # none of the threat classes fired. Expose it alongside the threat scores
    # so the UI can render one complete, comparable bar per project class.
    display_scores = {label: float(value) for label, value in score_map.items()}
    display_scores["background"] = float(1.0 - best_score)

    # Nothing cleared the bar -> background. Otherwise, whichever threat
    # class scored highest wins, regardless of how the other classes scored.
    if best_score < BACKGROUND_THRESHOLD:
        return "background", float(1.0 - best_score), mean_scores, display_scores

    return best_label, float(best_score), mean_scores, display_scores


def top_yamnet_labels(mean_scores, k=5):
    """The raw YAMNet labels behind a decision. The project classes are
    keyword rollups of ~521 AudioSet labels, so surfacing the underlying
    winners is what makes a wrong prediction debuggable from the UI."""
    order = np.argsort(mean_scores)[::-1][:k]
    return [
        {"label": YAMNET_CLASSES[int(i)], "score": float(mean_scores[int(i)])}
        for i in order
    ]


@app.route("/", methods=["GET"])
def index():
    return send_from_directory("static", "index.html")


@app.route("/meta", methods=["GET"])
def meta():
    """Model/config metadata so the UI can render its class breakdown before
    the first prediction arrives, rather than hardcoding a duplicate list."""
    return jsonify({
        "classes": PROJECT_CLASSES,
        "threat_classes": [c for c in PROJECT_CLASSES if c != "background"],
        "threshold": BACKGROUND_THRESHOLD,
        "sample_rate": SR,
        "clip_secs": CLIP_SECS,
        "model": "YAMNet",
        "model_url": YAMNET_URL,
    })


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
        class_name, confidence, scores, class_scores = classify_audio(audio)
        probs = scores.astype(float)
        top_idx = int(np.argmax(probs))
        if class_name.lower() == "background":
            top_idx = int(np.argmax(probs))
        return jsonify({
            "class": class_name,
            "confidence": float(confidence),
            "probs": probs.tolist(),
            "top_index": top_idx,
            "classes": PROJECT_CLASSES,
            "scores": class_scores,
            "threshold": BACKGROUND_THRESHOLD,
            "top_labels": top_yamnet_labels(probs),
        })
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        app.logger.error("YAMNet inference failed:\n" + tb)
        return jsonify({"error": f"YAMNet inference failed: {e}", "trace": tb}), 500


if __name__ == "__main__":
    print("Loaded YAMNet model from:", YAMNET_URL)
    app.run(host="0.0.0.0", port=5000)