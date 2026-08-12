"""
FALLBACK / Plan B: raw pretrained YAMNet, no custom training required.

Uses YAMNet's built-in AudioSet classes (521 total) directly -- it already
knows "Gunshot, gunfire", "Chainsaw", "Fireworks", "Vehicle", "Explosion",
etc. out of the box. This script just runs inference and reports scores
for the classes relevant to your project.

Use this if your custom classifier head (train_threat_classifier.py) isn't
ready in time, or as a sanity-check baseline to compare your custom model
against.

Usage:
    python3 yamnet_baseline.py path/to/audiofile.wav
    python3 yamnet_baseline.py path/to/folder_of_wavs/

pip install tensorflow tensorflow-hub librosa numpy
"""

import sys
import os
import csv
import io
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import librosa

# ---------------------------------------------------------------------------
# Load YAMNet + its class map (521 AudioSet class names)
# ---------------------------------------------------------------------------
YAMNET_URL = "https://tfhub.dev/google/yamnet/1"
print("Loading YAMNet (first run downloads ~15MB, cached after)...")
yamnet_model = hub.load(YAMNET_URL)

class_map_path = yamnet_model.class_map_path().numpy().decode("utf-8")
class_names = []
with tf.io.gfile.GFile(class_map_path) as f:
    reader = csv.reader(f)
    next(reader)  # skip header row
    for row in reader:
        class_names.append(row[2])  # display_name column

# Classes relevant to this project -- match by substring, case-insensitive
THREAT_KEYWORDS = [
    "gunshot", "gunfire", "machine gun", "artillery", "cap gun",
    "chainsaw",
    "firework", "fire cracker",
    "vehicle", "car", "engine", "motor vehicle", "truck",
    "explosion",
    "siren",
    "glass",
]

def matching_indices():
    matches = {}
    for idx, name in enumerate(class_names):
        lname = name.lower()
        for kw in THREAT_KEYWORDS:
            if kw in lname:
                matches.setdefault(kw, []).append((idx, name))
    return matches

RELEVANT = matching_indices()


def classify_file(wav_path, sr=16000, top_n=8):
    audio, _ = librosa.load(wav_path, sr=sr, mono=True)
    scores, embeddings, spectrogram = yamnet_model(audio)
    # scores: (num_frames, 521) -- mean pool across the whole clip
    mean_scores = scores.numpy().mean(axis=0)

    print(f"\n=== {os.path.basename(wav_path)} ===")

    # Top N overall predictions, for general sanity-checking
    top_idx = np.argsort(mean_scores)[::-1][:top_n]
    print(f"Top {top_n} overall predictions:")
    for i in top_idx:
        print(f"  {class_names[i]:<30s} {mean_scores[i]:.3f}")

    # Scores specifically for your threat-relevant classes
    print("Relevant threat-class scores:")
    seen = set()
    for kw, entries in RELEVANT.items():
        for idx, name in entries:
            if idx in seen:
                continue
            seen.add(idx)
            print(f"  {name:<30s} {mean_scores[idx]:.3f}")

    return mean_scores


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 yamnet_baseline.py <audio_file_or_folder>")
        sys.exit(1)

    target = sys.argv[1]
    if os.path.isdir(target):
        audio_files = [
            f for f in sorted(os.listdir(target))
            if f.lower().endswith((".wav", ".mp3", ".flac", ".m4a"))
        ]
        if not audio_files:
            print(f"No audio files found directly inside '{target}'.")
            print("(This script doesn't look inside subfolders -- point it at")
            print(" a specific class folder like data/gunshot, not data/ itself.)")
            sys.exit(1)
        for fname in audio_files:
            classify_file(os.path.join(target, fname))
    else:
        classify_file(target)