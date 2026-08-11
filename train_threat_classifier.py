"""
Starter: edge-deployable audio threat classifier via YAMNet transfer learning.

Pipeline:
  raw audio (wav, 16kHz mono) -> YAMNet frozen embeddings -> small trainable head
  -> classes: gunshot, chainsaw, vehicle, [your confusers: firework, backfire, dumpster_lid], background

Run in Colab (free GPU) for speed, or locally on CPU (still fast since YAMNet is frozen).

pip install tensorflow tensorflow-hub librosa numpy scikit-learn
"""

import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import librosa
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# 1. Load frozen YAMNet
# ---------------------------------------------------------------------------
YAMNET_URL = "https://tfhub.dev/google/yamnet/1"
yamnet_model = hub.load(YAMNET_URL)

def get_embedding(wav_path, target_sr=16000):
    """Load audio, resample to 16kHz mono, return mean-pooled YAMNet embedding."""
    audio, sr = librosa.load(wav_path, sr=target_sr, mono=True)
    # YAMNet expects float32 waveform in [-1, 1]
    scores, embeddings, spectrogram = yamnet_model(audio)
    # embeddings: (num_frames, 1024) -- mean pool across frames for a clip-level vector
    return embeddings.numpy().mean(axis=0)

# ---------------------------------------------------------------------------
# 2. Build dataset from a folder structure:
#    data/
#      gunshot/*.wav
#      chainsaw/*.wav
#      vehicle/*.wav
#      firework/*.wav        <- hard negative / confuser
#      backfire/*.wav        <- hard negative / confuser
#      dumpster_lid/*.wav    <- hard negative / confuser
#      background/*.wav
# ---------------------------------------------------------------------------
DATA_DIR = "data"
CLASSES = sorted(os.listdir(DATA_DIR))  # folder names = labels
print("Classes found:", CLASSES)

X, y = [], []
for label_idx, class_name in enumerate(CLASSES):
    class_dir = os.path.join(DATA_DIR, class_name)
    for fname in os.listdir(class_dir):
        if not fname.endswith(".wav"):
            continue
        try:
            emb = get_embedding(os.path.join(class_dir, fname))
            X.append(emb)
            y.append(label_idx)
        except Exception as e:
            print(f"Skipping {fname}: {e}")

X = np.array(X)
y = np.array(y)
print("Dataset shape:", X.shape, "labels:", y.shape)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ---------------------------------------------------------------------------
# 3. Train small classifier head on top of frozen embeddings
# ---------------------------------------------------------------------------
num_classes = len(CLASSES)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1024,)),
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(num_classes, activation="softmax"),
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=16,
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
)

model.save("threat_classifier_head.keras")

# ---------------------------------------------------------------------------
# 4. Check false-positive behavior explicitly on the confuser classes
# ---------------------------------------------------------------------------
from sklearn.metrics import classification_report, confusion_matrix

y_pred = model.predict(X_val).argmax(axis=1)
print(classification_report(y_val, y_pred, target_names=CLASSES))
print("Confusion matrix:\n", confusion_matrix(y_val, y_pred))

# ---------------------------------------------------------------------------
# 5. Export for edge inference (TFLite, quantized for microcontroller/edge daemon)
# ---------------------------------------------------------------------------
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # int8-ish quantization
tflite_model = converter.convert()

with open("threat_classifier_head.tflite", "wb") as f:
    f.write(tflite_model)

print("Saved threat_classifier_head.tflite for edge deployment.")
print("NOTE: at inference time on the edge node, you still need to run YAMNet")
print("(or a quantized/distilled version of it) on the rolling 2s buffer to")
print("produce the embedding this head consumes. For a true microcontroller-class")
print("target, look at YAMNet's smaller cousin or a distilled MobileNet-audio model —")
print("full YAMNet may be too heavy for something like an ESP32; fine for a Pi/laptop daemon.")
