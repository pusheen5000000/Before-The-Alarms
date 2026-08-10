"""
Flattens a nested gunshot dataset (e.g. data/gunshot/AK-47/*.wav,
data/gunshot/AK-12/*.wav, data/gunshot/IMI Desert Eagle/*.wav)
into flat files directly inside data/gunshot/, renaming to avoid
collisions between subfolders that might reuse filenames like "1.wav".

Run this from wherever your data/ folder lives:
    python3 flatten_gunshot.py
"""

import os
import shutil

GUNSHOT_DIR = "data/gunshot"
AUDIO_EXTS = (".wav", ".mp3", ".flac", ".m4a", ".ogg")

moved = 0
for root, dirs, files in os.walk(GUNSHOT_DIR):
    if root == GUNSHOT_DIR:
        continue  # skip files already flat at the top level
    subfolder_name = os.path.relpath(root, GUNSHOT_DIR).replace(os.sep, "_")
    for fname in files:
        if not fname.lower().endswith(AUDIO_EXTS):
            continue
        src = os.path.join(root, fname)
        # prefix with subfolder name so "1.wav" from AK-47 and AK-12 don't collide
        dst_name = f"{subfolder_name}_{fname}"
        dst = os.path.join(GUNSHOT_DIR, dst_name)
        shutil.move(src, dst)
        moved += 1

# remove now-empty subfolders
for root, dirs, files in os.walk(GUNSHOT_DIR, topdown=False):
    if root != GUNSHOT_DIR and not os.listdir(root):
        os.rmdir(root)

print(f"Moved {moved} audio files into {GUNSHOT_DIR}/ (flat).")
print("Remaining contents:", os.listdir(GUNSHOT_DIR))
