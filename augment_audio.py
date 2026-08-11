"""
Augments small audio classes by generating pitch/time/noise/volume variations
of existing clips, so thin classes (chainsaw, firework) get closer in size to
your larger ones without needing more raw downloads.

Each augmentation is a genuinely different-sounding variant (not a duplicate),
which helps the model learn what's invariant about "chainsaw-ness" vs.
"firework-ness" instead of memorizing your ~80 exact clips.

Usage:
    python3 augment_audio.py

Edit TARGETS below to set how many total clips you want per class.
Only classes listed in TARGETS get augmented; others are left alone.

pip install librosa soundfile numpy
"""

import os
import random
import numpy as np
import librosa
import soundfile as sf

DATA_DIR = "data"
SR = 16000  # match YAMNet's expected sample rate

# class_name -> desired total clip count after augmentation
TARGETS = {
    "chainsaw": 320,
    "firework": 320,
    "vehicle": 320,
}

AUDIO_EXTS = (".wav", ".mp3", ".flac", ".m4a", ".ogg")


def augment_clip(y, sr):
    """Apply a random combination of augmentations to one waveform."""
    out = y.copy()

    # Pitch shift: +/- up to 3 semitones
    if random.random() < 0.6:
        n_steps = random.uniform(-3, 3)
        out = librosa.effects.pitch_shift(out, sr=sr, n_steps=n_steps)

    # Time stretch: 0.85x-1.15x speed
    if random.random() < 0.6:
        rate = random.uniform(0.85, 1.15)
        out = librosa.effects.time_stretch(out, rate=rate)

    # Additive background noise at a random SNR
    if random.random() < 0.7:
        noise = np.random.randn(len(out))
        snr_db = random.uniform(10, 25)  # higher = quieter noise
        sig_power = np.mean(out ** 2) + 1e-10
        noise_power = np.mean(noise ** 2) + 1e-10
        target_noise_power = sig_power / (10 ** (snr_db / 10))
        noise = noise * np.sqrt(target_noise_power / noise_power)
        out = out + noise

    # Volume/gain change
    if random.random() < 0.5:
        gain = random.uniform(0.6, 1.4)
        out = out * gain

    # Prevent clipping
    max_val = np.max(np.abs(out)) + 1e-10
    if max_val > 1.0:
        out = out / max_val

    return out.astype(np.float32)


def augment_class(class_name, target_count):
    class_dir = os.path.join(DATA_DIR, class_name)
    existing_files = [
        f for f in os.listdir(class_dir) if f.lower().endswith(AUDIO_EXTS)
    ]
    # only touch files that are originals (skip files we already generated
    # in a previous run, marked with "_aug" in the name)
    originals = [f for f in existing_files if "_aug" not in f]

    if not originals:
        print(f"[{class_name}] No original files found, skipping.")
        return

    current_count = len(existing_files)
    needed = target_count - current_count
    if needed <= 0:
        print(f"[{class_name}] Already at {current_count} >= target {target_count}, skipping.")
        return

    print(f"[{class_name}] {current_count} clips -> generating {needed} augmented clips "
          f"from {len(originals)} originals...")

    generated = 0
    attempt = 0
    while generated < needed:
        attempt += 1
        src_name = originals[generated % len(originals)]
        src_path = os.path.join(class_dir, src_name)
        try:
            y, sr = librosa.load(src_path, sr=SR, mono=True)
            y_aug = augment_clip(y, sr)
            base = os.path.splitext(src_name)[0]
            out_name = f"{base}_aug{generated}.wav"
            out_path = os.path.join(class_dir, out_name)
            sf.write(out_path, y_aug, SR)
            generated += 1
        except Exception as e:
            print(f"  Skipping {src_name} due to error: {e}")
            if attempt > needed * 3:  # avoid infinite loop on persistent errors
                break

    print(f"[{class_name}] Done. Now {current_count + generated} total clips.")


if __name__ == "__main__":
    for class_name, target in TARGETS.items():
        class_dir = os.path.join(DATA_DIR, class_name)
        if not os.path.isdir(class_dir):
            print(f"[{class_name}] Folder not found at {class_dir}, skipping.")
            continue
        augment_class(class_name, target)

    print("\nDone. Re-check counts with: for d in data/*/; do echo $d $(ls \"$d\" | wc -l); done")
