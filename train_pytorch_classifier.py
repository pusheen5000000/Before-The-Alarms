"""Train a PyTorch audio classifier on folder-structured data/
"""

import os
import argparse
import random
from glob import glob

import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class AudioDataset(Dataset):
    def __init__(self, filepaths, labels, sample_rate=16000, clip_secs=2.0, n_mels=64, is_train=False):
        self.filepaths = filepaths
        self.labels = labels
        self.sr = sample_rate
        self.clip_len = int(sample_rate * clip_secs)
        self.n_mels = n_mels
        self.is_train = is_train

    def __len__(self):
        return len(self.filepaths)

    def _load(self, path):
        y, sr = librosa.load(path, sr=self.sr, mono=True)
        if len(y) < self.clip_len:
            pad_width = self.clip_len - len(y)
            y = np.pad(y, (0, pad_width), mode="constant")
        elif len(y) > self.clip_len:
            # Deterministic crop for validation/testing, random crop for training
            if self.is_train:
                start = np.random.randint(0, len(y) - self.clip_len + 1)
            else:
                start = 0
            y = y[start:start + self.clip_len]
        return y

    def _wav_to_melspec(self, y):
        # FIX: Check for silence / low RMS to avoid amplifying floor noise
        rms = np.sqrt(np.mean(y**2))
        if rms < 1e-4:
            return np.full((self.n_mels, int(np.ceil(self.clip_len / 512))), -1.0, dtype=np.float32)

        melspec = librosa.feature.melspectrogram(
            y=y,
            sr=self.sr,
            n_fft=1024,
            hop_length=512,
            n_mels=self.n_mels,
            power=2.0,
        )
        # Fixed dynamic range scaling instead of per-sample variance normalization
        log_mel = librosa.power_to_db(melspec, ref=np.max, top_db=80.0)
        log_mel = (log_mel / 40.0) + 1.0  # Maps [-80dB, 0dB] to [-1.0, 1.0]
        return log_mel.astype(np.float32)

    def __getitem__(self, idx):
        path = self.filepaths[idx]
        label = self.labels[idx]
        y = self._load(path)
        spec = self._wav_to_melspec(y)
        spec = np.expand_dims(spec, axis=0)
        return torch.from_numpy(spec), torch.tensor(label, dtype=torch.long)


class ImprovedCNN(nn.Module):
    def __init__(self, n_mels=64, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        # Adaptive pooling across spatial dimensions (height, width) -> (1, 1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Pool down to (batch, 64, 1, 1)
        x = self.pool(x)
        
        # Flatten to (batch, 64)
        x = torch.flatten(x, 1)
        
        # Linear layer expects (batch, 64) -> outputs (batch, num_classes)
        x = self.fc(x)
        return x


def find_files_and_labels(data_dir):
    classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    filepaths = []
    labels = []
    for idx, cls in enumerate(classes):
        cls_dir = os.path.join(data_dir, cls)
        for ext in ("wav", "mp3", "flac", "m4a", "ogg"):
            pattern = os.path.join(cls_dir, f"**/*.{ext}")
            for p in glob(pattern, recursive=True):
                filepaths.append(p)
                labels.append(idx)
    return classes, filepaths, labels


def train_epoch(model, loader, opt, device):
    model.train()
    total = 0
    correct = 0
    loss_sum = 0.0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        opt.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        opt.step()
        loss_sum += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)
    return loss_sum / total, correct / total


def eval_epoch(model, loader, device):
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss_sum += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)
    return loss_sum / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data", help="data root with class subfolders")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clip-secs", type=float, default=2.0)
    parser.add_argument("--n-mels", type=int, default=64)
    args = parser.parse_args()

    set_seed(args.seed)

    classes, filepaths, labels = find_files_and_labels(args.data_dir)
    if not classes:
        print("No class folders found in", args.data_dir)
        return
    print("Classes:", classes)

    # Group files by original non-augmented source to prevent data leakage
    orig_files, orig_labels = [], []
    aug_files, aug_labels = [], []

    for f, l in zip(filepaths, labels):
        if "_aug" in os.path.basename(f):
            aug_files.append(f)
            aug_labels.append(l)
        else:
            orig_files.append(f)
            orig_labels.append(l)

    # Split only non-augmented originals
    X_train_orig, X_val, y_train_orig, y_val = train_test_split(
        orig_files, orig_labels, test_size=0.2, stratify=orig_labels, random_state=args.seed
    )

    # Combine training set with augmented files
    X_train = X_train_orig + aug_files
    y_train = y_train_orig + aug_labels

    train_ds = AudioDataset(X_train, y_train, clip_secs=args.clip_secs, n_mels=args.n_mels, is_train=True)
    val_ds = AudioDataset(X_val, y_val, clip_secs=args.clip_secs, n_mels=args.n_mels, is_train=False)

    class_counts = np.bincount(np.array(y_train), minlength=len(classes)).astype(np.float32)
    class_weights = 1.0 / np.clip(class_counts, 1.0, None)
    sample_weights = [class_weights[label] for label in y_train]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImprovedCNN(n_mels=args.n_mels, num_classes=len(classes)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, opt, device)
        val_loss, val_acc = eval_epoch(model, val_loader, device)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} train_acc={train_acc:.3f} | val_loss={val_loss:.4f} val_acc={val_acc:.3f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "classes": classes,
            }, "pytorch_threat_model.pt")
            print(f"Saved best model (val_acc={best_val_acc:.3f}) to pytorch_threat_model.pt")


if __name__ == "__main__":
    main()