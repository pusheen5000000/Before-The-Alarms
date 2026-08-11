"""Train a PyTorch audio classifier on folder-structured data/

Usage example:
  python train_pytorch_classifier.py --data-dir data --epochs 30 --batch-size 16

Expects `data/` to contain one subfolder per class, each with audio files.
This script computes log-mel spectrograms with librosa, trains a small CNN,
and saves `pytorch_threat_model.pt` on success.
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
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class AudioDataset(Dataset):
    def __init__(self, filepaths, labels, sample_rate=16000, clip_secs=2.0, n_mels=64):
        self.filepaths = filepaths
        self.labels = labels
        self.sr = sample_rate
        self.clip_len = int(sample_rate * clip_secs)
        self.n_mels = n_mels

    def __len__(self):
        return len(self.filepaths)

    def _load(self, path):
        y, sr = librosa.load(path, sr=self.sr, mono=True)
        if len(y) < self.clip_len:
            # pad
            pad_width = self.clip_len - len(y)
            y = np.pad(y, (0, pad_width), mode="constant")
        elif len(y) > self.clip_len:
            # random crop
            start = np.random.randint(0, len(y) - self.clip_len + 1)
            y = y[start:start + self.clip_len]
        return y

    def _wav_to_melspec(self, y):
        melspec = librosa.feature.melspectrogram(
            y=y,
            sr=self.sr,
            n_fft=1024,
            hop_length=512,
            n_mels=self.n_mels,
            power=2.0,
        )
        log_mel = librosa.power_to_db(melspec, ref=np.max)
        # normalize per-sample
        log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-6)
        return log_mel.astype(np.float32)

    def __getitem__(self, idx):
        path = self.filepaths[idx]
        label = self.labels[idx]
        y = self._load(path)
        spec = self._wav_to_melspec(y)
        # shape: (n_mels, time) -> add channel dim
        spec = np.expand_dims(spec, axis=0)
        return torch.from_numpy(spec), torch.tensor(label, dtype=torch.long)


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
        x = x.view(x.size(0), -1)
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
    X_train, X_val, y_train, y_val = train_test_split(
        filepaths, labels, test_size=0.2, stratify=labels, random_state=args.seed
    )

    train_ds = AudioDataset(X_train, y_train, clip_secs=args.clip_secs, n_mels=args.n_mels)
    val_ds = AudioDataset(X_val, y_val, clip_secs=args.clip_secs, n_mels=args.n_mels)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN(n_mels=args.n_mels, num_classes=len(classes)).to(device)
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
            print("Saved best model (val_acc=%.3f) to pytorch_threat_model.pt" % best_val_acc)


if __name__ == "__main__":
    main()
