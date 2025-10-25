import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader

# ----------------------------
# Paths
# ----------------------------
outputs_folder = "/content/drive/MyDrive/deepfake_ftca/outputs_gpu"

cnn_real_path = os.path.join(outputs_folder, "cnn_inputs_real.npy")
cnn_fake_path = os.path.join(outputs_folder, "cnn_inputs_fake.npy")
drift_real_path = os.path.join(outputs_folder, "drift_features_real.npy")
drift_fake_path = os.path.join(outputs_folder, "drift_features_fake.npy")
labels_real_path = os.path.join(outputs_folder, "labels_real.npy")
labels_fake_path = os.path.join(outputs_folder, "labels_fake.npy")

# ----------------------------
# Memory-mapped Dataset
# ----------------------------
class NpyDataset(Dataset):
    def __init__(self, cnn_real, cnn_fake, drift_real, drift_fake, labels_real, labels_fake):
        print("🔹 Loading dataset files using memory-mapping (safe for large arrays)...")

        self.cnn_real = np.load(cnn_real, mmap_mode='r', allow_pickle=True)
        print("✅ Loaded cnn_real:", self.cnn_real.shape)
        self.cnn_fake = np.load(cnn_fake, mmap_mode='r', allow_pickle=True)
        print("✅ Loaded cnn_fake:", self.cnn_fake.shape)
        self.drift_real = np.load(drift_real, mmap_mode='r', allow_pickle=True)
        print("✅ Loaded drift_real:", self.drift_real.shape)
        self.drift_fake = np.load(drift_fake, mmap_mode='r', allow_pickle=True)
        print("✅ Loaded drift_fake:", self.drift_fake.shape)
        self.labels_real = np.load(labels_real, mmap_mode='r', allow_pickle=True)
        self.labels_fake = np.load(labels_fake, mmap_mode='r', allow_pickle=True)
        print("✅ Labels loaded: real =", len(self.labels_real), ", fake =", len(self.labels_fake))

        self.length_real = len(self.labels_real)
        self.length_fake = len(self.labels_fake)
        self.total_length = self.length_real + self.length_fake
        print(f"📊 Dataset ready. Total samples: {self.total_length}")

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        if idx < self.length_real:
            x = torch.tensor(self.cnn_real[idx], dtype=torch.float32)
            f = torch.tensor(self.drift_real[idx], dtype=torch.float32)
            y = torch.tensor(self.labels_real[idx], dtype=torch.float32)
        else:
            idx_fake = idx - self.length_real
            x = torch.tensor(self.cnn_fake[idx_fake], dtype=torch.float32)
            f = torch.tensor(self.drift_fake[idx_fake], dtype=torch.float32)
            y = torch.tensor(self.labels_fake[idx_fake], dtype=torch.float32)
        return x, f, y.unsqueeze(0)

# ----------------------------
# Transformer Encoder
# ----------------------------
class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=32, num_heads=4, ff_dim=128, num_layers=2, dropout=0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=ff_dim, dropout=dropout
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, x):
        x = self.encoder(x)
        return x.mean(dim=1)

# ----------------------------
# 3D-CNN + Transformer + FTCA
# ----------------------------
class FTCA_3DCNN_Transformer(nn.Module):
    def __init__(self, in_ch=3, drift_feat_dim=5, num_classes=1):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv3d(in_ch, 16, 3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)),
            nn.Conv3d(16, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((4, 1, 1))
        )
        self.transformer = TransformerEncoder(embed_dim=32)
        self.fc = nn.Linear(32 + drift_feat_dim, num_classes)

    def forward(self, x, drift_features):
        x = self.cnn(x)
        x = x.squeeze(-1).squeeze(-1).permute(0, 2, 1)
        x = self.transformer(x)
        x = torch.cat([x, drift_features], dim=1)
        return torch.sigmoid(self.fc(x))

# ----------------------------
# Training
# ----------------------------
if __name__ == "__main__":
    print("🚀 Starting FTCA 3D-CNN + Transformer training...")
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("📦 Preparing dataset...")
    dataset = NpyDataset(cnn_real_path, cnn_fake_path, drift_real_path, drift_fake_path, labels_real_path, labels_fake_path)
    print(f"Total samples: {len(dataset)}")

    print("🧩 Splitting dataset into train and validation sets...")
    split_idx = int(0.8 * len(dataset))
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [split_idx, len(dataset) - split_idx])
    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=2)
    print("✅ DataLoaders ready!")

    print("🧠 Initializing model...")
    model = FTCA_3DCNN_Transformer(in_ch=3, drift_feat_dim=5).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    epochs = 15

    print("🔥 Starting training loop...")
    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch + 1}/{epochs} ---")
        model.train()
        train_loss = 0
        start_epoch = time.time()

        for i, (xb, fb, yb) in enumerate(train_loader):
            xb, fb, yb = xb.to(device), fb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb, fb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)

            if (i + 1) % 20 == 0:
                print(f"   🔁 Batch {i + 1}/{len(train_loader)} - Current loss: {loss.item():.4f}")

        train_loss /= len(train_loader.dataset)
        print(f"✅ Epoch {epoch + 1} train loss: {train_loss:.4f}")

        # Validation
        model.eval()
        val_correct = 0
        with torch.no_grad():
            for xb, fb, yb in val_loader:
                xb, fb, yb = xb.to(device), fb.to(device), yb.to(device)
                out = model(xb, fb)
                val_pred = (out > 0.5).float()
                val_correct += (val_pred == yb).sum().item()
        val_acc = val_correct / len(val_loader.dataset)
        elapsed_epoch = (time.time() - start_epoch) / 60
        total_elapsed = (time.time() - start_time) / 60
        print(f"📈 Epoch {epoch + 1} - Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.3f} | Time: {elapsed_epoch:.2f} min (Total {total_elapsed:.2f} min)")

        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            ckpt_path = os.path.join(outputs_folder, f"ftca_epoch{epoch + 1}.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"💾 Saved checkpoint: {ckpt_path}")

    print("🎯 Training completed successfully!")
