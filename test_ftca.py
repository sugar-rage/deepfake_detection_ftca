import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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

checkpoint_path = os.path.join(outputs_folder, "ftca_epoch15.pt")  # your trained model

# ----------------------------
# Dataset
# ----------------------------
class NpyDataset(Dataset):
    def __init__(self, cnn_real, cnn_fake, drift_real, drift_fake, labels_real, labels_fake):
        self.cnn_real = np.load(cnn_real, mmap_mode='r', allow_pickle=True)
        self.cnn_fake = np.load(cnn_fake, mmap_mode='r', allow_pickle=True)
        self.drift_real = np.load(drift_real, mmap_mode='r', allow_pickle=True)
        self.drift_fake = np.load(drift_fake, mmap_mode='r', allow_pickle=True)
        self.labels_real = np.load(labels_real, mmap_mode='r', allow_pickle=True)
        self.labels_fake = np.load(labels_fake, mmap_mode='r', allow_pickle=True)

        self.len_real = len(self.labels_real)
        self.len_fake = len(self.labels_fake)
        self.total_len = self.len_real + self.len_fake

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        if idx < self.len_real:
            x = torch.tensor(self.cnn_real[idx], dtype=torch.float32)
            f = torch.tensor(self.drift_real[idx], dtype=torch.float32)
            y = torch.tensor(self.labels_real[idx], dtype=torch.float32)
        else:
            idx_fake = idx - self.len_real
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
# FTCA 3D-CNN + Transformer
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
# Testing function
# ----------------------------
def test_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = NpyDataset(cnn_real_path, cnn_fake_path, drift_real_path, drift_fake_path, labels_real_path, labels_fake_path)
    loader = DataLoader(dataset, batch_size=2, shuffle=False)

    model = FTCA_3DCNN_Transformer(in_ch=3, drift_feat_dim=5).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for xb, fb, yb in loader:
            xb, fb, yb = xb.to(device), fb.to(device), yb.to(device)
            out = model(xb, fb)
            preds = (out > 0.5).float()
            y_true.extend(yb.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print("\n📊 Test Results:")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")

if __name__ == "__main__":
    test_model()
