# ----------------------------
# FTCA + 3D-CNN + Transformer Hybrid
# ----------------------------
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# ----------------------------
# Mount Google Drive (uncomment when running in Colab)
# ----------------------------
# from google.colab import drive
# drive.mount('/content/drive')

# ----------------------------
# Paths
# ----------------------------
outputs_folder = "/content/drive/MyDrive/deepfake_project/outputs"
os.makedirs(outputs_folder, exist_ok=True)

cnn_inputs_path = os.path.join(outputs_folder, "cnn_inputs.npy")
drift_features_path = os.path.join(outputs_folder, "drift_features.npy")
labels_path = os.path.join(outputs_folder, "labels.npy")

# ----------------------------
# Transformer Encoder Block
# ----------------------------
class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=32, num_heads=4, ff_dim=128, num_layers=2, dropout=0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim, dropout=dropout
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, x):
        # x: (B, T, D)
        x = self.encoder(x)
        return x.mean(dim=1)  # average pooling across time


# ----------------------------
# Hybrid Model: 3D-CNN + FTCA + Transformer
# ----------------------------
class FTCA_3DCNN_Transformer(nn.Module):
    def __init__(self, in_ch=3, drift_feat_dim=5, num_classes=1):
        super().__init__()

        # 3D CNN feature extractor
        self.cnn = nn.Sequential(
            nn.Conv3d(in_ch, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)),

            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((4, 1, 1))  # output shape: (B, 32, T', 1, 1)
        )

        self.transformer = TransformerEncoder(embed_dim=32, num_heads=4, num_layers=2)
        self.fc = nn.Linear(32 + drift_feat_dim, num_classes)

    def forward(self, x, drift_features):
        x = self.cnn(x)  # (B, 32, T, 1, 1)
        x = x.squeeze(-1).squeeze(-1).permute(0, 2, 1)  # (B, T, 32)
        x = self.transformer(x)  # (B, 32)
        x = torch.cat([x, drift_features], dim=1)
        return torch.sigmoid(self.fc(x))


# ----------------------------
# Training Script
# ----------------------------
if __name__ == "__main__":
    start_time = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    if not (os.path.exists(cnn_inputs_path) and os.path.exists(drift_features_path) and os.path.exists(labels_path)):
        raise FileNotFoundError(f"Missing one or more .npy files in {outputs_folder}")

    cnn_inputs = np.load(cnn_inputs_path)
    drift_features = np.load(drift_features_path)
    labels = np.load(labels_path)

    X = torch.tensor(cnn_inputs, dtype=torch.float32)
    F = torch.tensor(drift_features, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

    # Split
    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    F_train, F_val = F[:split], F[split:]
    y_train, y_val = y[:split], y[split:]

    train_loader = DataLoader(TensorDataset(X_train, F_train, y_train), batch_size=2, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, F_val, y_val), batch_size=2, shuffle=False)

    model = FTCA_3DCNN_Transformer(in_ch=X.shape[1], drift_feat_dim=F.shape[1]).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epochs = 15
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for xb, fb, yb in train_loader:
            xb, fb, yb = xb.to(device), fb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb, fb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)

        train_loss /= len(train_loader.dataset)

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
        elapsed = (time.time() - start_time) / 60
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Val Acc: {val_acc:.3f} - Time: {elapsed:.2f} min")

        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            save_path = os.path.join(outputs_folder, f"ftca_transformer_epoch{epoch+1}.pt")
            torch.save(model.state_dict(), save_path)
            print(f"✅ Saved checkpoint at: {save_path}")

    # Total time
    total_time = (time.time() - start_time) / 60
    print(f"\n🏁 Training complete in {total_time:.2f} minutes")

    # Quick validation check
    print("\nSample predictions on validation set:")
    model.eval()
    with torch.no_grad():
        for i in range(min(10, len(X_val))):
            xb = X_val[i].unsqueeze(0).to(device)
            fb = F_val[i].unsqueeze(0).to(device)
            y_true = y_val[i].item()
            prob = model(xb, fb).item()
            print(f"Sample {i+1}: True={int(y_true)}, Pred={prob:.3f}, Label={1 if prob>0.5 else 0}")
