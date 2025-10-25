# =============================================
# FTCA 3D-CNN + Transformer with Explicit Drift Reasoning
# =============================================
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# ===============================
# Dataset
# ===============================
class DeepfakeDataset(Dataset):
    def __init__(self, dataset_path, split="train"):
        self.real = np.load(f"{dataset_path}/cnn_inputs_real.npy", mmap_mode="r")
        self.fake = np.load(f"{dataset_path}/cnn_inputs_fake.npy", mmap_mode="r")

        self.labels_real = np.zeros(len(self.real), dtype=np.int64)
        self.labels_fake = np.ones(len(self.fake), dtype=np.int64)
        self.length_real = len(self.real)
        self.length_fake = len(self.fake)

        # Combine and shuffle indices
        self.total_length = self.length_real + self.length_fake
        self.indices = np.arange(self.total_length)
        np.random.seed(42)
        np.random.shuffle(self.indices)

        split_idx = int(0.8 * self.total_length)
        if split == "train":
            self.indices = self.indices[:split_idx]
        else:
            self.indices = self.indices[split_idx:]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        index = self.indices[idx]
        if index < self.length_real:
            x = torch.tensor(self.real[index], dtype=torch.float32)
            y = torch.tensor(0.0)
        else:
            index -= self.length_real
            x = torch.tensor(self.fake[index], dtype=torch.float32)
            y = torch.tensor(1.0)
        return x, y.unsqueeze(0)

# ===============================
# Residual 3D Block
# ===============================
class ResidualBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, stride, 1)
        self.bn1 = nn.BatchNorm3d(out_ch)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, 1, 1)
        self.bn2 = nn.BatchNorm3d(out_ch)
        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 1, stride),
                nn.BatchNorm3d(out_ch)
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample:
            identity = self.downsample(identity)
        out += identity
        return self.relu(out)

# ===============================
# FTCA 3D Transformer with Drift Reasoning
# ===============================
class FTCA3DTransformer(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()

        # Input: RGB + Drift → 4 channels
        self.layer1 = ResidualBlock3D(4, 32)
        self.layer2 = ResidualBlock3D(32, 64, stride=2)
        self.layer3 = ResidualBlock3D(64, 128, stride=2)
        self.pool = nn.AdaptiveAvgPool3d((4, 4, 4))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128, nhead=4, batch_first=True, dropout=0.2, activation="relu", norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.fc = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def compute_drift(self, x):
        # x shape: [B, 3, T, H, W]
        drift = torch.zeros_like(x)
        drift[:, :, 1:, :, :] = torch.abs(x[:, :, 1:, :, :] - x[:, :, :-1, :, :])
        drift_mean = drift.mean(dim=1, keepdim=True)  # average across RGB channels
        return drift_mean  # shape: [B, 1, T, H, W]

    def forward(self, x):
        drift = self.compute_drift(x)
        x = torch.cat([x, drift], dim=1)  # now 4 channels (RGB + drift)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)
        x = x.view(x.size(0), 128, -1).transpose(1, 2)  # [B, seq_len, features]
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.fc(x)

# ===============================
# Training
# ===============================
def train_model(dataset_path, device, epochs=25, batch_size=4, lr=1e-4):
    train_dataset = DeepfakeDataset(dataset_path, split="train")
    val_dataset = DeepfakeDataset(dataset_path, split="val")

    class_counts = [len(train_dataset.real), len(train_dataset.fake)]
    class_weights = [1.0 / class_counts[0], 1.0 / class_counts[1]]
    samples_weight = np.array([class_weights[0]] * len(train_dataset.real) + [class_weights[1]] * len(train_dataset.fake))
    samples_weight = torch.from_numpy(samples_weight[train_dataset.indices])
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    model = FTCA3DTransformer().to(device)
    pos_weight = torch.tensor([class_counts[0] / class_counts[1]]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scaler = torch.cuda.amp.GradScaler()

    best_val_acc = 0.0
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for xb, yb in pbar:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                logits = model(xb)
                loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * xb.size(0)
            pbar.set_postfix({"loss": loss.item()})

        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                preds = (torch.sigmoid(logits) > 0.5).float()
                val_correct += (preds == yb).sum().item()
                val_total += yb.numel()

        val_acc = val_correct / val_total
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"{dataset_path}/ftca_drift_best.pt")
            print("💾 Saved new best model!")

    print("✅ Training Completed with Drift Reasoning.")
    return model

# ===============================
# Evaluation
# ===============================
def evaluate_model(dataset_path, device, model_path):
    val_dataset = DeepfakeDataset(dataset_path, split="val")
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=1)
    model = FTCA3DTransformer().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    y_true, y_pred, y_score = [], [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            outputs = model(xb)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            y_true.extend(yb.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_score.extend(probs[:, 0].cpu().numpy())

    print("\n📊 Classification Report:\n", classification_report(y_true, y_pred, digits=4))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
    plt.show()

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.legend()
    plt.show()
    print(f"✅ ROC AUC Score: {roc_auc:.4f}")

# ===============================
# Main
# ===============================
if __name__ == "__main__":
    dataset_path = "/content/drive/MyDrive/deepfake_ftca/outputs_gpu"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = train_model(dataset_path, device, epochs=25, batch_size=4, lr=1e-4)
    evaluate_model(dataset_path, device, f"{dataset_path}/ftca_drift_best.pt")
