# scripts/drift_features_gpu_batch.py
import os
import numpy as np
import torch
from load_frames import load_frames, create_windows
from compute_flow import compute_flow
from scipy.stats import entropy

T = 8  # window length
outputs_folder = "/content/drive/MyDrive/deepfake_ftca/outputs_gpu"
os.makedirs(outputs_folder, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def compute_drift_features(window):
    flow_mags = compute_flow(window)  # CPU
    if len(flow_mags) == 0:
        return np.zeros(5, dtype=np.float32)
    
    flow_tensor = torch.tensor(np.stack(flow_mags), dtype=torch.float32).to(device)
    mean_flows = flow_tensor.mean(dim=(1,2))
    std_flows = flow_tensor.std(dim=(1,2))
    drift_mean = torch.abs(mean_flows[1:] - mean_flows[:-1])
    drift_maps = torch.abs(flow_tensor[1:] - flow_tensor[:-1]).mean(dim=(1,2))
    all_flow = flow_tensor.flatten().cpu().numpy()
    hist, _ = np.histogram(all_flow, bins=16, range=(0, np.max(all_flow)+1e-6), density=True)
    motion_entropy = entropy(hist + 1e-8)
    
    return np.array([
        mean_flows.mean().item(),
        std_flows.mean().item(),
        drift_mean.mean().item() if drift_mean.numel() > 0 else 0.0,
        drift_mean.std().item() if drift_mean.numel() > 0 else 0.0,
        motion_entropy
    ], dtype=np.float32)

# -------------------------
# Process dataset in batches
# -------------------------
dataset_path = "/content/drive/MyDrive/deepfake_ftca/dataset"

for subfolder_name in ["real", "fake"]:
    folder_path = os.path.join(dataset_path, subfolder_name)
    label = 1 if subfolder_name == "real" else 0
    
    all_cnn_inputs = []
    all_drift_features = []
    all_labels = []

    clip_count = 0
    for clip_sub in sorted(os.listdir(folder_path)):
        if clip_sub.startswith('.'):
            continue
        clip_path = os.path.join(folder_path, clip_sub)
        if not os.path.isdir(clip_path):
            continue
        
        frames = load_frames(clip_path)
        if len(frames) == 0:
            print(f"No frames found in {clip_path}")
            continue

        windows = create_windows(frames, T)
        for window in windows:
            all_cnn_inputs.append(window.transpose(3,0,1,2))
            all_drift_features.append(compute_drift_features(window))
            all_labels.append(label)
        
        clip_count += 1
        if clip_count % 5 == 0:
            print(f"Processed {clip_count} clips in {subfolder_name}...")
            # Save intermediate results
            np.save(os.path.join(outputs_folder, f"cnn_inputs_{subfolder_name}.npy"), np.array(all_cnn_inputs, dtype=np.float32))
            np.save(os.path.join(outputs_folder, f"drift_features_{subfolder_name}.npy"), np.array(all_drift_features, dtype=np.float32))
            np.save(os.path.join(outputs_folder, f"labels_{subfolder_name}.npy"), np.array(all_labels, dtype=np.float32))

    # Final save for remaining clips
    np.save(os.path.join(outputs_folder, f"cnn_inputs_{subfolder_name}.npy"), np.array(all_cnn_inputs, dtype=np.float32))
    np.save(os.path.join(outputs_folder, f"drift_features_{subfolder_name}.npy"), np.array(all_drift_features, dtype=np.float32))
    np.save(os.path.join(outputs_folder, f"labels_{subfolder_name}.npy"), np.array(all_labels, dtype=np.float32))
    print(f"Finished processing {subfolder_name}, total clips: {clip_count}")
