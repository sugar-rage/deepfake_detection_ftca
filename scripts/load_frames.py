# scripts/load_frames.py
import os
import cv2
import numpy as np

T = 8  # number of consecutive frames per window
frame_size = (112, 112)  # resize frames

def load_frames(folder):
    """
    Load all images from a folder (no extra subfolder loop needed).
    Returns a list of resized frames as numpy arrays.
    """
    frames = []
    for fname in sorted(os.listdir(folder)):
        if fname.lower().endswith((".jpg", ".png", ".jpeg")):
            img_path = os.path.join(folder, fname)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, frame_size)
                frames.append(img)
    return frames

def create_windows(frames, T):
    """
    Convert a list of frames into sliding windows of length T.
    Returns a list of numpy arrays of shape (T,H,W,C)
    """
    windows = []
    for i in range(len(frames) - T + 1):
        window = frames[i:i+T]
        windows.append(np.stack(window, axis=0))  # (T,H,W,C)
    return windows

if __name__ == "__main__":
    # Test loading a single clip
    folder_path = r"/content/drive/MyDrive/deepfake_ftca/dataset/real/018_15_0_3_frames"
    frames = load_frames(folder_path)
    windows = create_windows(frames, T)
    print(f"Loaded {len(frames)} frames, created {len(windows)} windows")
