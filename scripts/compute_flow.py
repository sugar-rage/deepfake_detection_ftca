# scripts/compute_flow.py
import cv2
import numpy as np
from load_frames import load_frames, create_windows

def compute_flow(window):
    """
    window: (T,H,W,C) numpy array
    returns: list of flow magnitude arrays
    """
    flow_magnitudes = []
    for i in range(1, len(window)):
        prev = cv2.cvtColor(window[i-1], cv2.COLOR_BGR2GRAY)
        curr = cv2.cvtColor(window[i], cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev, curr, None,
                                            0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        flow_magnitudes.append(mag)
    return flow_magnitudes

if __name__ == "__main__":
    # Update to your actual dataset folder
    folder_path = r"/content/drive/MyDrive/deepfake_ftca/dataset/real/sad_woman1_0_3_frames"  # change for real frames
    frames = load_frames(folder_path)
    windows = create_windows(frames, 8)
    
    if len(windows) > 0:
        flow_mags = compute_flow(windows[0])
        print(f"Computed flow for first window, shape per frame: {flow_mags[0].shape}")
    else:
        print("No windows found in this folder!")
