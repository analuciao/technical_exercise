"""Build (frame, mask) training pairs from the expert tracings in VolumeTracings.csv."""
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.draw import polygon
 
from config import (DATA_DIR, FILELIST_PATH, LIMITS, OUTPUT_DIR, PLOT_CHECK,
                    TRACINGS_PATH, VIDEO_DIR)
 
 
def load_frame(video_path, frame_idx):
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"Could not read frame {frame_idx} from {video_path.name}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 
 
def tracing_to_mask(rows, h, w):
    rows = rows.iloc[1:]                                   # drop long-axis line
    xs = np.concatenate([rows.X1[::-1], rows.X2])
    ys = np.concatenate([rows.Y1[::-1], rows.Y2])
    rr, cc = polygon(ys, xs, shape=(h, w))
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[rr, cc] = 1
    return mask
 
 
def build_split(split, files, tracings, limit=None):
    """Return frames, masks and ids for every traced frame in one split."""
    names = [n.removesuffix(".avi") for n in files.loc[files["Split"] == split, "FileName"]][:limit]
 
    frames, masks, ids = [], [], []
    for name in names:
        video_rows = tracings[tracings["Stem"] == name]
        if video_rows.empty:
            print(f"[{split}] no tracings for {name}, skipping")
            continue
        for frame_idx, rows in video_rows.groupby("Frame"):
            try:
                frame = load_frame(VIDEO_DIR / f"{name}.avi", frame_idx)
            except ValueError as e:
                print(f"[{split}] {e}, skipping")
                continue
            frames.append(frame)
            masks.append(tracing_to_mask(rows, *frame.shape))
            ids.append(f"{name}_{frame_idx}")
 
    return np.stack(frames), np.stack(masks), np.array(ids)
 
 
def save_split(split, frames, masks, ids):
    prefix = split.lower()
    np.save(DATA_DIR / f"{prefix}_frames.npy", frames)
    np.save(DATA_DIR / f"{prefix}_masks.npy", masks)
    np.save(DATA_DIR / f"{prefix}_ids.npy", ids) 
 
def plot_mask_check(frames, masks, ids, out_path, n=60):
    """Overlay the first n expert masks on their frames and save to a grid image."""
    n = min(n, len(frames))
    cols = 6
    rows = -(-n // cols)                                   # round up
    fig, axes = plt.subplots(rows, cols, figsize=(12, 2 * rows))
    axes = axes.ravel()
    for i in range(n):
        axes[i].imshow(frames[i], cmap="gray")
        axes[i].contour(masks[i], levels=[0.5], colors="red", linewidths=1)
        axes[i].set_title(ids[i][:14], fontsize=7)
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"Saved mask check to {out_path}")
 
 
def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
 
    files = pd.read_csv(FILELIST_PATH)
    tracings = pd.read_csv(TRACINGS_PATH)
    tracings["Stem"] = tracings["FileName"].str.removesuffix(".avi")
 
    for split, limit in LIMITS.items():
        frames, masks, ids = build_split(split, files, tracings, limit)
        save_split(split, frames, masks, ids)
        if PLOT_CHECK and split == "TRAIN":
            plot_mask_check(frames, masks, ids, OUTPUT_DIR / "mask_check.png")
 
 
if __name__ == "__main__":
    main()