"""Predict LV masks with the best checkpoint, compare to the expert tracings, and summarize per video."""
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch

from config import DATA_DIR, EVAL_SPLIT, FILELIST_PATH, MODEL_PATH, OUTPUT_DIR, THRESHOLD, TRACINGS_PATH

device = "mps" if torch.backends.mps.is_available() else "cpu"

def load_model():
    """Rebuild the U-Net and load the weights plus the preprocessing used in training."""
    ckpt = torch.load(MODEL_PATH, map_location=device)
    model = smp.Unet(ckpt["encoder"], encoder_weights=None, in_channels=1, classes=1)
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), ckpt


def predict_mask(model, ckpt, frame):
    h, w = frame.shape
    size = ckpt["img_size"]
    f = cv2.resize(frame, (size, size)).astype(np.float32) / 255.0
    f = (f - ckpt["mean"]) / ckpt["std"]
    x = torch.from_numpy(f)[None, None].to(device)        # (1, 1, size, size)
    with torch.no_grad():
        prob = torch.sigmoid(model(x))[0, 0].cpu().numpy()
    pred = (prob > THRESHOLD).astype(np.uint8)
    return cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)  # back to original size


def dice(pred, expert):
    inter = (pred * expert).sum()
    return 2 * inter / (pred.sum() + expert.sum() + 1e-7)


def get_apex(tracings, vid_id):
    """Expert apex = the end of the long-axis line nearer the top of the image."""
    name, frame_idx = vid_id.rsplit("_", 1)
    rows = tracings[(tracings["Stem"] == name) & (tracings["Frame"] == int(frame_idx))]
    long_axis = rows.iloc[0]                       # first row = long axis
    if long_axis.Y1 < long_axis.Y2:                # smaller Y = higher in image
        return long_axis.X1, long_axis.Y1
    return long_axis.X2, long_axis.Y2


def plot_overlays(frames, experts, preds, ids, apexes, out_path, n=24):
    n = min(n, len(frames))
    cols = 6
    rows = -(-n // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(12, 2.25 * rows))
    axes = axes.ravel()
    for i in range(n):
        ax = axes[i]
        ax.imshow(frames[i], cmap="gray")
        ax.contour(experts[i], levels=[0.5], colors="red", linewidths=1)   # expert
        ax.contour(preds[i], levels=[0.5], colors="lime", linewidths=1)    # model
        ax.plot(*apexes[i], "yo", markersize=3)                            # expert apex
        ax.set_title(ids[i][:14], fontsize=7)
    for ax in axes:
        ax.axis("off")
    plt.suptitle("Red = expert, Green = model, Yellow = apex")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"Saved {out_path}")


def summarize(ids, experts, preds, apexes):
    """One row per video: ED/ES frames, ED apex, FAC from the model's masks, and EF."""
    records = []
    for vid_id, expert, pred, apex in zip(ids, experts, preds, apexes):
        name, frame_idx = vid_id.rsplit("_", 1)
        records.append({"video": name, "frame": int(frame_idx),
                        "expert_area": int(expert.sum()), "pred_area": int(pred.sum()),
                        "apex_x": apex[0], "apex_y": apex[1]})
    areas = pd.DataFrame(records)

    summary = []
    for name, g in areas.groupby("video"):
        if len(g) != 2:
            continue                               # need exactly one ED and one ES frame
        g = g.sort_values("expert_area", ascending=False)   # larger = ED, smaller = ES
        ed, es = g.iloc[0], g.iloc[1]
        summary.append({
            "video": name,
            "ED_frame": int(ed.frame),
            "ES_frame": int(es.frame),
            "ED_apex_x": round(float(ed.apex_x), 2),
            "ED_apex_y": round(float(ed.apex_y), 2),
            "FAC_pred": (ed.pred_area - es.pred_area) / ed.pred_area * 100 if ed.pred_area > 0 else np.nan,
            "FAC_expert": (ed.expert_area - es.expert_area) / ed.expert_area * 100,
        })
    summary = pd.DataFrame(summary)

    filelist = pd.read_csv(FILELIST_PATH)
    filelist["video"] = filelist["FileName"].str.removesuffix(".avi")
    return summary.merge(filelist[["video", "EF"]], on="video", how="left")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prefix = EVAL_SPLIT.lower()
    frames = np.load(DATA_DIR / f"{prefix}_frames.npy")
    experts = np.load(DATA_DIR / f"{prefix}_masks.npy")
    ids = np.load(DATA_DIR / f"{prefix}_ids.npy")

    tracings = pd.read_csv(TRACINGS_PATH)
    tracings["Stem"] = tracings["FileName"].str.removesuffix(".avi")

    model, ckpt = load_model()
    preds = np.stack([predict_mask(model, ckpt, f) for f in frames])   # predict once, reuse below
    apexes = [get_apex(tracings, vid_id) for vid_id in ids]

    plot_overlays(frames, experts, preds, ids, apexes, OUTPUT_DIR / f"predictions_vs_expert_{prefix}.png")

    summary = summarize(ids, experts, preds, apexes)
    summary.to_csv(OUTPUT_DIR / f"image_results_{prefix}.csv", index=False)
    print(summary.round(2).head(10))

    # --- metrics ---
    dices = [dice(p, e) for p, e in zip(preds, experts)]
    fac_err = (summary.FAC_pred - summary.FAC_expert).abs()
    print(f"\n[{EVAL_SPLIT}] {len(frames)} frames, {len(summary)} videos")
    print(f"Mean Dice: {np.mean(dices):.3f}")
    print(f"FAC mean absolute error: {fac_err.mean():.1f} points")
    print(f"Empty predictions: {summary.FAC_pred.isna().sum()}")
    print(summary[["EF", "FAC_expert", "FAC_pred"]].corr().round(3))


if __name__ == "__main__":
    main()