Part 2: Left ventricle segmentation

Expected input (set in config.py):

EchoNet-Dynamic/
    Videos/*.avi
    FileList.csv
    VolumeTracings.csv

Run in order from this folder:

bash
python expert_tracings.py   # build frame/mask pairs for TRAIN, VAL, TEST
python train.py             # train, keep the best checkpoint
python predict.py           # predict, plot, summarize

All settings (paths, number of videos per split, encoder, image size, epochs, learning rate, threshold, which split to evaluate) live in config.py.

Files
config.py:	Shared settings
expert_tracings.py:	Converts each expert tracing into a filled mask and saves frames, masks and ids as .npy per split
train.py:	Fine-tunes the U-Net, saves the checkpoint with the best VAL Dice
predict.py:	Predicts masks, saves overlays and a per-video results CSV, prints metrics

Details

Masks from tracings. In VolumeTracings.csv each traced frame has ~20 rows. The first is the long axis (apex to mitral annulus); the rest are horizontal chords from wall to wall. The outline is built by walking up one wall and down the other, then filled into a binary mask.

Data splits. The official Split column is used. TRAIN updates the weights, VAL picks the best epoch, TEST is used once for the final numbers. Both traced frames of a video always stay in the same split.

Preprocessing. Frames are resized from 112×112 to 128×128 (the U-Net needs sizes divisible by 32) and standardized with the training set's mean and standard deviation. Masks are resized with nearest-neighbour interpolation so they stay 0/1. The normalization stats are stored inside the checkpoint so prediction always uses the same preprocessing as training.

Training. Loss = Dice + binary cross-entropy. Adam, learning rate 1e-4, batch size 16, up to 30 epochs with early stopping (stop after 5 epochs without VAL Dice improvement). Fixed random seed.

Only ED and ES frames are used for training, since they are the only frames with labels. They are the two extremes of the cardiac cycle, so the model sees the full range of LV sizes.

Outputs
outputs/mask_check.png:	Expert masks overlaid on training frames (sanity check of the mask construction)
outputs/predictions_vs_expert_<split>.png:	Expert outline (red), model outline (green), expert apex (yellow)
outputs/image_results_<split>.csv:	One row per video: ED and ES frame numbers, ED apex (x, y), FAC from the model, FAC from the experts, and the dataset's EF

FAC = (ED area − ES area) / ED area × 100. ED is the traced frame with the larger expert area. FAC is based on 2D areas, so it is lower than the volume-based EF but should correlate strongly with it.

Apex is the end of the expert long-axis line nearer the top of the image (the probe sits at the apex in apical views).

