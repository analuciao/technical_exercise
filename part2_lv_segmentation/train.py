"""Fine-tune a U-Net on the expert LV masks and keep the checkpoint with the best VAL Dice."""
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader, Dataset

from config import (BATCH_SIZE, DATA_DIR, ENCODER, EPOCHS, IMG_SIZE, LR, MODEL_PATH,
                    PATIENCE, THRESHOLD)

torch.manual_seed(0)       # same shuffling and initialization every run


class EchoDataset(Dataset):
    def __init__(self, frames, masks, mean, std):
        self.frames, self.masks = frames, masks
        self.mean, self.std = mean, std

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, i):
        f = cv2.resize(self.frames[i], (IMG_SIZE, IMG_SIZE))
        m = cv2.resize(self.masks[i], (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
        f = (f.astype(np.float32) / 255.0 - self.mean) / self.std
        return torch.from_numpy(f)[None], torch.from_numpy(m.astype(np.float32))[None]


def loss_fn(logits, target, dice_loss=smp.losses.DiceLoss(mode="binary"),
            bce_loss=torch.nn.BCEWithLogitsLoss()):
    return dice_loss(logits, target) + bce_loss(logits, target)


def dice_score(logits, target):
    pred = (torch.sigmoid(logits) > THRESHOLD).float()
    inter = (pred * target).sum()
    return (2 * inter / (pred.sum() + target.sum() + 1e-7)).item()


def main():
    train_frames = np.load(DATA_DIR / "train_frames.npy")
    train_masks = np.load(DATA_DIR / "train_masks.npy")
    val_frames = np.load(DATA_DIR / "val_frames.npy")
    val_masks = np.load(DATA_DIR / "val_masks.npy")

    # normalization stats from the TRAINING set only
    mean = float((train_frames / 255.0).mean())
    std = float((train_frames / 255.0).std())
    print(f"MEAN={mean:.3f}  STD={std:.3f}")

    train_loader = DataLoader(EchoDataset(train_frames, train_masks, mean, std),
                              batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(EchoDataset(val_frames, val_masks, mean, std), batch_size=BATCH_SIZE)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = smp.Unet(ENCODER, encoder_weights="imagenet", in_channels=1, classes=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_dice = 0
    epochs_without_improvement = 0
    for epoch in range(EPOCHS):
        model.train()
        losses = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()          # clear old gradients
            loss = loss_fn(model(x), y)    # forward pass + loss
            loss.backward()                # compute gradients
            optimizer.step()               # update weights
            losses.append(loss.item())

        model.eval()
        scores = []
        with torch.no_grad():              # no learning during validation
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                scores.append(dice_score(model(x), y))
        val_dice = np.mean(scores)

        saved = ""
        if val_dice > best_dice:           # keep the best version
            best_dice = val_dice
            epochs_without_improvement = 0
            torch.save({"model": model.state_dict(), "mean": mean, "std": std,
                        "encoder": ENCODER, "img_size": IMG_SIZE}, MODEL_PATH)
            saved = "  <- saved"
        else:
            epochs_without_improvement += 1
        print(f"epoch {epoch:2d}: train loss {np.mean(losses):.3f}  val Dice {val_dice:.3f}{saved}")

        if epochs_without_improvement >= PATIENCE:
            print(f"No improvement for {PATIENCE} epochs, stopping early.")
            break

    print(f"Best val Dice: {best_dice:.3f}  (checkpoint: {MODEL_PATH})")


if __name__ == "__main__":
    main()