import torch
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.transforms.functional as FT
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from tqdm import tqdm

from loss import Loss
from model import YOLO
from dataset import VOCDataset
from utils import *

seed = 123
torch.manual_seed(seed)

# Hyperparameters etc.
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 10 # 64 in original paper
EPOCHS = 90

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("DEVICE: ", DEVICE)

import os
NUM_WORKERS = os.cpu_count() // 2
print("NUM_WORKERS: ", NUM_WORKERS)
PIN_MEMORY = True

LOAD_MODEL = True
MODEL = "data/best_model_epoch21.pt"

IMG_DIR = "data/images"
LABEL_DIR = "data/labels"
TRAIN_CSV = "data/train_2000.csv"
VALIDATE_CSV = "data/val_600.csv"

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes

        return img, bboxes
transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor(),])

def train_fn(train_loader, model: YOLO, optimizer, loss_fn: Loss):
    loop = tqdm(train_loader, leave=True)
    losses = []

    for b, (x, y) in enumerate(loop):
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        out = model(x)

        loss = loss_fn(out, y)
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_postfix(loss=loss.item())

    mean_loss = sum(losses) / len(losses)
    print(f"Mean Training Loss {mean_loss}")
    return mean_loss

def validate(loader, model, loss_fn):
    model.eval()
    loop = tqdm(loader, leave = True)
    losses = []
    with torch.no_grad():
        for b, (x, y) in enumerate(loop):
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            out = model(x)
            loss = loss_fn(out, y)
            losses.append(loss.item())
            loop.set_postfix(loss=loss.item())

    mean_loss = sum(losses) / len(losses)
    print(f"Mean Validation Loss {mean_loss}")
    model.train()
    return mean_loss 

def main():
    model = YOLO().to(DEVICE)
    if LOAD_MODEL:
        model.load_state_dict(torch.load(MODEL, weights_only=True))

    loss_fn = Loss()
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    train_dataset = VOCDataset(
        TRAIN_CSV,
        transform=transform,
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR,
    )

    validation_dataset = VOCDataset(
        VALIDATE_CSV,
        transform=transform,
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR,
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True,
    )

    validation_loader = DataLoader(
        dataset=validation_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True,
    )

    train_losses = []
    val_losses = []

    best_val_loss = float('inf')
    best_model_path = None

    for epoch in range(EPOCHS):
        for x, y in train_loader:
           x = x.to(DEVICE)
           for idx in range(8):
               bboxes = cellboxes_to_boxes(model(x))
               bboxes = non_max_suppression(bboxes[idx], iou_threshold=0.5, threshold=0.4, box_format="midpoint")
               plot_image(x[idx].permute(1,2,0).to("cpu"), bboxes)

        print(f"========== Epoch {epoch + 1}/{EPOCHS} ==========")

        train_loss = train_fn(train_loader, model, optimizer, loss_fn)
        val_loss = validate(validation_loader, model, loss_fn)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"best_model_epoch{epoch + 1}.pt")

            if best_model_path is not None and os.path.exists(best_model_path):
                os.remove(best_model_path)
            best_model_path = f"best_model_epoch{epoch + 1}.pt"

            print(f"New best model saved at epoch {epoch + 1} with val_loss = {val_loss:.4f}")

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('curve.png') 

if __name__ == "__main__":
    main()
