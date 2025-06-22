import torch, torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from app.utils import create_dataframe, sample_dataframe, FaceDataset
from app.cnn import AttentionVGG as CNNClassifier
from app.model import FaceDetector
from PIL import Image
from tqdm import tqdm
import os
import random, numpy as np
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
def train_one_epoch(model, loader, optimizer, criterion, device, scheduler):
    model.train()
    running_loss = 0.0
    for imgs, labs in tqdm(loader, desc="  Training"):
        imgs, labs = imgs.to(device), labs.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, labs)
        loss.backward()
        optimizer.step()
        scheduler.step()
        running_loss += loss.item()
    return running_loss / len(loader)

def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labs in tqdm(loader, desc="  Validating"):
            imgs, labs = imgs.to(device), labs.to(device)
            logits = model(imgs)
            loss   = criterion(logits, labs)
            running_loss += loss.item()

            preds = logits.argmax(1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labs.tolist())

    avg_loss = running_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds) * 100
    prec = precision_score(all_labels, all_preds)
    rec  = recall_score(all_labels, all_preds)
    f1   = f1_score(all_labels, all_preds)
    print(f"Val — acc {acc:.1f}%  prec {prec:.2f}  rec {rec:.2f}  F1 {f1:.2f}")
    return avg_loss, acc
def main():
    
    train_dir = os.path.join("dataset", "train")
    val_dir   = os.path.join("dataset", "validate")

    full_train_df = create_dataframe(train_dir)
    full_val_df   = create_dataframe(val_dir)

    train_df = sample_dataframe(full_train_df, n_per_class=1000, seed=42)
    val_df   = sample_dataframe(full_val_df,   n_per_class=400, seed=42)

    # loaders
    train_loader = DataLoader(FaceDataset(train_df, train=True), batch_size=32, shuffle=True)
    val_loader   = DataLoader(FaceDataset(val_df,   train=False), batch_size=32)

    # model, loss, optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = CNNClassifier().to(device)
    for p in model.block3.parameters():
        p.requires_grad = False
    for p in model.block4.parameters():
        p.requires_grad = False
    for p in model.block5.parameters():
        p.requires_grad = False
    total_epochs = 10
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam([
        {"params": model.block3.parameters(),   "lr": 1e-5},
        {"params": model.block4.parameters(),   "lr": 1e-5},
        {"params": model.block5.parameters(),   "lr": 1e-5},
        {"params": model.classifier.parameters(), "lr": 1e-3},
    ], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[1e-5, 1e-5, 1e-5, 1e-3],
        steps_per_epoch=len(train_loader),
        epochs=total_epochs
    )
    best_acc, wait, patience = 0.0, 0, 3
    for epoch in range(total_epochs):
        print(f"Epoch {epoch+1}")
        if epoch == 3:
            for p in model.block3.parameters():
                p.requires_grad = True
            for p in model.block4.parameters():
                p.requires_grad = True
            for p in model.block5.parameters():
                p.requires_grad = True
        tr_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scheduler)
        val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, device)
        if val_acc > best_acc:
            best_acc, wait = val_acc, 0
            torch.save(model.state_dict(), "models/face_detector.pth")
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    #for epoch in range(3, 10):
    #    tr_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
    #    val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, device)
    #    scheduler.step()
    #    print(f"Epoch {epoch+1}") 
              #— train loss {tr_loss:.4f}, val loss {val_loss:.4f}, val acc {val_acc:.1f}%")
    #    if val_acc > best_acc:
    #        best_acc, wait = val_acc, 0
    #        torch.save(model.state_dict(), "models/face_detector.pth")
    #    else:
    #        wait += 1
    #        if wait >= patience:
    #            print(f"Early stopping at epoch {epoch+1}")
    #            break
    #detector = FaceDetector(weights_path="models/face_detector.pth")
    #img = Image.open("datasetsmall/validate/0/00005.jpg").convert("RGB")
    #print("Real-face prob:", detector.predict(img))
if __name__ == "__main__":
    main()