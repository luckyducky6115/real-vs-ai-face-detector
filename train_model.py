import torch, torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from app.utils import create_dataframe, FaceDataset
from app.cnn   import CNNClassifier
from app.model import FaceDetector
from PIL import Image
from tqdm import tqdm
import os
import random, numpy as np
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
def main():
    train_dir = os.path.join("datasetsmall", "train")
    val_dir   = os.path.join("datasetsmall", "validate")

    # then pass those into your dataframe creator
    train_df = create_dataframe(train_dir)
    val_df   = create_dataframe(val_dir)    

    # loaders
    train_loader = DataLoader(FaceDataset(train_df, train=True), batch_size=32, shuffle=True)
    val_loader   = DataLoader(FaceDataset(val_df,   train=False), batch_size=32)

    # model, loss, optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = CNNClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

    patience = 2
    wait = 0
    best_acc = 0.

    for epoch in range(10):
        model.train()
        running_loss = 0.0
        for imgs, labs in tqdm(train_loader, desc=f"Train {epoch+1}"):
            imgs, labs = imgs.to(device), labs.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss   = criterion(logits, labs)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()

        avg_train_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}: Avg Train Loss = {avg_train_loss:.4f}", end=" | ")

        # validation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, labs in tqdm(val_loader, desc=f"Val   {epoch+1}"):
                imgs = imgs.to(device)
                out  = model(imgs)
                preds = out.argmax(1).cpu().tolist()
                all_preds.extend(preds)
                all_labels.extend(labs.tolist())

        acc = accuracy_score(all_labels, all_preds)*100
        print(f"Epoch {epoch+1}: Val Acc = {acc:.2f}%")
        if acc > best_acc:
            best_acc = acc
            best_path = f"models/cnn_epoch{epoch+1}_acc{acc:.1f}.pth"
            wait = 0
            torch.save(model.state_dict(), best_path)
            print(f"→ Saved new best model: {best_path}")
        else:
            wait += 1
        if wait >= patience:
            print(f"No improvement for {patience} epochs—stopping early at epoch {epoch+1}.")
            break
    detector = FaceDetector(weights_path=best_path)
    img = Image.open("datasetsmall/validate/0/00005.jpg").convert("RGB")
    print("Real-face prob:", detector.predict(img))
if __name__ == "__main__":
    main()