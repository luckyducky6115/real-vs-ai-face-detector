import torch, torch.nn as nn
from torch.utils.data import DataLoader
from app.utils import create_dataframe, FaceDataset
from app.cnn import AttentionVGG
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = AttentionVGG().to(device)
    model.load_state_dict(torch.load("models/face_detector.pth", map_location=device))
    model.eval()

    test_df = create_dataframe("Human Faces Dataset")
    loader  = DataLoader(FaceDataset(test_df, train=False), batch_size=32)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    all_preds, all_labels = [], []
    loss_sum = 0.0

    with torch.no_grad():
        for imgs, labs in tqdm(loader, "Testing"):
            imgs, labs = imgs.to(device), labs.to(device)
            logits = model(imgs)
            loss_sum += criterion(logits, labs).item()
            all_preds.extend(logits.argmax(1).cpu().tolist())
            all_labels.extend(labs.cpu().tolist())

    print("Test Loss:", loss_sum/len(loader))
    print("Acc:", accuracy_score(all_labels, all_preds))
    print("Precision:", precision_score(all_labels, all_preds))
    print("Recall:", recall_score(all_labels, all_preds))
    print("F1:", f1_score(all_labels, all_preds))

if __name__=="__main__":
    main()
