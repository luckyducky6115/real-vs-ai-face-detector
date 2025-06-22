import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def create_dataframe(data_dir):
    data = []
    for label in ["0","1"]:
        class_dir = os.path.join(data_dir, label)
        for img_name in os.listdir(class_dir):
            data.append((os.path.join(class_dir, img_name), int(label)))
    return pd.DataFrame(data, columns=["image_path","label"])

def sample_dataframe(df: pd.DataFrame,
                     n_per_class: int,
                     seed: int = 42) -> pd.DataFrame:
    dfs = []
    for lbl in df.label.unique():
        dfs.append(df[df.label == lbl]
                   .sample(n=n_per_class, random_state=seed))
    return pd.concat(dfs).sample(frac=1, random_state=seed).reset_index(drop=True)

class FaceDataset(Dataset):
    def __init__(self, dataframe, train=True):
        self.df = dataframe
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8,1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(0.2,0.2,0.2,0.1),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]) if train else transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        path,label = self.df.iloc[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label
def inference_transform(size: int = 224):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])