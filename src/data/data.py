import pandas as pd
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from datasplit import DataSplit
from config import cfg


data_transforms = {
    "default": transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor()
    ]),
    'train': transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ]),
}


class FaceToBMIDataset(Dataset):
    def __init__(self, csv_file, image_dir):
        self.annotation = pd.read_csv(csv_file, index_col=False)
        self.image_dir = image_dir
        self.set_transform()

    def __len__(self):
        return len(self.annotation)

    def set_transform(self, type=None):
        if type == "train":
            self.transform = data_transforms['train']
        elif type == "val":
            self.transform = data_transforms['val']
        elif type == "test":
            self.transform = data_transforms['test']
        else:
            self.transform = data_transforms['default']

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_path = self.annotation.iloc[idx, 0]
        image = Image.open(img_path)
        image = self.transform(image)
        height = torch.from_numpy(
            self.annotation.iloc[idx, 1].reshape(-1, 1).squeeze(axis=1)).float()
        weight = torch.from_numpy(
            self.annotation.iloc[idx, 2].reshape(-1, 1).squeeze(axis=1)).float()
        bmi = torch.from_numpy(
            self.annotation.iloc[idx, 3].reshape(-1, 1).squeeze(axis=1)).float()
        return image, height, weight, bmi


def train_val_test_split(type="full"):
    # dataset = torch.load(cfg.total_data_processed_file)
    if type == "full":
        dataset = FaceToBMIDataset(
            csv_file=cfg.full_annotation_file, image_dir=cfg.image_path)
    elif type == "female":
        dataset = FaceToBMIDataset(
            csv_file=cfg.female_annotation_file, image_dir=cfg.image_path)
    elif type == "male":
        dataset = FaceToBMIDataset(
            csv_file=cfg.male_annotation_file, image_dir=cfg.image_path)
    elif type == "test":
        dataset = FaceToBMIDataset(
            csv_file=cfg.test_data_annotation_file, image_dir=cfg.cropped_data_path)

    if type == "test":
        split = DataSplit(dataset=dataset, test_train_split=0)
    else:
        split = DataSplit(dataset, shuffle=True)

    train_loader, valid_loader, test_loader = split.get_split(
        batch_size=cfg.batch_size)

    train_loader.dataset.set_transform("train")
    valid_loader.dataset.set_transform("val")
    test_loader.dataset.set_transform("test")
    print("Train size", len(train_loader))
    print("Validation size", len(valid_loader))
    print("Test size", len(test_loader))
    return train_loader, valid_loader, test_loader


if __name__ == "__main__":
    train_loader, valid_loader, test_loader = train_val_test_split()
    print("Train size", len(train_loader))
    print("Validation size", len(valid_loader))
    print("Test size", len(test_loader))
