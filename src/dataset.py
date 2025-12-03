from pathlib import Path

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Dossier racine où on mettra les données préparées
DATA_DIR = Path("data/processed")

def get_transforms(image_size: int = 224):
    """
    Crée deux pipelines de transformations :
    - train_transform : avec augmentation de données
    - eval_transform : sans augmentation, pour val/test
    """
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    return train_transform, eval_transform


def get_dataloaders(batch_size: int = 32, image_size: int = 224):
    """
    Crée les DataLoaders train / val / test
    à partir de la structure :
    data/processed/train/<classe>/
    data/processed/val/<classe>/
    data/processed/test/<classe>/
    """
    train_transform, eval_transform = get_transforms(image_size)

    train_dir = DATA_DIR / "train"
    val_dir = DATA_DIR / "val"
    test_dir = DATA_DIR / "test"

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=eval_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=eval_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
