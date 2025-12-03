import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


NUM_CLASSES = 5  # cockpit, engine, tail, wheels, wing-tip


def get_device() -> torch.device:
    """Retourne 'cuda' si dispo, sinon 'cpu'."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_model(num_classes: int = NUM_CLASSES) -> nn.Module:
    """
    Crée un ResNet18 pré-entraîné sur ImageNet,
    et remplace la dernière couche pour s'adapter à nos classes.
    """
    # 1) Charger un modèle pré-entraîné
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)

    # 2) Geler les couches de base (on ne les entraîne pas au début)
    for param in model.parameters():
        param.requires_grad = False

    # 3) Remplacer la couche finale (fc) par une couche pour num_classes
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model
