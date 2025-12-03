import torch
from PIL import Image
from torchvision import transforms

from .model import create_model, get_device


# Same transform as eval_transform in dataset.py
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

CLASSES = ["cockpit", "engine", "tail", "wheels", "wing-tip"]


def predict(image_path: str):
    device = get_device()

    # 1. Recreate and load the model
    model = create_model()
    state = torch.load("models/resnet18_aircraft_v1.pt", map_location=device)
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()

    # 2. Load and preprocess the image
    img = Image.open(image_path).convert("RGB")

    # 3. Transform the image
    img_t = transform(img).unsqueeze(0).to(device)

    # 4. Prediction
    with torch.no_grad():
        outputs = model(img_t)
        pred_idx = outputs.argmax(dim=1).item()
        prob = torch.softmax(outputs, dim=1)[0][pred_idx].item()

    print(f"Classe prédite : {CLASSES[pred_idx]} ({prob*100:.2f}%)")


if __name__ == "__main__":
    import sys
    predict(sys.argv[1])
