from PIL import Image
import torch
from .dataset import get_transforms
from .model import create_model, get_device

CLASSES = ["cockpit", "engine", "tail", "wheels", "wing-tip"]

def predict(image_path: str):
    device = get_device()

    # 1. Recreate model and load weights
    model = create_model()
    state = torch.load("models/resnet18_aircraft_v1.pt", map_location=device)
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()

    # 2. Transforms with the same size as during training
    _, eval_transform = get_transforms(image_size=224)

    # 3. Load image
    img = Image.open(image_path).convert("RGB")

    # 4. Transforms
    img_t = eval_transform(img).unsqueeze(0).to(device)

    # 5. Prediction
    with torch.no_grad():
        outputs = model(img_t)
        pred_idx = outputs.argmax(dim=1).item()
        prob = torch.softmax(outputs, dim=1)[0][pred_idx].item()

    print(f"Prediction : {CLASSES[pred_idx]} ({prob*100:.2f}%)")

if __name__ == "__main__":
    import sys
    predict(sys.argv[1])