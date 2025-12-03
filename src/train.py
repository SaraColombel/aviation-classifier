import torch
from torch import nn

from .dataset import get_dataloaders
from .model import create_model, get_device

def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        # 1. Forward pass
        outputs = model(images)

        # 2. Compute loss
        loss = loss_fn(outputs, labels)

        # 3. Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = loss_fn(outputs, labels)

            total_loss += loss.item()

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    return total_loss / len(loader), accuracy


def main():
    print(">>> train.py démarré")
    device = get_device()
    print("Device used:", device)

    # 1. Get dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=16)

    # 2. Create model
    model = create_model()
    model = model.to(device)

    # 3. Define optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    # 4. Training loop
    epochs = 5
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)

        print(f"Epoch {epoch}/{epochs}:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val loss  : {val_loss:.4f}")
        print(f"  Val Acc   : {val_acc*100:.2f}%")
        print("")

    # 5. Final evaluation on test set
    test_loss, test_acc = evaluate(model, test_loader, loss_fn, device)
    print("== Final Test ==")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Acc : {test_acc*100:.2f}%")

    # 6. Save the model
    import pathlib
    model_dir = pathlib.Path("models")
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / "resnet18_aircraft_v1.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()