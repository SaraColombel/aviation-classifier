from .dataset import get_dataloaders

if __name__ == "__main__":
    try:
        train_loader, val_loader, test_loader = get_dataloaders()

        print("✅ Dataloaders créés avec succès.")
        print("Taille du dataset d'entraînement :", len(train_loader.dataset))
        print("Taille du dataset de validation  :", len(val_loader.dataset))
        print("Taille du dataset de test        :", len(test_loader.dataset))
        print("Classes :", train_loader.dataset.classes)
    except Exception as e:
        print("❌ Erreur lors de la création des dataloaders :")
        print(e)
