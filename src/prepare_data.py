from pathlib import Path
import json
import shutil

# Dossiers de base
RAW_DIR = Path("data/raw/airplane_parts")
PROCESSED_DIR = Path("data/processed")

# Regroupement des classes COCO -> classes finales
CLASS_MAP = {
    "cockpit": "cockpit",
    "engine": "engine",
    "engine-back": "engine",
    "engine-front": "engine",
    "engine-side": "engine",
    "tail": "tail",
    "tail-fin": "tail",
    "wheels": "wheels",
    "wing-tip": "wing-tip",
    "airplane-parts": None,
}


def process_split(split_name: str, processed_split_name: str | None = None):
    """
    Prépare un split (train / valid / test) :
    - lit le fichier COCO JSON
    - regroupe les classes
    - copie les images dans data/processed/<split>/<classe>/
    """
    if processed_split_name is None:
        processed_split_name = split_name

    images_dir = RAW_DIR / split_name
    annotations_path = RAW_DIR / "annotations" / f"{split_name}.json"

    print(f"==> Traitement du split '{split_name}'")
    print(f"    Images : {images_dir}")
    print(f"    Annotations : {annotations_path}")

    with annotations_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # 1) id d'image -> nom de fichier
    image_id_to_filename = {
        img["id"]: img["file_name"] for img in data["images"]
    }

    # 2) id de catégorie -> nom de classe COCO
    category_id_to_name = {
        cat["id"]: cat["name"] for cat in data["categories"]
    }

    # 3) image_id -> classe finale (en appliquant CLASS_MAP)
    image_id_to_final_class: dict[int, str] = {}

    for ann in data["annotations"]:
        image_id = ann["image_id"]
        category_id = ann["category_id"]

        coco_name = category_id_to_name[category_id]
        final_name = CLASS_MAP.get(coco_name, None)

        # On ignore les classes qu'on ne garde pas
        if final_name is None:
            continue

        # Si l'image n'a pas encore de classe assignée, on met celle-ci
        if image_id not in image_id_to_final_class:
            image_id_to_final_class[image_id] = final_name

    # 4) Copie des images dans data/processed/<split>/<classe>/
    out_root = PROCESSED_DIR / processed_split_name

    num_copied = 0
    for image_id, final_class in image_id_to_final_class.items():
        filename = image_id_to_filename[image_id]
        src_path = images_dir / filename

        class_dir = out_root / final_class
        class_dir.mkdir(parents=True, exist_ok=True)

        dst_path = class_dir / filename
        shutil.copy2(src_path, dst_path)
        num_copied += 1

    print(f"    Copié {num_copied} images pour le split '{split_name}'.")


def main():
    # train -> train
    process_split("train", "train")

    # valid -> val (on renomme pour coller à notre code)
    if (RAW_DIR / "valid").exists():
        process_split("valid", "val")
    elif (RAW_DIR / "val").exists():
        process_split("val", "val")
    else:
        print("⚠️ Aucun dossier 'valid' ou 'val' trouvé pour la validation.")

    # test -> test
    if (RAW_DIR / "test").exists():
        process_split("test", "test")
    else:
        print("⚠️ Aucun dossier 'test' trouvé.")

    print("✅ Préparation terminée.")


if __name__ == "__main__":
    main()
