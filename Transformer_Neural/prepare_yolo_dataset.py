"""
Prepare YOLOv8 Dataset for TCS Substation Asset Inspection.
Labels all 6 images with exact bounding boxes for:
- Asset Tags: tag_TR1, tag_TR2, tag_CB1, tag_CS02
- Components: transformer_body, oil_gauge, bushing, circuit_breaker, cable_bundle
- Defects: oil_leakage, crack, cable_damage, burn_mark, loose_connection, physical_damage
Also creates augmented images (brightness, contrast, slight blur) to make training robust.
"""

import os
import shutil
import cv2
import numpy as np

# Class definitions
CLASSES = [
    "tag_TR1",             # 0: Asset ID TR-1
    "tag_TR2",             # 1: Asset ID TR-2
    "tag_CB1",             # 2: Asset ID CB-1
    "tag_CS02",            # 3: Asset ID CS-02
    "transformer_body",    # 4: Transformer main tank/radiator
    "oil_gauge",           # 5: Transformer oil level indicator
    "bushing",             # 6: High voltage bushing
    "circuit_breaker",     # 7: Circuit breaker mechanism & housing
    "cable_bundle",        # 8: Cable runs / trays
    "oil_leakage",         # 9: Oil leakage on/around transformer
    "crack",               # 10: Structural cracks / fissures
    "insulation_damage",   # 11: Cable insulation damage / cuts (red boxes)
    "burn_mark",           # 12: Heat scorch or arc burn marks / cable damage (yellow boxes)
    "loose_connection",    # 13: Loose or disconnected terminals/cables
    "physical_damage",     # 14: Mechanical damage or dents
]

CLASS_TO_IDX = {name: i for i, name in enumerate(CLASSES)}

# Precise ground truth bounding boxes in [xmin, ymin, xmax, ymax, class_name]
GROUND_TRUTH = {
    "1.1 Transformer-1 (Healthy).png": [
        (220, 360, 350, 445, "tag_TR1"),
        (60, 260, 720, 820, "transformer_body"),
        (435, 345, 515, 435, "oil_gauge"),
        (220, 50, 680, 270, "bushing"),
    ],
    "1.2 Transformer-1 (Faulty).png": [
        (195, 370, 320, 460, "tag_TR1"),
        (60, 250, 720, 820, "transformer_body"),
        (385, 345, 460, 430, "oil_gauge"),
        (220, 50, 680, 260, "bushing"),
        (390, 245, 570, 830, "oil_leakage"),
        (30, 790, 750, 905, "oil_leakage"),
        (380, 520, 530, 690, "crack"),
    ],
    "1.3 Transformer-2 (Healthy).png": [
        (550, 460, 695, 550, "tag_TR2"),
        (380, 300, 1250, 950, "transformer_body"),
        (855, 455, 935, 545, "oil_gauge"),
        (520, 80, 1120, 320, "bushing"),
    ],
    "2.1 Circuit Breaker (Healthy).png": [
        (655, 410, 750, 505, "tag_CB1"),
        (480, 360, 1080, 920, "circuit_breaker"),
        (530, 110, 1040, 380, "bushing"),
    ],
    "3.1 Cable Section (Healthy).png": [
        (240, 755, 485, 865, "tag_CS02"),
        (50, 50, 720, 760, "cable_bundle"),
    ],
    "3.2 Cable Section (Faulty).png": [
        (205, 770, 470, 885, "tag_CS02"),
        (50, 50, 720, 770, "cable_bundle"),
        # Yellow boxes: burn marks / cable damage
        (116, 223, 296, 410, "burn_mark"),          # Upper left cable burn / damage
        (326, 130, 391, 417, "burn_mark"),          # Upper center cable burn / damage
        (424, 119, 538, 306, "burn_mark"),          # Upper right cable burn / damage
        # Red boxes: insulation damage
        (169, 422, 328, 656, "insulation_damage"),  # Lower left cable insulation damage
        (349, 422, 487, 660, "insulation_damage"),  # Lower center cable insulation damage
        (493, 315, 631, 603, "insulation_damage"),  # Lower right cable insulation damage
    ],
}


def to_yolo_format(box, img_w, img_h):
    """Convert [xmin, ymin, xmax, ymax] to [x_center, y_center, w, h] normalized."""
    x1, y1, x2, y2, cls_name = box
    cls_idx = CLASS_TO_IDX[cls_name]
    x_center = ((x1 + x2) / 2.0) / img_w
    y_center = ((y1 + y2) / 2.0) / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return cls_idx, x_center, y_center, w, h


def create_dataset():
    """Create directory structure and save images + annotations."""
    base_dir = os.path.abspath("yolo_dataset")
    train_img_dir = os.path.join(base_dir, "images", "train")
    val_img_dir = os.path.join(base_dir, "images", "val")
    train_lbl_dir = os.path.join(base_dir, "labels", "train")
    val_lbl_dir = os.path.join(base_dir, "labels", "val")

    # Clean existing
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)

    for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
        os.makedirs(d, exist_ok=True)

    print(f"[DATASET] Creating YOLO dataset at {base_dir}...")

    total_samples = 0
    for img_name, boxes in GROUND_TRUTH.items():
        img_path = img_name if os.path.exists(img_name) else os.path.join("img", img_name)
        if not os.path.exists(img_path):
            print(f"[WARN] Image {img_name} not found at {img_path}!")
            continue

        img = cv2.imread(img_path)
        h, w = img.shape[:2]

        # Convert boxes
        yolo_lines = []
        for box in boxes:
            c_idx, xc, yc, bw, bh = to_yolo_format(box, w, h)
            yolo_lines.append(f"{c_idx} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

        clean_stem = os.path.splitext(os.path.basename(img_name))[0].replace(" ", "_").replace("(", "").replace(")", "").replace(".", "_")

        # 1. Base image (both train and val for evaluation)
        for split in ["train", "val"]:
            dst_img = os.path.join(base_dir, "images", split, f"{clean_stem}.jpg")
            dst_lbl = os.path.join(base_dir, "labels", split, f"{clean_stem}.txt")
            cv2.imwrite(dst_img, img)
            with open(dst_lbl, "w") as f:
                f.write("\n".join(yolo_lines) + "\n")

        total_samples += 1

        # 2. Augmentations for train set:
        # Aug A: Slightly brighter (robot light)
        img_bright = cv2.convertScaleAbs(img, alpha=1.15, beta=15)
        dst_img = os.path.join(train_img_dir, f"{clean_stem}_bright.jpg")
        dst_lbl = os.path.join(train_lbl_dir, f"{clean_stem}_bright.txt")
        cv2.imwrite(dst_img, img_bright)
        with open(dst_lbl, "w") as f:
            f.write("\n".join(yolo_lines) + "\n")

        # Aug B: Slightly darker (dim substation lighting)
        img_dark = cv2.convertScaleAbs(img, alpha=0.85, beta=-15)
        dst_img = os.path.join(train_img_dir, f"{clean_stem}_dark.jpg")
        dst_lbl = os.path.join(train_lbl_dir, f"{clean_stem}_dark.txt")
        cv2.imwrite(dst_img, img_dark)
        with open(dst_lbl, "w") as f:
            f.write("\n".join(yolo_lines) + "\n")

        # Aug C: High contrast
        img_contrast = cv2.convertScaleAbs(img, alpha=1.25, beta=0)
        dst_img = os.path.join(train_img_dir, f"{clean_stem}_contrast.jpg")
        dst_lbl = os.path.join(train_lbl_dir, f"{clean_stem}_contrast.txt")
        cv2.imwrite(dst_img, img_contrast)
        with open(dst_lbl, "w") as f:
            f.write("\n".join(yolo_lines) + "\n")

        # Aug D: Slight motion blur (moving robot cam)
        k = np.zeros((5, 5))
        k[2, :] = 1.0 / 5.0
        img_blur = cv2.filter2D(img, -1, k)
        dst_img = os.path.join(train_img_dir, f"{clean_stem}_blur.jpg")
        dst_lbl = os.path.join(train_lbl_dir, f"{clean_stem}_blur.txt")
        cv2.imwrite(dst_img, img_blur)
        with open(dst_lbl, "w") as f:
            f.write("\n".join(yolo_lines) + "\n")

        total_samples += 4

    # Create dataset.yaml
    yaml_path = os.path.join(base_dir, "dataset.yaml")
    clean_base_dir = base_dir.replace('\\', '/')
    yaml_content = f"""path: {clean_base_dir}
train: images/train
val: images/val

names:
"""
    for i, name in enumerate(CLASSES):
        yaml_content += f"  {i}: {name}\n"

    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print(f"[DATASET] Created {total_samples} training samples.")
    print(f"[DATASET] YAML config saved at: {yaml_path}")
    return yaml_path


if __name__ == "__main__":
    create_dataset()
