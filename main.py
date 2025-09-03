import os
import random
import shutil
import argparse
from pathlib import Path
from zipfile import ZipFile
import xml.etree.ElementTree as ET

CLASS_NAME_TO_ID = {}


def extract_classes_from_xmls(label_files):
    """从XML标注文件中提取类别名称"""
    class_names = set()
    for xml_path in label_files:
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            for obj in root.findall("object"):
                name = obj.find("name")
                if name is not None:
                    class_names.add(name.text)
        except Exception as e:
            print(f"[ERROR] Failed to parse {xml_path}: {e}")
    return sorted(class_names)


def extract_classes_from_txts(label_files, classes_file=None):
    """从YOLO格式的TXT标注文件中提取类别ID，并尝试从classes.txt获取类别名称"""
    class_ids = set()

    # 从TXT标注文件中提取所有使用的类别ID
    for txt_path in label_files:
        try:
            with open(txt_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split()
                        if (
                            len(parts) >= 5
                        ):  # YOLO格式：class_id x_center y_center width height
                            class_id = int(parts[0])
                            class_ids.add(class_id)
        except Exception as e:
            print(f"[ERROR] Failed to parse {txt_path}: {e}")

    # 尝试从classes.txt文件读取类别名称
    class_names_from_file = []
    if classes_file and classes_file.exists():
        try:
            with open(classes_file, "r", encoding="utf-8") as f:
                class_names_from_file = [line.strip() for line in f if line.strip()]
            print(
                f"[INFO] Loaded {len(class_names_from_file)} class names from {classes_file}"
            )
        except Exception as e:
            print(f"[ERROR] Failed to read classes file {classes_file}: {e}")

    # 构建最终的类别名称列表
    if class_names_from_file and class_ids:
        # 根据实际使用的类别ID获取对应的类别名称
        result_names = []
        for class_id in sorted(class_ids):
            if class_id < len(class_names_from_file):
                result_names.append(class_names_from_file[class_id])
            else:
                print(
                    f"[WARN] Class ID {class_id} exceeds available class names, using default name"
                )
                result_names.append(f"class_{class_id}")
        return result_names
    elif class_ids:
        # 没有classes.txt文件，使用类别ID作为名称
        return [f"class_{class_id}" for class_id in sorted(class_ids)]
    else:
        # 没有找到任何类别ID
        print("[WARN] No class IDs found in annotation files")
        return []


def detect_annotation_format(labels_dir):
    """检测标注文件格式：XML (VOC) 或 TXT (YOLO)"""
    label_files = list(labels_dir.iterdir())
    xml_count = len([f for f in label_files if f.suffix.lower() == ".xml"])
    txt_count = len(
        [
            f
            for f in label_files
            if f.suffix.lower() == ".txt" and f.name != "classes.txt"
        ]
    )

    if xml_count > txt_count:
        return "xml"
    elif txt_count > 0:
        return "txt"
    else:
        return "unknown"


def convert_voc_to_yolo(xml_path: Path, txt_path: Path, class_name_to_id: dict):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    if size is None:
        print(f"[ERROR] No size info in {xml_path}")
        return

    img_w = int(size.find("width").text)
    img_h = int(size.find("height").text)

    lines = []
    for obj in root.findall("object"):
        name = obj.find("name").text
        class_id = class_name_to_id.get(name)
        if class_id is None:
            print(f"[WARN] Unknown class '{name}' in {xml_path.name}, skipped.")
            continue

        bndbox = obj.find("bndbox")
        xmin = int(float(bndbox.find("xmin").text))
        ymin = int(float(bndbox.find("ymin").text))
        xmax = int(float(bndbox.find("xmax").text))
        ymax = int(float(bndbox.find("ymax").text))

        x_center = ((xmin + xmax) / 2) / img_w
        y_center = ((ymin + ymax) / 2) / img_h
        box_width = (xmax - xmin) / img_w
        box_height = (ymax - ymin) / img_h

        lines.append(
            f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"
        )

    if lines:
        txt_path.write_text("\n".join(lines))


def generate_data_yaml(output_dir: Path, class_name_to_id: dict):
    yaml_path = output_dir / "data.yaml"
    lines = [
        f"path: .",
        f"train: images/train",
        f"val: images/val",
        f"test: images/test",
        "",
        "names:",
    ]
    for class_id, name in sorted(class_name_to_id.items(), key=lambda x: x[1]):
        lines.append(f"  {class_id}: {name}")
    yaml_path.write_text("\n".join(lines))
    print(f"[INFO] Generated data.yaml at: {yaml_path}")


def split_dataset(
    base_dir: Path,
    output_dir: Path,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    convert_xml=True,
    generate_yaml=False,
):
    images_dir = base_dir / "images"
    labels_dir = base_dir / "annotations"

    output_images_dir = output_dir / "images"
    output_labels_dir = output_dir / "labels"

    image_exts = {".jpg", ".jpeg", ".png"}
    label_exts = {".txt", ".xml"}

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    for split in ["train", "val", "test"]:
        (output_images_dir / split).mkdir(parents=True, exist_ok=True)
        (output_labels_dir / split).mkdir(parents=True, exist_ok=True)

    image_files = [f for f in images_dir.iterdir() if f.suffix.lower() in image_exts]
    image_stem_to_path = {f.stem: f for f in image_files}

    label_files = [f for f in labels_dir.iterdir() if f.suffix.lower() in label_exts]
    label_stem_to_path = {f.stem: f for f in label_files}

    # 自动检测标注格式
    annotation_format = detect_annotation_format(labels_dir)
    print(f"[INFO] Detected annotation format: {annotation_format}")

    # 根据检测到的格式提取类别信息
    class_names = []
    class_name_to_id = {}

    if annotation_format == "xml":
        xml_files = [f for f in label_files if f.suffix.lower() == ".xml"]
        class_names = extract_classes_from_xmls(xml_files)
        class_name_to_id = {name: idx for idx, name in enumerate(class_names)}
        print(f"[INFO] Extracted classes from XML files: {class_name_to_id}")
    elif annotation_format == "txt":
        txt_files = [
            f
            for f in label_files
            if f.suffix.lower() == ".txt" and f.name != "classes.txt"
        ]
        classes_file = labels_dir / "classes.txt"
        class_names = extract_classes_from_txts(txt_files, classes_file)
        class_name_to_id = {name: idx for idx, name in enumerate(class_names)}
        print(f"[INFO] Extracted classes from TXT files: {class_name_to_id}")
    else:
        print(f"[WARN] Unknown annotation format or no annotation files found")

    # 保存 classes.txt
    if class_names:
        (output_dir / "classes.txt").write_text("\n".join(class_names))
        print(f"[INFO] Saved classes.txt with {len(class_names)} classes")

    # 匹配图像和标签
    matched_pairs = []
    for stem, img_path in image_stem_to_path.items():
        label_path = label_stem_to_path.get(stem)
        if label_path:
            matched_pairs.append((img_path, label_path))
        else:
            print(f"[WARN] No label file found for: {img_path.name}")

    random.shuffle(matched_pairs)
    total = len(matched_pairs)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    splits = {
        "train": matched_pairs[:train_end],
        "val": matched_pairs[train_end:val_end],
        "test": matched_pairs[val_end:],
    }

    # 复制文件并处理标注格式转换
    for split, pairs in splits.items():
        for img_path, label_path in pairs:
            shutil.copy(img_path, output_images_dir / split / img_path.name)

            if label_path.suffix.lower() == ".xml" and convert_xml:
                # XML转YOLO格式
                yolo_txt_path = output_labels_dir / split / f"{label_path.stem}.txt"
                convert_voc_to_yolo(label_path, yolo_txt_path, class_name_to_id)
            else:
                # 直接复制TXT文件或不转换的XML文件
                shutil.copy(label_path, output_labels_dir / split / label_path.name)

    # 生成data.yaml文件
    if generate_yaml and class_names:
        generate_data_yaml(output_dir, {v: k for k, v in class_name_to_id.items()})

    print(f"[INFO] Dataset splitting completed. Total files: {total}")
    for split, pairs in splits.items():
        print(f"[INFO] {split}: {len(pairs)} files")


def zip_output_dir(output_dir: Path, dist_dir: Path):
    dist_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dist_dir / f"{output_dir.name}.zip"
    with ZipFile(zip_path, "w") as zipf:
        for root, _, files in os.walk(output_dir):
            for file in files:
                file_path = Path(root) / file
                zipf.write(file_path, file_path.relative_to(output_dir.parent))
    print(f"[INFO] Output directory compressed to: {zip_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Split dataset into train/val/test, convert XML to YOLO TXT, and compress output."
    )

    parser.add_argument(
        "--base_dir",
        type=str,
        default="datasets",
        help="Input dataset directory (default: datasets). Must contain 'images/' and 'annotations/' subdirectories.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Output directory for split dataset (default: output).",
    )
    parser.add_argument(
        "--dist_dir",
        type=str,
        default="dist",
        help="Directory to store ZIP file (default: dist).",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Train split ratio (default: 0.8)",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Validation split ratio (default: 0.1)",
    )
    parser.add_argument(
        "--test_ratio", type=float, default=0.1, help="Test split ratio (default: 0.1)"
    )
    parser.add_argument(
        "--convert_xml",
        type=lambda x: x.lower() in ["true", "1", "yes"],
        default=True,
        help="Whether to convert XML annotations to YOLO TXT format. For TXT annotations, this parameter is ignored (default: True)",
    )
    parser.add_argument(
        "--generate_yaml",
        type=lambda x: x.lower() in ["true", "1", "yes"],
        default=False,
        help="Whether to generate YOLO-style data.yaml config file. Works with both XML and TXT annotations (default: False)",
    )

    args = parser.parse_args()

    split_dataset(
        base_dir=Path(args.base_dir),
        output_dir=Path(args.output_dir),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        convert_xml=args.convert_xml,
        generate_yaml=args.generate_yaml,
    )

    zip_output_dir(output_dir=Path(args.output_dir), dist_dir=Path(args.dist_dir))


if __name__ == "__main__":
    main()
