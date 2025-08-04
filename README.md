# Dataset Split and Annotation Converter Tool

A CLI tool to **split image datasets** into `train/val/test`, **convert VOC-style XML annotations** to YOLO format `.txt`, **generate YOLO-compatible `data.yaml`**, and optionally **compress the output** into a `.zip` file.

## Input Directory Structure

Your dataset directory (`--base_dir`) must follow this format:

```
datasets/
├── images/
│   ├── 1.jpg
│   └── 2.png
└── annotations/
    ├── 1.xml
    └── 2.txt
```

> Image files must be in `images/`, and label files must be in `annotations/`.

---

## Output Directory Structure

After splitting and processing, the output directory (`--output_dir`) will look like:

```
output/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
├── classes.txt      # if --convert_xml is True
├── data.yaml        # if --generate_yaml is True
```

> A `.zip` archive will also be created in the `--dist_dir` directory.

---

## Installation

This script requires only the Python standard library (≥3.6). No external dependencies are needed.

---

## Usage

```bash
python split_dataset.py [options]
```

### Options

| Option            | Description                                                  | Default    |
| ----------------- | ------------------------------------------------------------ | ---------- |
| `--base_dir`      | Root input directory containing `images/` and `annotations/` | `datasets` |
| `--output_dir`    | Output directory to store processed dataset                  | `output`   |
| `--dist_dir`      | Directory to save the compressed `.zip` file                 | `dist`     |
| `--train_ratio`   | Proportion of training data                                  | `0.8`      |
| `--val_ratio`     | Proportion of validation data                                | `0.1`      |
| `--test_ratio`    | Proportion of test data                                      | `0.1`      |
| `--convert_xml`   | Convert `.xml` labels to YOLO `.txt` format                  | `True`     |
| `--generate_yaml` | Generate `data.yaml` file for YOLO training                  | `False`    |

### Example

```bash
# Split dataset, convert XML, generate YAML and compress output
python split_dataset.py --base_dir datasets --output_dir output --dist_dir dist --generate_yaml true
```

```bash
# Only split, no XML conversion or YAML generation (assumes .txt annotations already exist)
python split_dataset.py --convert_xml false --generate_yaml false
```

---

## data.yaml Format (YOLO)

If `--generate_yaml true` is set, a `data.yaml` will be generated like:

```yaml
path: .  # dataset root
train: images/train
val: images/val
test: images/test

names:
  0: person
  1: cat
  2: car
```

---

## Notes

* Class names are extracted from `.xml` annotations and saved to `classes.txt`.
* If `--convert_xml false`, any `.xml` files will be ignored (not converted).
* Unmatched images or labels will be skipped with warnings.

---

## Future Work (optional ideas)

* [ ] Support nested folders for images/annotations
* [ ] Add label consistency checker
* [ ] Support image/label format conversion (e.g., `.png` to `.jpg`)
