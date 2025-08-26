import json
import os

# === CONFIG ===
json_path = "D:/Autoflow/dataset/fcb/val.json"
images_dir = "D:/Autoflow/dataset/fcb/val/images"
labels_dir = "D:/Autoflow/dataset/fcb/val/labels"    # directory to save YOLO labels

# Create labels directory if it doesn't exist
os.makedirs(labels_dir, exist_ok=True)

# Load the COCO-style JSON
with open(json_path) as f:
    data = json.load(f)

# Build a map from image_id to file_name, width, height
image_map = {}
for img in data['images']:
    image_map[img['id']] = {
        'file_name': img['file_name'],
        'width': img['width'],
        'height': img['height']
    }

# Write YOLO-format labels
for ann in data['annotations']:
    image_id = ann['image_id']
    category_id = ann['category_id']
    bbox = ann['bbox']  # [x_min, y_min, width, height]

    image_info = image_map[image_id]
    width = image_info['width']
    height = image_info['height']

    # Convert to YOLO format: normalize and convert to center coordinates
    x_min, y_min, box_width, box_height = bbox
    x_center = x_min + box_width / 2
    y_center = y_min + box_height / 2

    # Normalize
    x_center /= width
    y_center /= height
    box_width /= width
    box_height /= height

    # Build label line
    yolo_line = f"{category_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n"

    # File path for the label
    image_filename = os.path.splitext(image_info['file_name'])[0]
    label_path = os.path.join(labels_dir, f"{image_filename}.txt")

    # Append annotation to label file
    with open(label_path, "a") as f:
        f.write(yolo_line)

print(f"✅ YOLO annotations written to: {labels_dir}")
