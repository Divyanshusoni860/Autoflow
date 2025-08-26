import json
import os
from PIL import Image

# Define paths
splits = ['train', 'val', 'test']
classes = ['start', 'process', 'decision', 'end', 'arrow']
class_to_id = {cls_name: i for i, cls_name in enumerate(classes)}

for split in splits:
    json_path = f'dataset/fcb/{split}.json'
    images_dir = f'dataset/fcb/{split}'
    labels_dir = f'dataset/fcb/{split}_labels'

    os.makedirs(labels_dir, exist_ok=True)

    with open(json_path, 'r') as f:
        data = json.load(f)

    for item in data:
        image_name = item['imagePath']
        image_path = os.path.join(images_dir, image_name)
        label_path = os.path.join(labels_dir, image_name.replace('.jpg', '.txt').replace('.png', '.txt'))

        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            continue

        with Image.open(image_path) as img:
            img_w, img_h = img.size

        with open(label_path, 'w') as f_out:
            for shape in item['shapes']:
                label = shape['label'].lower()
                if label not in class_to_id:
                    print(f"Unknown label: {label}")
                    continue
                cls_id = class_to_id[label]

                points = shape['points']
                x1 = min(p[0] for p in points)
                y1 = min(p[1] for p in points)
                x2 = max(p[0] for p in points)
                y2 = max(p[1] for p in points)

                x_center = (x1 + x2) / 2 / img_w
                y_center = (y1 + y2) / 2 / img_h
                width = (x2 - x1) / img_w
                height = (y2 - y1) / img_h

                f_out.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

    print(f"Done converting {split} annotations to YOLO format.")
