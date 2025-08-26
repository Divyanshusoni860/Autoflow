import os
from yolov5.detect import run as yolo_detect
from utils_main.parse_detections import parse_detections
from utils_main.flowchart_graph import FlowchartGraph
from utils_main.ocr import extract_texts

image_path = 'data/fc1.png'
output_dir = 'outputs/detections'
weights_path = 'D:/Autoflow/yolov5/runs/train/flowchart-yolo5/weights/best.pt'

# Run YOLOv5 detection
yolo_detect(
    weights=weights_path,
    source=image_path,
    project='outputs',
    name='detections',
    save_txt=True,
    save_conf=True,
    exist_ok=True,
    imgsz=(640, 640),

    conf_thres=0.25,
    iou_thres=0.45
)

# Parse YOLO outputs
elements = parse_detections(output_dir, image_path)

# Extract OCR text
elements = extract_texts(elements, image_path)

# Build graph and generate code
graph = FlowchartGraph(elements)
code = graph.generate_code()

# Print result
print("\nGenerated Python Code:\n")
print(code)
