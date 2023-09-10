from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.yaml')  # build a new model from YAML
# load a pretrained model (recommended for training)
model = YOLO('yolov8n.pt')
# build from YAML and transfer weights
model = YOLO('yolov8n.yaml').load('yolov8n.pt')

# Train the model
results = model.train(data='coco128.yaml', epochs=100, imgsz=640)
