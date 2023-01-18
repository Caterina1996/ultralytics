from ultralytics import YOLO

# Load a model

model = YOLO("yolov8l.yaml")  # build a new model from scratch
model = YOLO("yolov8l.pt")  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data="halimeda.yaml", epochs=100, imgsz=1024)