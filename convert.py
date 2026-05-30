from ultralytics import YOLO

# Load your healthy model
model = YOLO("yolov8n-face.pt") 

# Export with the specific inference size
# Note: imgsz must be a multiple of 32 (320 is valid)
model.export(format="ncnn", imgsz=320)
