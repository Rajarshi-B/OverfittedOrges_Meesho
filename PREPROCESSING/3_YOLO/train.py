from ultralytics import YOLO
import os
import cv2


# Load the pre-trained YOLOv8 
model = YOLO("yolov8n.pt")  

# Fine-tune the model, chnage the path for config.yaml file 
results = model.train(data="./config.yaml", epochs=100)