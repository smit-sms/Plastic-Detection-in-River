import logging
import datetime
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO)

# Load the 'yolov8m' model from YOLO
model = YOLO('yolov8m.pt')

logging.info(f"Training started at {datetime.datetime.now()}")

try:
    # Train the model
    model.train(
        data='plastic.yaml',
        epochs=20,
        imgsz=(1280, 720),  # w,h
        batch=5,
        optimizer="Adam",
        lr0=1e-3,
    )
    logging.info(f"Training completed at {datetime.datetime.now()}")
except (KeyboardInterrupt, Exception) as e:
    logging.exception(f"Got an exception as {e}")
