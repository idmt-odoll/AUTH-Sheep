import os
import time
from ultralytics import YOLO

# predict on images and save results in yolo format

# declare paths
path_images = "/path/to/images"
path_to_models = "models"
path_to_save_to = "predictions"
if not os.path.exists(path_to_save_to):
    os.mkdir(path_to_save_to)

# for each trained model in path
for scan in os.scandir(path_to_models):
    weights_path = scan.path
    name = scan.name.split(".")[0]

    # Load the YOLO model
    yolo_model = YOLO(weights_path)
    results = yolo_model(source=path_images, save_txt=True, save_conf=True, name=name+"_SheepCounter", project=path_to_save_to,
                         iou=0.6, imgsz=832)

    # prevent the program from running too fast
    time.sleep(10)
