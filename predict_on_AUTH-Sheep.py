import os
import cv2
from ultralytics import YOLO

# predict on video frames and save results in yolo format

# declare paths
path_test_videos = "/path/to/test/videos/"
path_to_models = "models"
path_to_save_to = "predictions"
if not os.path.exists(path_to_save_to):
    os.mkdir(path_to_save_to)


def frame_generator(video_path):
    """
    Generator to get each frame from a video.

    Parameters:
    video_path (str): Path to the video file.
    """
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    while success:
        yield image
        success, image = vidcap.read()


# for each trained model in path
for scan in os.scandir(path_to_models):
    weights_path = scan.path
    name = scan.name.split(".")[0]

    # Load the YOLO model
    yolo_model = YOLO(weights_path)

    # predict on video frames and save text files
    results_all = yolo_model.predict(source=path_test_videos, save_txt=True, save_conf=True,
                                     project=path_to_save_to, name=name+"_AUTH-Sheep",
                                     iou=0.6, imgsz=832)
