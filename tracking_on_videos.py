import glob
import os
import cv2
import yaml
import time
from tracker.Defs import tracker_dict, available_trackers
from ultralytics import YOLO
from tqdm import tqdm
import pycocotools.mask as maskUtils

# predict on videos and apply tracking
# results are saved in mots format

# Path to the test videos
path_test_videos = "/path/tp/test/videos"
sheep_videos = glob.glob(path_test_videos + "*.mp4")
path_to_models = "models"
path_to_save_to = "predictions"
if not os.path.exists(path_to_save_to):
    os.mkdir(path_to_save_to)

# load tracker confugurations and check if available
# tracker to use: [bytetrack, botsort]
tracker_to_use = "botsort"
assert tracker_to_use in available_trackers, f"Tracker '{tracker_to_use}' not available!\nAvailable trackers: {available_trackers}"

tracker_args = None
try:
    tracker_args = yaml.safe_load(open(f"tracker/configs/{tracker_to_use}.yaml", "r"))
except FileNotFoundError:
    print(f"Configuration file 'tracker/configs/{tracker_to_use}.yaml' for \n"
          f"tracker {tracker_to_use} not found! Exiting...")
    exit(1)


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

    # for each video in path_test_videos
    for video_path in sheep_videos:
        video_name = os.path.split(video_path)[-1].split(".")[0]
        print(f"Video name: {video_name}")
        subfolder_to_save_to = name + "_AUTH-Sheep_" + tracker_to_use
        save_path = os.path.join(path_to_save_to, subfolder_to_save_to, f"{video_name}.txt")

        # check if subfolder already exists
        if not os.path.join(path_to_save_to, subfolder_to_save_to):
            os.mkdir(os.path.join(path_to_save_to, subfolder_to_save_to))
        # check if video already processed in this configuration (by checking if there are already files in the folder)
        elif os.path.exists(save_path):
            print(f"Video {video_name} already processed for tracker '{tracker_to_use}'. Skipping...")
            continue

        # create generator for video frames
        video_frames = frame_generator(video_path)

        # initiate Tracker
        tracker = tracker_dict[tracker_to_use](args=tracker_args, frame_rate=30)

        #  for each frame in video predict and track
        width, height = 0, 0
        frame_nr = 0
        results_to_save = list()
        for frame in tqdm(video_frames):
            frame_nr += 1
            if frame_nr == 1:
                height, width, *_ = frame.shape

            if "OBB" in name:
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                img = frame

            results_all = yolo_model(img, imgsz=832, verbose=False, iou=0.6, conf=0.1)

            # get the results
            if results_all[0].obb is None:
                results = results_all[0].boxes
            else:
                results = results_all[0].obb

            # skip frame if no results
            if results is None:
                continue

            # apply tracking
            # returns format:
            #   HBB: [x_cen, y_cen, width, height,        track_id, conf, class, idx]   for bytetrack + botsort
            #   OBB: [x_cen, y_cen, width, height, angle, track_id, conf, class, idx]   for bytetrack + botsort
            tracked_results = tracker.update(results, img=None)
            for tracked_result in tracked_results:

                idx = int(tracked_result[-1])

                # get xyxyxyxy format of boxes to convert to rle
                segm = None
                if hasattr(results, "xywhr"):
                    segm = [results.xyxyxyxy.cpu().numpy()[idx].flatten().tolist()]
                else:
                    x, y, w, h = tracked_result[:4]
                    segm = [[x - w/2, y - h/2,   x - w/2, y + h/2,    x + w/2, y + h/2,     x + w/2, y - h/2]]

                # polygon -- a single object might consist of multiple parts
                # we merge all parts into one mask rle code
                rles = maskUtils.frPyObjects(segm, height, width)
                rle = maskUtils.merge(rles)

                # time_frame id class_id img_height img_width rle
                results_to_save.append([frame_nr, int(tracked_result[-4]), 1, height, width, rle.get("counts").decode("ascii")])

        # save results
        with open(os.path.join(path_to_save_to, subfolder_to_save_to, f"{video_name}.txt"), "w") as f:
            for res in results_to_save:
                f.write(" ".join([str(r) for r in res]) + "\n")
            f.close()

        # check if file is written
        while not f.closed:
            time.sleep(0.02)
