import os.path
import json
import numpy as np
import cv2

# paths
path_to_coco_annotations = "path/to/coco/annotations"
path_to_save_OBB = "path/to/save/obb/yolo/annotations"
path_to_save_HBB = "path/to/save/hbb/yolo/annotations"
subsets = ["valid", "train"]


def rotate(points, origin=(0, 0), angle=0, isdegrees=True):
    if isdegrees:
        angle = np.deg2rad(angle)
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle), np.cos(angle)]])
    o = np.atleast_2d(origin)
    points = np.atleast_2d(points)
    return np.squeeze((R @ (points.T - o.T) + o.T).T)


def save_yolo_horizontal(annotation_data, save_folder):
    for key in annotation_data.keys():
        boxes = annotation_data.get(key).get("horizontal")
        img_height = annotation_data.get(key).get("height")
        img_width = annotation_data.get(key).get("width")
        save_path = os.path.join(save_folder, key.replace(".jpg", ".txt"))

        with open(save_path, "w") as file:
            for box in boxes:
                min_x, min_y = box[0]
                max_x, max_y = box[2]
                x_center = ((min_x + max_x) / 2) / img_width
                y_center = ((min_y + max_y) / 2) / img_height
                width = (max_x - min_x) / img_width
                height = (max_y - min_y) / img_height
                file.write(f"0 {x_center} {y_center} {width} {height}\n")
            file.close()
    print("annotations saved in YOLO format")
    return


def save_yolov8_obb(annotation_data, save_folder):

    # NOTE: points must be clockwise
    # message from author on github:
    # >>>
    # In YOLOv8-OBB, the ROTATED bounding box (OBB) is indeed defined by the parameters (cx, cy, w, h, angle), where:
    #       cx, cy are the center coordinates of the bounding box.
    #       w is the width of the box, which is the length of the longer side.
    #       h is the height of the box, which refers to the shorter side.
    #       angle defines the rotation of the box around its center.
    # With respect to the angle, it should be in the range of [0, 180) degrees, as this range sufficiently defines all
    # possible rotations for a rectangle (rotating beyond 180 degrees would only retrace rotations already represented
    # within the first 180 degrees due to the symmetry of rectangles).
    #
    # The condition w >= h helps to maintain a consistent definition where w is always regarded as the longer side of
    # the bounding box. This ensures uniformity and helps to avoid ambiguity during model training and inference.
    # <<<

    for dict_key in annotation_data.keys():
        boxes = annotation_data.get(dict_key).get("rotated")
        img_height = annotation_data.get(dict_key).get("height")
        img_width = annotation_data.get(dict_key).get("width")
        save_path = os.path.join(save_folder, dict_key.replace(".jpg", ".txt"))

        with open(save_path, "w") as file:
            for box in boxes:
                box = box.astype(float)
                box[:, 0] = box[:, 0] / img_width
                box[:, 1] = box[:, 1] / img_height
                list(box.flatten().astype(str))
                #k = ", ".join(["0"] + list(box.flatten().astype(str))) + "\n"
                file.write(" ".join(["0"] + list(box.flatten().astype(str))) + "\n")
            file.close()

    print("annotations saved for YOLOv8-OBB")
    return


for subset in subsets:
    # annotation_file = os.path.join(path_to_coco_annotations, subset, "instances_default.json")
    annotation_file = f"_new_{subset}_annotations_coco.json"

    # coco bounding box: [x, y, width, height], (x, y) istop-left corner of bbox
    with open(annotation_file, "r") as file:
        data = json.load(file)

    # dict for 'id' to 'file name'
    image_names = dict()
    for image in data.get("images"):
        image_names[image.get("id")] = {"file_name": image.get("file_name"),
                                        "width": image.get("width"),
                                        "height": image.get("height")} # .replace(".jpg", "")

    # dict for collecting annotations per file
    annotations = dict()
    for key in image_names.keys():
        image_info = image_names.get(key)
        annotations[image_info.get("file_name")] = {"horizontal": list(), "rotated": list(),
                                                    "width": image_info.get("width"), "height": image_info.get("height")}

    # transform annotation format
    # YOLO format: [x_center, y_center, width, height]
    # loaded COCO format: [top left x, top left y, width, height]
    for annotation in data.get("annotations"):
        x1, y1, w, h = annotation.get("bbox")
        rotation = annotation.get("attributes").get("rotation")
        image_name = image_names.get(annotation.get("image_id")).get("file_name")

        # counter-clockwise starting top left
        # box_to_rotate = np.array([[x1, y1], [x1, y1+h], [x1+w, y1+h], [x1+w, y1]])

        # clockwise starting top right
        box_to_rotate = np.array([[x1+w, y1], [x1+w, y1+h], [x1, y1+h], [x1, y1]])

        rotated_box = rotate(box_to_rotate,
                             np.array([x1 + w/2, y1 + h/2]), rotation).astype(int)
        annotations[image_name]["rotated"].append(rotated_box)
        min_x, max_x = min(rotated_box[:, 0]), max(rotated_box[:, 0])
        min_y, max_y = min(rotated_box[:, 1]), max(rotated_box[:, 1])
        annotations[image_name]["horizontal"].append(np.array([[min_x, min_y], [min_x, max_y], [max_x, max_y], [max_x, min_y]]))

    # check if paths exist
    if not os.path.exists(os.path.join(path_to_save_HBB, subset)):
        os.makedirs(os.path.join(path_to_save_HBB, subset))
    if not os.path.exists(os.path.join(path_to_save_HBB, subset, "labels")):
        os.makedirs(os.path.join(path_to_save_HBB, subset, "labels"))

    if not os.path.exists(os.path.join(path_to_save_OBB, subset)):
        os.makedirs(os.path.join(path_to_save_OBB, subset))
    if not os.path.exists(os.path.join(path_to_save_OBB, subset, "labels")):
        os.makedirs(os.path.join(path_to_save_OBB, subset, "labels"))

    # create and save annotations
    save_yolo_horizontal(annotations, os.path.join(path_to_save_HBB, subset, "labels"))
    save_yolov8_obb(annotations, os.path.join(path_to_save_OBB, subset, "labels"))


