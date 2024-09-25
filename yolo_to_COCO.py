import globox
import os
import json
import glob
from pycocotools.coco import COCO

# script for convert YOLO predictions into COCO format for evaluation

images_test = "/path/to/images"
annotations_test = "/path/to/groundtruth/coco/annotations"

dict_image_ids = dict()
with open(annotations_test, "r") as f:
    gt_test = json.load(f)
    for image in gt_test.get("images"):
        dict_image_ids[image.get("file_name")] = image.get("id")
dict_category = {"sheep": 1, "0": 1}

project_folder = "/path/to/predictions" # folder which contains the subfolders with the predictions
for scan in os.scandir(project_folder):
    folder_path = scan.path
    name = scan.name
    print(name)

    if os.path.exists(os.path.join(folder_path, "_coco_annotations.json")):
        print("      already exists")
        continue

    if "obb" not in name and "OBB" not in name:

        image_path = images_test
        id_dict = dict_image_ids

        yolo = globox.AnnotationSet.from_yolo_v5(
            folder=os.path.join(folder_path, "labels"),
            image_folder=image_path
        )

        yolo.save_coco(os.path.join(folder_path, "_coco_annotations.json"),
                       imageid_to_id=id_dict, label_to_id=dict_category)
        print(f"      finished {name}")

    else:
        dict_for_json = {'licenses': [{'name': '', 'id': 0, 'url': ''}],
                         'info': {'contributor': '', 'data_created': '', 'description': '', 'url': '', 'version': '',
                                  'year': ''},
                         'categories': [{'id': 1, 'name': 'sheep', 'supercategory': 'none'},
                                        {'id': 1, 'name': '0', 'supercategory': 'none'},],
                         'images': gt_test.get("images"),
                         'annotations': list()}

        images = gt_test.get("images")

        obb_anno_files = glob.glob(os.path.join(folder_path, "labels", "*.txt"))
        for obb_anno_file in obb_anno_files:
            image_name = os.path.basename(obb_anno_file).replace(".txt", ".jpg")

            image_width = None
            image_height = None
            for image in images:
                if image.get("file_name") == image_name:
                    image_width = image.get("width")
                    image_height = image.get("height")
                    break
            cor_fac = 1

            if image_width is None or image_height is None:
                print(f"couldn't find {image_name}")
                exit()

            image_id = dict_image_ids.get(image_name)

            with open(obb_anno_file, "r") as file:
                lines = file.readlines()
                file.close()

            for line in lines:
                id, x1, y1, x2, y2, x3, y3, x4, y4, confidence = line.split(" ")
                x1, x2, x3, x4 = float(x1), float(x2), float(x3), float(x4)
                y1, y2, y3, y4 = float(y1) * cor_fac, float(y2) * cor_fac, float(y3) * cor_fac, float(y4) * cor_fac

                x1, x2, x3, x4 = x1 * image_width, x2 * image_width, x3 * image_width, x4 * image_width
                y1, y2, y3, y4 = y1 * image_height, y2 * image_height, y3 * image_height, y4 * image_height

                dict_for_json["annotations"].append({'image_id': image_id,
                                                     'category_id': 1,
                                                     'segmentation': [[float(x1), float(y1), float(x2), float(y2),
                                                                      float(x3), float(y3), float(x4), float(y4)]],
                                                     'score': float(confidence)})

        with open(os.path.join(folder_path, "_coco_annotations.json"), "w") as file:
            json.dump(dict_for_json, file)
        print(f"      finished {name}")
