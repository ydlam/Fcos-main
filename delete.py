import json
import os
import random
import shutil

import cv2
import numpy as np
import torch

def _get_data_info():
    data_info = []

    # img_dir = os.path.join("/Users/ygq/杨广奇/DeepLearning/第7期课件资料/目标检测/data/bdd100k", "images", "100k", "train")
    img_dir = os.path.join("/Users/ygq/杨广奇/DeepLearning/第7期课件资料/目标检测/data/bdd100k", "images", "100k", "val")

    for img in os.listdir(img_dir):
        if img.endswith(".jpg"):
            img_path = os.path.join(img_dir, img)
            anno_path = img_path.replace("images", "labels").replace(
                ".jpg", ".json")
            if os.path.isfile(anno_path):
                data_info.append((img_path, anno_path))

    return data_info

def _get_json_anno(img_path,json_path):
    cls_names = [
        "background", "car", "bus", "truck", "motor", "bike", "pedestrian",
        "rider", "train"
    ]
    cls_names_dict = {name: idx
                      for idx, name in enumerate(cls_names)}  # {name: idx}

    labels, boxes = [], []
    with open(json_path, 'r') as f:
        anno = json.load(f)
        # print('json_path',json_path)
        if 'labels' in anno:
            objs = anno["labels"]
            for obj in objs:
                if obj["category"] in cls_names:
                    labels.append(cls_names_dict[obj["category"]])
                    boxes.append([
                        obj["box2d"]["x1"],
                        obj["box2d"]["y1"],
                        obj["box2d"]["x2"],
                        obj["box2d"]["y2"],
                    ])

        else:
            fpath, imgname = os.path.split(img_path)
            jsonpath, jsonname = os.path.split(json_path)

            shutil.move(img_path, '/Users/ygq/杨广奇/DeepLearning/第7期课件资料/目标检测/data/bdd100k/labels/100k/delete_img_0boxes/'+imgname)
            shutil.move(json_path, '/Users/ygq/杨广奇/DeepLearning/第7期课件资料/目标检测/data/bdd100k/labels/100k/delete_json_0boxes/'+jsonname)
    if len(boxes) == 0:
        fpath, imgname = os.path.split(img_path)
        jsonpath, jsonname = os.path.split(json_path)
        shutil.move(img_path, '/Users/ygq/杨广奇/DeepLearning/第7期课件资料/目标检测/data/bdd100k/labels/100k/delete_img/' + imgname)
        shutil.move(json_path,
                    '/Users/ygq/杨广奇/DeepLearning/第7期课件资料/目标检测/data/bdd100k/labels/100k/delete_json/' + jsonname)
    return np.array(labels), np.array(boxes)

# def _get_json_anno(img_path,json_path):
#     cls_names = [
#         "background", "car", "bus", "truck", "motor", "bike", "pedestrian",
#         "rider", "train"
#     ]
#     cls_names_dict = {name: idx
#                       for idx, name in enumerate(cls_names)}  # {name: idx}
#
#     labels, boxes = [], []
#     with open(json_path, 'r') as f:
#         anno = json.load(f)
#         # print('json_path',json_path)
#         objs = anno["labels"]
#         for obj in objs:
#             if obj["category"] in cls_names:
#                 labels.append(cls_names_dict[obj["category"]])
#                 boxes.append([
#                     obj["box2d"]["x1"],
#                     obj["box2d"]["y1"],
#                     obj["box2d"]["x2"],
#                     obj["box2d"]["y2"],
#                 ])
#
#     if len(boxes) == 0:
#         fpath, imgname = os.path.split(img_path)
#         jsonpath, jsonname = os.path.split(json_path)
#         shutil.move(img_path, '/Users/ygq/杨广奇/DeepLearning/第7期课件资料/目标检测/data/bdd100k/labels/100k/delete_img/' + imgname)
#         shutil.move(json_path, '/Users/ygq/杨广奇/DeepLearning/第7期课件资料/目标检测/data/bdd100k/labels/100k/delete_json/'+jsonname)
#     return np.array(labels), np.array(boxes)
if __name__ == '__main__':
    data_info = _get_data_info()

    for i in range(len(data_info)):
        img_path, anno_path = data_info[i]
        _get_json_anno(img_path,anno_path)