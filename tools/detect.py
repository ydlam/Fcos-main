# -*- coding: utf-8 -*-
"""
# @file name  : inference.py
# @author     : chenzhanpeng https://github.com/chenzpstar
# @date       : 2022-01-18
# @brief      : FCOS推理
"""

import argparse
import os
import sys
import time

from tensorboardX import SummaryWriter
from torchvision import transforms

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, ".."))

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from data.bdd100k import BDD100KDataset
from data.kitti import KITTIDataset
from data.transform import Normalize, Resize
from models.fcos import FCOSDetector
import torch
import torchvision.models
import hiddenlayer as hl

# 添加解析参数
parser = argparse.ArgumentParser(description="Inference")
parser.add_argument("--data_folder",
                    default="bdd100k",
                    type=str,
                    help="dataset folder name")
parser.add_argument("--ckpt_folder",
                    default=None,
                    type=str,
                    help="checkpoint folder name")
args = parser.parse_args()

if __name__ == "__main__":
    writer = SummaryWriter(log_dir=None,comment='',filename_suffix='')

    # 0. config
    cmap = plt.get_cmap("rainbow")
    colors = [cmap(i) for i in np.linspace(0, 1, 10)]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_dir = os.path.join(BASE_DIR, "..", "..", "data", "bdd100k")
    assert os.path.exists(data_dir)

    # 1. dataset
    img_dir = os.path.join(data_dir, "images", "100k", "test")

    # 2. model
    model = FCOSDetector(mode="inference")
    checkpoint = torch.load("./results/checkpoint_5.pth",
                            map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    # writer.add_graph(model, input_to_model=None, verbose=None)  # 模型及模型输入数据
    # writer.close()


    print("INFO ==> finish loading model")

    root = "./test_images/"
    names = os.listdir(root)
    for name in names:
        img_bgr = cv2.imread(root + name)
        img = cv2.cvtColor(img_bgr.copy(), cv2.COLOR_BGR2RGB)
        img1 = transforms.ToTensor()(img)
        img1 = transforms.Normalize([0.5, 0.5, 0.5], [1., 1., 1.], inplace=True)(img1)


        start_t = time.time()
        with torch.no_grad():
            out = model(img1.unsqueeze_(dim=0))
        end_t = time.time()
        cost_t = 1000 * (end_t - start_t)
        print("===>success processing img, cost time %.2f ms" % cost_t)
        scores, classes, boxes = out

        boxes = boxes[0].cpu().numpy().tolist()
        classes = classes[0].cpu().numpy().tolist()
        scores = scores[0].cpu().numpy().tolist()

        for i, box in enumerate(boxes):
            pt1 = (int(box[0]), int(box[1]))
            pt2 = (int(box[2]), int(box[3]))
            cv2.rectangle(img_bgr, pt1, pt2, (0, 255, 0))
            cv2.putText(img_bgr, "%s %.3f" % (BDD100KDataset.cls_names[int(classes[i])], scores[i]),
                        (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255))
            print(BDD100KDataset.cls_names[int(classes[i])], scores[i])

        # cv2.imshow('img', img_bgr)
        # cv2.waitKey(0)
        cv2.imwrite("./result_images/"+name,img_bgr)