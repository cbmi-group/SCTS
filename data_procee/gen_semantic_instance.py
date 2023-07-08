import os
import numpy as np
from pycocotools.coco import COCO
from skimage import io
from matplotlib import pyplot as plt
import cv2
import random
import matplotlib.pyplot as plt
import numpy as np

import pathlib
from tqdm import tqdm

import skimage.io
import skimage.segmentation

dataset_dir = r'/workspace/Swin-Transformer-Object-Detection/er_stress/images/train_8bit'
json_file = r'/workspace/Swin-Transformer-Object-Detection/er_stress/annotations/train.json'
save_path = r'/workspace/Swin-Transformer-Object-Detection/er_stress/semantic_instance'

import colorsys

def _get_colors(num_colors):
    colors=[]
    for i in np.arange(0., 360., 360. / num_colors):
        hue = i/360.
        lightness = (50 + np.random.rand() * 10)/100.
        saturation = (90 + np.random.rand() * 10)/100.
        colors.append(colorsys.hls_to_rgb(hue, lightness, saturation))
    return colors

def get_color2():
    colors = []
    for i in range(18):
        for j in range(18):
            for k in range(18):
                colors.append((i*20,j*20,k*20))
    return colors
colors= get_color2()[1:]
print(colors)

if not os.path.isdir(save_path):
    os.makedirs(save_path)
coco = COCO(json_file)
catIds = coco.getCatIds() # catIds=1 表示人这一类
print(catIds)
imgIds = coco.getImgIds(catIds=1) # 图片id，许多值
print(imgIds)

for i in tqdm(range(len(imgIds))):
    img = coco.loadImgs(imgIds[i])[0]
    I = cv2.imread(os.path.join(dataset_dir , img['file_name']))
    mask = np.zeros((I.shape))
    annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)
    coco.showAnns(anns)
    color_index = 0
    color_num = 0
    cell_id = 0
    for ann in anns:  
        segmentation=[]
        for index in range(0,len(ann['segmentation'][0]),2):
            segmentation.append([ann['segmentation'][0][index],ann['segmentation'][0][index+1]])
        segmentation=np.array(segmentation)
        # cv2.fillPoly(mask, np.int32([segmentation]), colors[color_index])
        cell_id += 1
        color = (cell_id,cell_id,cell_id)
        # print(colors[color_num])
        cv2.fillPoly(mask, np.int32([segmentation]), color)
        color_index += 1
        color_num += 1

    #ret = np.concatenate((I[:,:,0],mask),1)
    print(np.unique(mask))
    print(mask[:,:,0].shape)
    cv2.imwrite(os.path.join(save_path,img['file_name'].replace('.tif','.png')),mask[:,:,0].astype(np.uint8))
