import os
import cv2
import json
from pycocotools.coco import COCO
from pycococreatortools import pycococreatortools
from tqdm import tqdm
import numpy as np
import random
from collections import defaultdict
import skimage.io
import skimage.segmentation
from skimage import io
from  albumentations  import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomContrast, RandomBrightness, Flip, OneOf, Compose,RandomSizedCrop,PadIfNeeded,
    VerticalFlip,ElasticTransform,RandomBrightnessContrast,RandomGamma,CenterCrop,RandomScale,Rotate

) 
RandomScale_aug=RandomScale(p=0.5,scale_limit = 0.1)
Rotate_aug=Rotate(p=1,border_mode=cv2.BORDER_CONSTANT,value=0)
RandomBrightnessContrast_aug=RandomBrightnessContrast(p=0.7,brightness_limit=[-0.5,0.2])
    
def cal_iou_ioa(mask1,mask2):
    area1 = mask1.sum()
    area2 = mask2.sum()
    inter = ((mask1/255+mask2/255)==2).sum()
    mask_iou = inter / (area1+area2-inter)
    mask_ioa = inter / min(area1,area2)
    return mask_iou,mask_ioa

json_path = '/workspace/detectron2-ResNeSt/datasets/livecell/annotations/LIVECell_proceed/livecell_coco_train_png.json'
image_path = '/workspace/detectron2-ResNeSt/datasets/livecell/images/livecell_train_val_images_png'
save_path = '/workspace/detectron2-ResNeSt/datasets/livecell/images/instance_v1'
os.makedirs(save_path,exist_ok=True)
save_path_img = os.path.join(save_path,"image")
save_path_mask = os.path.join(save_path,"mask")
os.makedirs(save_path_img,exist_ok=True)
os.makedirs(save_path_mask,exist_ok=True)
times_dict = defaultdict(int)
api = COCO(json_path)
image_ids = api.imgs.keys()
for image_id in tqdm(image_ids):
    file_name = api.imgs[image_id]['file_name']
    img = cv2.imread(os.path.join(image_path,file_name))
    # print(os.path.join(image_path,file_name))
    annos = api.imgToAnns[image_id]
    for anno in annos:
        bbox = [int(max(anno['bbox'][0]-3,0)),int(max(anno['bbox'][1]-3,0)),int(min(anno['bbox'][2]+anno['bbox'][0]+3,img.shape[1])),int(min(anno['bbox'][3]+anno['bbox'][1]+3,img.shape[0]))]
        img_crop = img[bbox[1]:bbox[3],bbox[0]:bbox[2]]
        mask = np.zeros(img.shape)
        segmentation=[]
        for index in range(0,len(anno['segmentation'][0]),2):
            segmentation.append([anno['segmentation'][0][index],anno['segmentation'][0][index+1]])
        segmentation=np.array(segmentation)
        mask = cv2.fillPoly(mask, np.int32([segmentation]), (255, 255, 255))
        mask_crop = mask[bbox[1]:bbox[3],bbox[0]:bbox[2]] 
        file_name_crop = file_name.replace(".png","")+'_'+str(anno['id'])+'.png'
        img_crop = img_crop * (mask_crop/255)
        cv2.imwrite(os.path.join(save_path_img,file_name_crop),img_crop)
        cv2.imwrite(os.path.join(save_path_mask,file_name_crop),mask_crop)
