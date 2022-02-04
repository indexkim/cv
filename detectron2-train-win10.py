import os
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

import csv
import torch, torchvision
import numpy as np
import cv2
import json
import itertools
from glob import glob
import xml.etree.ElementTree as elemTree

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.structures import BoxMode

print(torch.__version__)

DATASET_ROOT = 'D:\\walk\\dataset'

def get_label_list(label_path):
    
    labels = []

    with open(label_path) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            labels.append(row[0])
    return labels

label_path = os.path.join(DATASET_ROOT, 'aihub_27_classes_label.csv')
labels= get_label_list(label_path)

def get_crosswalk_dicts():
    label_path = os.path.join(DATASET_ROOT, 'aihub_27_classes_label.csv')
    image_path = os.path.join(DATASET_ROOT, 'images')

    dirs = glob(os.path.join(image_path,'*'))
    xmls = glob(os.path.join(image_path,'*/*.xml'))

    labels = get_label_list(label_path)
    dataset_dicts = []
    for path in glob('D:/walk/dataset/images', recursive=False):
        for folder in sorted(os.listdir(path)):
            for file in sorted(os.listdir(path+'/'+folder)):            
                if file.endswith('jpg'):
                    jpg_path = path+'/'+folder+'/'+file
                elif file.endswith('xml'):
                    tree = elemTree.parse(path+'/'+'/'+folder+'/'+file)
                    for image in tree.findall('./image'):
                        record = {}
                        record['file_name'] = path+'/'+folder+'/'+image.attrib['name']
                        record['height'] = int(image.attrib['height'])
                        record['width'] = int(image.attrib['width'])

                        objs = []
                        for box in image.findall('./box'):
                            obj = {
                                'bbox': [
                                        float(box.attrib['xtl']), 
                                        float(box.attrib['ytl']), 
                                        float(box.attrib['xbr']), 
                                        float(box.attrib['ybr'])],
                                    'bbox_mode': BoxMode.XYXY_ABS,
                                    'category_id': labels.index(box.attrib['label']),
                                    'iscrowd': 0
                                }
                            objs.append(obj)
                        record['annotations'] = objs
                        dataset_dicts.append(record)
    return dataset_dicts

from detectron2.data import DatasetCatalog, MetadataCatalog
DatasetCatalog.register('crosswalk/train', get_crosswalk_dicts)
MetadataCatalog.get('crosswalk/train').set(thing_classes=labels)
crosswalk_metadata = MetadataCatalog.get('crosswalk/train')

os.chdir('C:/Users/jskim/Desktop/detectron2-windows')

from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg

cfg = get_cfg()
cfg.merge_from_file('./configs/COCO-Detection/retinanet_R_50_FPN_3x.yaml')
cfg.DATASETS.TRAIN = ('crosswalk/train',)
cfg.DATASETS.TEST = ()   
cfg.DATALOADER.NUM_WORKERS = 0
cfg.SOLVER.IMS_PER_BATCH = 1
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 5000    
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 27  
cfg.MODEL.WEIGHTS = 'retinanet_r_50_fpn_3x_aihub_final.pth'

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=True)
trainer.train()