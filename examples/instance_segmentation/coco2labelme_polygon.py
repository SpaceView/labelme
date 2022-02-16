from asyncio.windows_events import NULL
import json
import numpy as np
import copy

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon

from collections import defaultdict

import os
from os import listdir
from os.path import isfile, join
import pathlib

import cv2

import math
from itertools import groupby

import base64
debug_display_base64 = False

from labelme import __version__
from labelme.logger import logger
from labelme import PY2
from labelme import QT4
from labelme import utils

#from mask_to_bbox import mask_to_bbox

import sys
#sys.path.insert(1, 'D:/py/dataset_conversion_statistics/')  
FILE = pathlib.Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = pathlib.Path(os.path.relpath(ROOT, pathlib.Path.cwd()))  # relative

PI = 3.1415926535897932
MASKCHAR = 255

#NOTE： the root dir depends on the dir where PYTHON is executed
#       e.g.  '../Rotated_DONE/',  'E:/img/Tr0805rot/rot/', etc.
#os.environ["DATA_ROOT_PATH"] = 
#os.environ["TARGET_PATH"]    = 

data_roots = [
    'E:/EsightData/JX05ANN/resize/',
    'E:/EsightData/JX05ANN/scale1/',
    'E:/EsightData/JX05ANN/scale2/'
]

target_roots = [
    'E:/EsightData/JX05ANN/resize_labelme/',
    'E:/EsightData/JX05ANN/scale1_labelme/',
    'E:/EsightData/JX05ANN/scale2_labelme/'
]

#image_root = data_roots[0] #os.path.join(data_root, 'images')
#ann_root = data_roots[0]   #os.path.join(data_root, 'ann')
#target_lable_root = target_roots[0] #os.path.join(target_root, 'lme')

path_label = 'E:/EsightData/JX05ANN/labels.txt'

#===========================================================================
def run_fast_scandir(dir, ext):    # dir: str, ext: list
    subfolders, files = [], []

    for f in os.scandir(dir):
        if f.is_dir():
            subfolders.append(f.path)
        if f.is_file():
            if os.path.splitext(f.name)[1].lower() in ext:
                files.append(f.path)

    for dir in list(subfolders):
        sf, f = run_fast_scandir(dir, ext)
        subfolders.extend(sf)
        files.extend(f)

    return subfolders, files

class LabelFileError(Exception):
    pass

def coco2labelme(image_root, ann_root, target_label_root):
    #subfolders, files = run_fast_scandir(data_root, [".bmp", ".png", ".jpg", ".jpeg"])
    #for fld in subfolders:
    #    print(fld)
    if not os.path.exists(image_root):
        print('FATAL: Your image root doesnot exit!')
        return

    if not os.path.exists(ann_root):
        print('FATAL: Your coco annotation root doesnot exit!')
        return  

    if not os.path.exists(target_label_root):
        os.makedirs(target_label_root)

    onlyfiles = [f for f in listdir(image_root) if isfile(join(image_root, f)) ]
    my_imgfiles = []
    my_imgPaths = []
    my_jsonfiles = []
    my_jsonPaths = []

    class_name_to_id = {}
    id_to_class_name = {}
    for i, line in enumerate(open(path_label).readlines()):
        class_id = i - 1  # starts with -1
        class_name = line.strip()
        if class_id == -1:
            assert class_name == "__ignore__"
            continue
        id_to_class_name[class_id] = class_name
        class_name_to_id[class_name] = class_id
        
    for i  in range(len(onlyfiles)):
        if(pathlib.Path(onlyfiles[i]).suffix=='.png') or (pathlib.Path(onlyfiles[i]).suffix=='.jpg') or (pathlib.Path(onlyfiles[i]).suffix=='.jpeg'):
            json_file = pathlib.Path(onlyfiles[i]).stem + '.json'
            label_file = pathlib.Path(onlyfiles[i]).stem + '.txt'
            annotation_file = os.path.join(ann_root, json_file)
            img_file = os.path.join(image_root, onlyfiles[i])
            if not isfile(annotation_file):
                #os.remove(img_file)
                print("--------> empty image file (no corresponding annotations): ", img_file)
                empty_images_count = empty_images_count + 1
                continue
            my_imgfiles.append(onlyfiles[i])
            my_imgPaths.append(img_file)
            my_jsonfiles.append(json_file)
            my_jsonPaths.append(annotation_file)

    TOT_IMG_NUM = len(my_imgfiles)
    for i in range(TOT_IMG_NUM):   #for i in range(2):     
        print(i, '/', TOT_IMG_NUM)

        img_filepath = my_imgPaths[i]
        if not os.path.exists(img_filepath):
            continue

        image_file = my_imgfiles[i]

        json_filepath = my_jsonPaths[i]
        if not os.path.exists(json_filepath):
            continue
        tgt_jsonpath = json_filepath.replace(ann_root, target_label_root)

        dataset = json.load(open(json_filepath, 'r'))        

        if not 'annotations' in dataset:
            continue
        if not 'images' in dataset:
            continue
        
        annInfo = dataset['annotations']
        imgInfo = dataset['images'][0]
        img_width = imgInfo['width']
        img_height = imgInfo['height']

        #==================================================================
        #(SECTION II) image resize
        #img_org = cv2.imread(img_filepath)
        #
        imageData = []
        with open(img_filepath, 'rb') as fimg:
            imageData = fimg.read()
            imageData = base64.b64encode(imageData).decode("utf-8")
            t_imageData = utils.img_b64_to_arr(imageData)
            assert(t_imageData.shape[0] == img_height)
            assert(t_imageData.shape[1] == img_width)                
            if (debug_display_base64):
                cv2.imshow("debug_display_base64",t_imageData)
                cv2.waitKey(0)
        if imageData is None:
            continue

        #==================================================================
        #(SECTION I) annotation resize
        label_shapes = []
        for an in range(len(annInfo)):
            anItem = annInfo[an]
            if not 'bbox' in anItem:
                continue
            if not 'segmentation' in anItem:
                continue
            if not 'type' in anItem:
                continue
            if not 'category_id' in anItem:
                continue

            bbox_org = anItem['bbox']
            if len(bbox_org)<4:
                continue

            # labelme accepts polygon only, thus we should have polygon
            if not ('polygon'==anItem['type']):    # -------->label type
                continue         
            
            points =  anItem['segmentation'][0]
            #xpts, ypts, rsz_points = resize_polygon((img_width, img_height), points) 
            xpts = points[0:len(points):2]
            ypts = points[1:len(points):2]
            assert(len(xpts)==len(ypts))

            x1, x2 = min(xpts), max(xpts)
            y1, y2 = min(ypts), max(ypts)
            #anItem['bbox'] = [x1, y1, x2-x1, y2-y1]

            label_shapes.append(
                dict(
                    label=id_to_class_name[anItem['category_id']],
                    points=[(x, y) for x, y in zip(xpts, ypts)],
                    group_id=None,
                    shape_type='polygon',
                    flags={},
                )
            )
        
        #with open(tgt_jsonpath, "w") as fjs:
        #    json.dump(dataset, fjs)

        if not 'otherData' in locals():
            otherData = {}
        if not 'flags' in locals():
            flags = {}
        tgt_data = dict(
            version=__version__,
            flags=flags,
            shapes=label_shapes,
            imagePath=image_file,
            imageData=imageData,
            imageHeight=img_height,
            imageWidth=img_width,
        )
        for key, value in otherData.items():
            assert key not in tgt_data
            tgt_data[key] = value

        with open(tgt_jsonpath, "w") as fjs:
            json.dump(tgt_data, fjs, ensure_ascii=False, indent=2)
        
        #print('--------------------------------------')

    print('==================================================')

if __name__ == "__main__":
    for data_root, target_root in zip(data_roots, target_roots):
        print("=====================> data_root: ", data_root, "target_root: ", target_root, " <=====================")
        coco2labelme(data_root, data_root, target_root)       
    print('Main Done!')
    
print('All done!')

