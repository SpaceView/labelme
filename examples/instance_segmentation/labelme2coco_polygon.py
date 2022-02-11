#!/usr/bin/env python

import argparse
import collections
import datetime
import glob
import json
import os
import os.path as osp
import sys
import uuid

import imgviz
import numpy as np
from itertools import groupby

import labelme
from pycocotools import coco

from label_file_v2 import LabelFile2

try:
    import pycocotools.mask
except ImportError:
    print("Please install pycocotools:\n\n    pip install pycocotools\n")
    sys.exit(1)



# below section added bytmc
dir_src = 'E:/EsightData/JX05ANN'
dir_target = 'E:/EsightData/JX05ANN/cofmt'
path_label = 'E:/EsightData/JX05ANN/labels.txt'

# False == polygon points; True = RLE mask
use_rle_format = False 
#use_rle_format = True

def binary_mask_to_rle(binary_mask):
    rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
        if i == 0 and value == 1:
            counts.append(0)
        counts.append(len(list(elements)))
    return rle    
"""
def binary_mask_to_rle(binary_mask):
    rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')

    last_elem = 0
    running_length = 0

    for i, elem in enumerate(binary_mask.ravel(order='F')):
        if elem == last_elem:
            pass
        else:
            counts.append(running_length)
            running_length = 0
            last_elem = elem
        running_length += 1

    counts.append(running_length)

    return rle
"""

# end added bytmc

# use shoelace formula to calculate the area of a polygon
# ref. https://iq.opengenus.org/area-of-polygon-shoelace/
# other methods ref. https://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates
def PolyArea(x,y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--input_dir", default= dir_src, help="input annotated directory")

    parser.add_argument("--output_dir", default = dir_target, help="output dataset directory")

    #parser.add_argument("--labels", default = path_label, help="labels file", required=True)
    parser.add_argument("--labels", default = path_label, help="labels file")

    #parser.add_argument("--noviz", help="no visualization", action="store_true")
    parser.add_argument("--noviz", default = True, help="no visualization", action="store_true")

    parser.add_argument("--noseperate", default = False, help="no visualization", action="store_true")

    args = parser.parse_args()

    #if osp.exists(args.output_dir):
    #    print("Output directory already exists:", args.output_dir)
    #    sys.exit(1)        
    #os.makedirs(args.output_dir)
    if not osp.exists(args.output_dir):
        os.makedirs(args.output_dir)

    out_PNG_path = osp.join(args.output_dir, "PNGImages")
    if not osp.exists(out_PNG_path):
        os.makedirs(out_PNG_path)

    out_coco_path = osp.join(args.output_dir, "ann")
    if not osp.exists(out_coco_path):
        os.makedirs(out_coco_path)

    if not args.noviz:
        out_VIZ_path = osp.join(args.output_dir, "Visualization")
        if not osp.exists(out_VIZ_path):
            os.makedirs(out_VIZ_path)

    print("creating dataset:", args.output_dir)

    now = datetime.datetime.now()

    data = dict(
        info=dict(
            description=None,
            url=None,
            version=None,
            year=now.year,
            contributor=None,
            date_created=now.strftime("%Y-%m-%d %H:%M:%S.%f"),
        ),
        licenses=[dict(url=None, id=0, name=None,)],
        images=[
            # license, url, file_name, height, width, date_captured, id
        ],
        type="instances",
        annotations=[
            # segmentation, area, iscrowd, image_id, bbox, category_id, id
        ],
        categories=[
            # supercategory, id, name
        ],
    )

    class_name_to_id = {}
    for i, line in enumerate(open(args.labels).readlines()):
        class_id = i - 1  # starts with -1
        class_name = line.strip()
        if class_id == -1:
            assert class_name == "__ignore__"
            continue
        class_name_to_id[class_name] = class_id
        data["categories"].append(
            dict(supercategory=None, id=class_id, name=class_name,)
        )

    out_ann_file = osp.join(args.output_dir, "annotations.json")
    label_files = glob.glob(osp.join(args.input_dir, "*.json"))
    for image_id, filename in enumerate(label_files):
        print("Generating dataset from:", filename)

        label_file = LabelFile2(filename=filename)

        base = osp.splitext(osp.basename(filename))[0]
        out_img_file = osp.join(out_PNG_path, base + ".png")
        tgt_coco_file = osp.join(out_coco_path, base + ".json")

        img = labelme.utils.img_data_to_arr(label_file.imageData)
        imgviz.io.imsave(out_img_file, img)
        data["images"].append(
            dict(
                license=0,
                url=None,
                file_name=osp.relpath(out_img_file, osp.dirname(out_ann_file)),
                height=img.shape[0],
                width=img.shape[1],
                date_captured=None,
                id=image_id,
            )
        )

        masks = {}  # for area
        segmentations = collections.defaultdict(list)  # for segmentation
        dataset = {}
        danns = []
        d_id = 0

        for shape in label_file.shapes:
            points = shape["points"]
            label = shape["label"]
            group_id = shape.get("group_id")
            shape_type = shape.get("shape_type", "polygon")
            mask = labelme.utils.shape_to_mask(
                img.shape[:2], points, shape_type
            )

            #NOTE: use the following code will crash the program on saving
            #rle = coco.maskUtils.encode(mask.T) # bytmc            
            #rle = coco.maskUtils.encode(np.asfortranarray(mask))  # bytmc
            #NOTE: use the below code
            ground_truth_binary_mask = np.array(mask, dtype=np.uint8)
            fortran_binary_mask = np.asfortranarray(ground_truth_binary_mask) #can be ommitted
            rle = binary_mask_to_rle(fortran_binary_mask)

            if group_id is None:
                group_id = uuid.uuid1()

            instance = (label, group_id)

            if instance in masks:
                masks[instance] = masks[instance] | mask
            else:
                masks[instance] = mask

            if shape_type == "rectangle":
                (x1, y1), (x2, y2) = points
                x1, x2 = sorted([x1, x2])
                y1, y2 = sorted([y1, y2])
                points = [x1, y1, x2, y1, x2, y2, x1, y2]
            if shape_type == "circle":
                (x1, y1), (x2, y2) = points
                r = np.linalg.norm([x2 - x1, y2 - y1])
                # r(1-cos(a/2))<x, a=2*pi/N => N>pi/arccos(1-x/r)
                # x: tolerance of the gap between the arc and the line segment
                n_points_circle = max(int(np.pi / np.arccos(1 - 1 / r)), 12)
                i = np.arange(n_points_circle)
                x = x1 + r * np.sin(2 * np.pi / n_points_circle * i)
                y = y1 + r * np.cos(2 * np.pi / n_points_circle * i)
                points = np.stack((x, y), axis=1).flatten().tolist()
            else:
                points = np.asarray(points).flatten().tolist()
            
            #segmentations[instance].append(points)
            if use_rle_format:
                segmentations[instance] = rle
            else:
                segmentations[instance].append(points)

            if not args.noseperate:
                # begin of annItem
                annItem = {}
                annItem['id'] = d_id
                annItem['image_id'] = int(base)

                if label not in class_name_to_id:
                    continue
                annItem['category_id'] = class_name_to_id[label]
                annItem['type'] = shape_type
                annItem['iscrowd'] = 0
                
                lpts = len(points)
                xpts = points[0:lpts:2]
                ypts = points[1:lpts:2]
                x1, x2 = min(xpts), max(xpts)
                y1, y2 = min(ypts), max(ypts)
                annItem['bbox'] = [x1, y1, x2-x1, y2-y1]
                annItem['area'] = PolyArea(xpts, ypts) 
                annItem['segmentation'] = [points]

                d_id = d_id + 1
                danns.append(annItem)
                # end of annItem

        if not args.noseperate:
            dataset['annotations'] = danns

            dinfo = {}
            dinfo['file_name']= label_file.imagePath
            dinfo['width'] = label_file.imageWidth
            dinfo['height'] = label_file.imageHeight
            dinfo['id'] = 0
            dataset['images'] = [dinfo]

            #save seperate coco files, etc.
            with open(tgt_coco_file, "w") as f:
                json.dump(dataset, f)
            print('-----> save seperate file: ', tgt_coco_file, '<-----')

        segmentations = dict(segmentations)

        for instance, mask in masks.items():
            cls_name, group_id = instance
            if cls_name not in class_name_to_id:
                continue
            cls_id = class_name_to_id[cls_name]

            mask = np.asfortranarray(mask.astype(np.uint8))
            mask = pycocotools.mask.encode(mask)
            area = float(pycocotools.mask.area(mask))
            bbox = pycocotools.mask.toBbox(mask).flatten().tolist()

            data["annotations"].append(
                dict(
                    id=len(data["annotations"]),
                    image_id=image_id,
                    category_id=cls_id,
                    segmentation=segmentations[instance],
                    area=area,
                    bbox=bbox,
                    iscrowd=0,
                )
            )

        if not args.noviz:
            viz = img
            if masks:
                labels, captions, masks = zip(
                    *[
                        (class_name_to_id[cnm], cnm, msk)
                        for (cnm, gid), msk in masks.items()
                        if cnm in class_name_to_id
                    ]
                )
                viz = imgviz.instances2rgb(
                    image=img,
                    labels=labels,
                    masks=masks,
                    captions=captions,
                    font_size=15,
                    line_width=2,
                )
            out_viz_file = osp.join(
                args.output_dir, "Visualization", base + ".jpg"
            )
            imgviz.io.imsave(out_viz_file, viz)

    with open(out_ann_file, "w") as f:
        json.dump(data, f)

if __name__ == "__main__":
    main()
    print("All done!")
