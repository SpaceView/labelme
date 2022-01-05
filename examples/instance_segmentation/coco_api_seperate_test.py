#
# windows version cocoapi
# https://github.com/philferriere/cocoapi
#
#
import pycocotools
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import json
import os
 
import matplotlib as mpl
mpl.use('TkAgg')
import pylab
import matplotlib.rcsetup as rcsetup
pylab.rcParams['figure.figsize'] = (8.0, 10.0)
 
#dataDir='..'
#dataType='val2017'
#dataDir='F:/BigData/msCoco2014'
#dataType='val2014'
 
dataDir='E:/BigData/Coco2017'
dataType='val2017'
annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)
targetDir = 'E:/BigData/Coco2017/seperate/'
 
# initialize COCO api for instance annotations
coco=COCO(annFile)
 
# display COCO categories and supercategories
catIds = coco.getCatIds()
cats = coco.loadCats(catIds)
#print the names out
nms=[cat['name'] for cat in cats] 
print('COCO categories: \n{}\n'.format(' '.join(nms)))
#print the supercat out
nms = set([cat['supercategory'] for cat in cats])
print('COCO supercategories: \n{}'.format(' '.join(nms)))
 
# recursively display all images and its masks
imgIds = coco.getImgIds()
for id in imgIds: 
    annIds = coco.getAnnIds([id], catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)
    imgIds = coco.getImgIds(imgIds = [id])
    img = coco.loadImgs(imgIds[0])[0]
    file_name_ext=img['file_name']
    (filename,extension) = os.path.splitext(file_name_ext)
    #file_path = "coco/" + filename + ".json"
    file_path = targetDir + filename + ".json"
    data = {"annotations":anns}

    #just for test purpose (RLE format or POLYGON format?), can be commented out
    for item in anns:
        if not 'segmentation' in item:
            continue
        seg = item['segmentation']
        if not 'size' in seg or not 'counts' in seg:
            continue
        print('got a RLE encoding')
        msk = coco.annToMask(item)
        print(msk.shape)

    with open(file_path, 'w') as result_file:
        json.dump(data, result_file)

    #below code just for test of display, can be commented out
    I = io.imread('%s/%s/%s'%(dataDir,dataType,img['file_name']))
    mpl.pyplot.imshow(I)
    mpl.pyplot.axis('off')
    coco.showAnns(anns)
    