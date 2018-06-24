import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

# Import Mask RCNN
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.config import Config
from mrcnn.model import log
#### All of this needs to be refactored to config.yaml
# Root directory of the project
ROOT_DIR = os.path.abspath("/home/rave/tana-crunch/waves/deepimagery/data/raw/wv2")

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "models")

# # Local path to trained weights file
# COCO_MODEL_PATH = os.path.join(MODEL_DIR, "mask_rcnn_coco.h5")
# # Download COCO trained weights from Releases if needed
# if not os.path.exists(COCO_MODEL_PATH):
#     utils.download_trained_weights(COCO_MODEL_PATH)
EIGHTCHANNEL_DIR = os.path.join(ROOT_DIR, 'eightchannels')
TRAIN_DIR = os.path.join(ROOT_DIR, 'images')
VALIDATION_DIR = os.path.join(ROOT_DIR, 'wv2/masks')
TEST_DIR = os.path.join(ROOT_DIR, 'test')
IMAGERY_DIR = os.path.join(ROOT_DIR, 'projectedtiffs')
GROUNDTRUTH_DIR = os.path.join(ROOT_DIR, 'rasterized_wv2_labels')

try:
    os.mkdir(EIGHTCHANNEL_DIR)
    os.mkdir(MODEL_DIR)
    os.mkdir(TRAIN_DIR)
    os.mkdir(VALIDATION_DIR)
    os.mkdir(TEST_DIR)
except:
    FileExistsError
####
def load_merge_wv2(image_id):
    # Load image\n",
    os_path = IMAGERY_DIR+'/'+image_id+'_MS_OS.tif'
    gs_path = IMAGERY_DIR+'/'+image_id+'_MS_GS.tif'
    os_image = skimage.io.imread(os_path)
    gs_image = skimage.io.imread(gs_path)
    # If has more than 4 bands, select correct bands
    # will need to provide image config in future
    # to programmaticaly use correct band mappings
    if os_image.shape[-1] != 4:
       os_image = np.dstack((os_image[:,:,1:3],os_image[:,:,4],os_image[:,:,6]))
    if gs_image.shape[-1] != 4:
       gs_image = np.dstack((gs_image[:,:,1:3],gs_image[:,:,4],gs_image[:,:,6]))
    stacked_image = np.dstack((os_image, gs_image))
    stacked_image_path = EIGHTCHANNEL_DIR +'/'+ image_id + '_OSGS_ms.tif'
    return (stacked_image_path, stacked_image)



# all files, including ones we don't care about
file_ids_all = next(os.walk(IMAGERY_DIR))[2],
# all multispectral on and off season tifs\n",
image_ids_all = [image_id for image_id in file_ids_all if 'MS' in image_id]
#check for duplicates
assert len(image_ids_all) != len(set(image_ids_all))

image_ids_gs = [image_id for image_id in image_ids_all if 'GS' in image_id]
image_ids_os = [image_id for image_id in image_ids_all if 'OS' in image_id]

#check for equality
assert len(image_ids_os) == len(image_ids_gs)
image_ids_short = [image_id[0:2] for image_id in image_ids_gs]
stacked_dict = {}
for imid in image_ids_short:
    path, arr = load_merge_wv2(imid)
    stacked_dict.update({path:arr})
for key, val in stacked_dict.items():
    skimage.io.imsave(key,val,plugin='tifffile')

def train_test_split(imagerydir, traindir, testdir, kprop):
    image_list = next(os.walk(imagerydir))[2]
    k = round(kprop*len(image_list))
    test_list = random.sample(image_list,k)
    for test in test_list:
        shutil.copyfile(os.path.join(imagerydir,test),os.path.join(testdir,test))
    train_list = list(set(next(os.walk(imagerydir))[2]) - set(test_list))
    for train in train_list:
       shutil.copyfile(os.path.join(imagerydir,train),os.path.join(traindir,train))
    print(len(train_list, "train list len"))
    print(len(test_list, "test list len"))
train_test_split(EIGHTCHANNEL_DIR,TRAIN_DIR, TEST_DIR, .1)
groundtruth_list = next(os.walk(GROUNDTRUTH_DIR))[2]
for file in groundtruth_list:
    shutil.copyfile(os.path.join(GROUNDTRUTH_DIR,file),os.path.join(VALIDATION_DIR,file))
