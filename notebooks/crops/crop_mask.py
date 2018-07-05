"""
Mask R-CNN
Train on the fields segmentation dataset from Debats 2016 https://doi.org/10.1016/j.rse.2016.03.010 or other remotely sensed datasets.

Original nucleus.py example written by Waleed Abdulla at 
https://github.com/matterport/Mask_RCNN/blob/master/samples/nucleus/nucleus.py
------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from ImageNet weights
    python3 crop_mask.py train --dataset=data/raw/wv2 --subset=train --weights=imagenet

    # Train a new model starting from specific weights file
    python3 crop_mask.py train --dataset=data/raw/wv2 --subset=train --weights=/path/to/weights.h5

    # Resume training a model that you had trained earlier
    python3 crop_mask.py train --dataset=data/raw/wv2 --subset=train --weights=last

    # Generate submission file
    python3 crop_mask.py detect --dataset=data/raw/wv2 --subset=train --weights=<last or /path/to/weights.h5>
"""


############################################################
#  Pre-processing and train/test split
############################################################
# Set matplotlib backend
# This has to be done before other importa that might
# set it, but only if we're running in script mode
# rather than being imported.
if __name__ == '__main__':
    import matplotlib
    # Agg backend runs without a display
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
import random
import os
import sys
import shutil
import copy
from skimage import measure
from skimage import morphology as skim
import skimage.io as skio
import warnings

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")
DATASET_DIR = os.path.join(ROOT_DIR, 'data/raw/wv2')
EIGHTCHANNEL_DIR = os.path.join(DATASET_DIR, 'eightchannels')
TRAIN_DIR = os.path.join(DATASET_DIR, 'train')
TEST_DIR = os.path.join(DATASET_DIR, 'test')
WV2_DIR = os.path.join(DATASET_DIR, 'gridded_wv2')
LABELS_DIR = os.path.join(DATASET_DIR, 'gridded_wv2_labels')
CONNECTED_COMP_DIR = os.path.join(DATASET_DIR, 'connected_comp_labels')
OPENED_LABELS_DIR = os.path.join(DATASET_DIR, 'opened_labels')
try:
    os.mkdir(OPENED_LABELS_DIR)
    os.mkdir(CONNECTED_COMP_DIR)
    os.mkdir(EIGHTCHANNEL_DIR)
    os.mkdir(MODEL_DIR)
    os.mkdir(TRAIN_DIR)
    os.mkdir(TEST_DIR)
    os.mkdir(WV2_DIR)
    os.mkdir(LABELS_DIR)
except:
    FileExistsError
    
random.seed(4)
def preprocess():
    
    def remove_dir_folders(directory):
        '''
        Removes all files and sub-folders in a folder and keeps the folder.
        '''
    
        folderlist = [ f for f in os.listdir(directory)]
        for f in folderlist:
            shutil.rmtree(os.path.join(directory,f))

    def load_merge_wv2(image_id, source_dir):
        """Load the specified wv2 os/gs image pairs and return a [H,W,8] 
        Numpy array. Channels are ordered [B, G, R, NIR, B, G, R, NIR], OS 
        first.
        """
        # Load image
        os_path = source_dir+'/'+image_id+'_MS_OS.tif'
        gs_path = source_dir+'/'+image_id+'_MS_GS.tif'
        os_image = skio.imread(os_path)
        gs_image = skio.imread(gs_path)
        # If has more than 4 bands, select correct bands 
        # will need to provide image config in future
        # to programmaticaly use correct band mappings
        if os_image.shape[-1] != 4:
            os_image = np.dstack((os_image[:,:,1:3],os_image[:,:,4],os_image[:,:,6]))
        if gs_image.shape[-1] != 4:
            gs_image = np.dstack((gs_image[:,:,1:3],gs_image[:,:,4],gs_image[:,:,6]))
        stacked_image = np.dstack((os_image, gs_image))
        stacked_image_path = EIGHTCHANNEL_DIR +'/'+ image_id + '_OSGS_ms.tif'
        assert stacked_image.ndim == 3
        if -1.7e+308 not in stacked_image:
            skio.imsave(stacked_image_path,stacked_image, plugin='tifffile')
        #else:
            #might try later
            #stacked_image[stacked_image==-1.7e+308]=0
            #skio.imsave(stacked_image_path,stacked_image, plugin='tifffile')
            
    # all files, including ones we don't care about
    file_ids_all = next(os.walk(WV2_DIR))[2]
    # all multispectral on and off season tifs
    image_ids_all = [image_id for image_id in file_ids_all if 'MS' in image_id and '.aux' not in image_id]
    #check for duplicates
    assert len(image_ids_all) == len(set(image_ids_all))

    image_ids_gs = [image_id for image_id in image_ids_all if 'GS' in image_id]
    image_ids_os = [image_id for image_id in image_ids_all if 'OS' in image_id]

    #check for equality
    assert len(image_ids_os) == len(image_ids_gs)

    image_ids_short = [image_id[0:9] for image_id in image_ids_gs]

    for imid in image_ids_short:
        load_merge_wv2(imid, WV2_DIR)

    image_list = next(os.walk(EIGHTCHANNEL_DIR))[2]
    
    def move_img_to_folder(filename):
        '''Moves a file with identifier pattern ZA0165086_MS_GS.tif to a 
        folder path ZA0165086/image/ZA0165086_MS_GS.tif
        Also creates a masks folder at ZA0165086/masks'''
        
        folder_name = os.path.join(TRAIN_DIR,filename[:9])
        if os.path.isdir(folder_name):
            shutil.rmtree(folder_name)
        os.mkdir(folder_name)
        new_path = os.path.join(folder_name, 'image')
        mask_path = os.path.join(folder_name, 'masks')
        os.mkdir(new_path)
        file_path = os.path.join(EIGHTCHANNEL_DIR,filename)
        os.rename(file_path, os.path.join(new_path, filename))
        os.mkdir(mask_path)

    for img in image_list:
        move_img_to_folder(img)

    label_list = next(os.walk(LABELS_DIR))[2]

    for name in label_list:
        arr = skio.imread(os.path.join(LABELS_DIR,name))
        arr[arr == -1.7e+308]=0
        label_name = name[0:15]+'.tif'
        opened_path = os.path.join(OPENED_LABELS_DIR,name)
        kernel = np.ones((5,5))
        arr = skim.binary_opening(arr, kernel)
        arr=1*arr
        assert arr.ndim == 2
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            skio.imsave(opened_path, 1*arr)

    label_list = next(os.walk(OPENED_LABELS_DIR))[2]

    for name in label_list:
        arr = skio.imread(os.path.join(OPENED_LABELS_DIR,name))
        blob_labels = measure.label(arr, background=0)
        blob_vals = np.unique(blob_labels)
        for blob_val in blob_vals[blob_vals!=0]:
            labels_copy = blob_labels.copy()
            labels_copy[blob_labels!=blob_val] = 0
            labels_copy[blob_labels==blob_val] = 1
            label_name = name[0:15]+str(blob_val)+'.tif'
            label_path = os.path.join(CONNECTED_COMP_DIR,label_name)
            assert labels_copy.ndim == 2
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                skio.imsave(label_path, labels_copy)

    def move_mask_to_folder(filename):
        '''Moves a mask with identifier pattern ZA0165086_label_1.tif to a 
        folder path ZA0165086/mask/ZA0165086_label_01.tif. Need to run 
        connected components first.
        '''
        if os.path.isdir(os.path.join(TRAIN_DIR,filename[:9])):
            folder_path = os.path.join(TRAIN_DIR,filename[:9])
            mask_path = os.path.join(folder_path, 'masks')
            file_path = os.path.join(CONNECTED_COMP_DIR,filename)
            os.rename(file_path, os.path.join(mask_path, filename))

    mask_list = next(os.walk(CONNECTED_COMP_DIR))[2]
    for mask in mask_list:
        move_mask_to_folder(mask)

    import pandas as pd
    id_list = next(os.walk(TRAIN_DIR))[1]
    no_field_list = []
    for fid in id_list:
        mask_folder = os.path.join(DATASET_DIR,'train',fid, 'masks')
        if not os.listdir(mask_folder): 
            no_field_list.append(mask_folder)
    no_field_frame = pd.DataFrame(no_field_list)
    no_field_frame.to_csv(os.path.join(DATASET_DIR,'no_field_list.csv'))
    
    for fid in id_list:
        mask_folder = os.path.join(DATASET_DIR, 'train',fid, 'masks')
        im_folder = os.path.join(DATASET_DIR, 'train',fid, 'image')
        if not os.listdir(mask_folder): 
            im_path = os.path.join(im_folder, os.listdir(im_folder)[0])
            arr = skio.imread(im_path)
            mask = np.zeros_like(arr[:,:,0])
            assert mask.ndim == 2
            # ignores warning about low contrast image
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                skio.imsave(os.path.join(mask_folder, fid + '_label_0.tif'),mask)

    def train_test_split(train_dir, test_dir, kprop):
        """Takes a sample of folder ids and copies them to a test directory
        from a directory with all folder ids. Each sample folder contains an 
        images and corresponding masks folder."""

        remove_dir_folders(test_dir)
        sample_list = next(os.walk(train_dir))[1]
        k = round(kprop*len(sample_list))
        test_list = random.sample(sample_list,k)
        for test_sample in test_list:
            shutil.copytree(os.path.join(train_dir,test_sample),os.path.join(test_dir,test_sample))
        train_list = list(set(next(os.walk(train_dir))[1]) - set(test_list))
        return train_list, test_list
        
    train_test_split(TRAIN_DIR, TEST_DIR, .1)
    print('preprocessing complete, ready to run model.')

def get_arr_channel_mean(channel):
    means = []
    for i, fid in enumerate(id_list):
        im_folder = os.path.join('train',fid, 'image')
        im_path = os.path.join(im_folder, os.listdir(im_folder)[0])
        arr = skio.imread(im_path)
        arr[arr==-1.7e+308]=np.nan
        means.append(np.nanmean(arr[:,:,channel]))
    return np.mean(means)

############################################################
#  Set model paths and imports
############################################################

import json
import datetime
import numpy as np
from imgaug import augmenters as iaa

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
from mrcnn import model as modellib
from mrcnn import visualize

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "models/mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

# Results directory
# Save submission files here
RESULTS_DIR = os.path.join(ROOT_DIR, "results/wv2/")        

############################################################
#  Configurations
############################################################

class WV2Config(Config):
    """Configuration for training on worldview-2 imagery. 
     Overrides values specific to WV2.
    
    Descriptive documentation for each attribute is at
    https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/config.py"""
    
    def __init__(self, N):
        """Set values of computed attributes. Channel dimension is overriden, 
        replaced 3 with N as per this guideline: https://github.com/matterport/Mask_RCNN/issues/314
        THERE MAY BE OTHER CODE CHANGES TO ACCOUNT FOR 3 vs N channels. See other 
        comments."""
        # https://github.com/matterport/Mask_RCNN/wiki helpful for N channels
        # Effective batch size
        self.BATCH_SIZE = self.IMAGES_PER_GPU * self.GPU_COUNT
        
        # Input image size
        if self.IMAGE_RESIZE_MODE == "crop":
            self.IMAGE_SHAPE = np.array([self.IMAGE_MIN_DIM, self.IMAGE_MIN_DIM, N])
        else:
            self.IMAGE_SHAPE = np.array([self.IMAGE_MAX_DIM, self.IMAGE_MAX_DIM, N])

        # Image meta data length
        # See compose_image_meta() for details
        self.IMAGE_META_SIZE = 1 + 3 + 3 + 4 + 1 + self.NUM_CLASSES
        self.CHANNELS_NUM = N
    
    # LEARNING_RATE = .0001 
    
    # Image mean (RGBN RGBN) from WV2_MRCNN_PRE.ipynb
    # filling with N values, need to compute mean of each channel
    # values are for gridded wv2 no partial grids
    MEAN_PIXEL = np.array([259.6, 347.0, 259.8, 416.3, 228.23, 313.4, 187.5, 562.9])
    
    # Give the configuration a recognizable name
    NAME = "wv2-gridded-no-partial"

    # Batch size is 4 (GPUs * images/GPU).
    # New parralel_model.py allows for multi-gpu
    GPU_COUNT = 2
    IMAGES_PER_GPU = 4

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + ag

    # Use small images for faster training. Determines the image shape.
    # From build() in model.py
    # Exception("Image size must be dividable by 2 at least 6 times "
    #     "to avoid fractions when downscaling and upscaling."
    #    "For example, use 256, 320, 384, 448, 512, ... etc. "
    IMAGE_RESIZE_MODE = "crop"
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256

    # Use smaller anchors because our image and objects are small.
    # Setting Large upper scale since some fields take up nearly 
    # whole image
    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 300)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 100

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 1000
    
    #reduces the max number of field instances
    MAX_GT_INSTANCES = 50

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 100
    
    # Backbone network architecture
    # Supported values are: resnet50, resnet101.
    # You can also provide a callable that should have the signature
    # of model.resnet_graph. If you do so, you need to supply a callable
    # to COMPUTE_BACKBONE_SHAPE as well
    BACKBONE = "resnet50"
    
    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = False
    MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask
    


class WV2InferenceConfig(WV2Config):
    # Set batch size to 1 to run one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # Don't resize imagery for inferencing
    IMAGE_RESIZE_MODE = "pad64"
    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.7


############################################################
#  Dataset
############################################################

class WV2Dataset(utils.Dataset):
    """Generates the Imagery dataset."""
    
    def load_image(self, image_id):
        """Load the specified image and return a [H,W,8] Numpy array.
        Channels are ordered [B, G, R, NIR]. This is called by the 
        Keras data_generator function
        """
        # Load image
        image = skio.imread(self.image_info[image_id]['path'])
    
        assert image.shape[-1] == 8
        assert image.ndim == 3
    
        return image
    
    def load_wv2(self, dataset_dir, subset):
        """Load a subset of the fields dataset.

        dataset_dir: Root directory of the dataset
        subset: Subset to load.
                * train: training images/masks excluding testing
                * test: testing images moved by train/test split func
        """
        # Add classes. We have one class.
        # Naming the dataset wv2, and the class agriculture
        self.add_class("wv2", 1, "agriculture")

        assert subset in ["train", "test"]
        dataset_dir = os.path.join(dataset_dir, subset)
        if subset == "test":
            image_ids = test_list
        else:
            image_ids = train_list
        
        # Add images
        for image_id in image_ids:
            self.add_image(
                "wv2",
                image_id=image_id,
                path=os.path.join(dataset_dir, image_id, "image/{}.tif".format(image_id+'_OSGS_ms')))
    
    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        info = self.image_info[image_id]
        # Get mask directory from image path
        mask_dir = os.path.join(os.path.dirname(os.path.dirname(info['path'])), "masks")

        # Read mask files from .png image
        mask = []
        for f in next(os.walk(mask_dir))[2]:
            if f.endswith(".tif"):
                m = skio.imread(os.path.join(mask_dir, f)).astype(np.bool)
                mask.append(m)
                assert m.ndim == 2
        mask = np.stack(mask, axis=-1)
        assert mask.ndim == 3
        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID, we return an array of ones
        return mask, np.ones([mask.shape[-1]], dtype=np.int32)
    
    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "field":
            return info["id"]
        else:
            super(self.__class__, self).image_reference(image_id)

############################################################
#  Training
############################################################

def train(model, dataset_dir, subset):
    """Train the model."""
    # Training dataset.
    dataset_train = WV2Dataset()
    dataset_train.load_wv2(dataset_dir, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = WV2Dataset()
    dataset_val.load_wv2(dataset_dir, "test")
    dataset_val.prepare()

    # Image augmentation
    # http://imgaug.readthedocs.io/en/latest/source/augmenters.html
    augmentation = iaa.SomeOf((0, 2), [
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.OneOf([iaa.Affine(rotate=90),
                   iaa.Affine(rotate=180),
                   iaa.Affine(rotate=270)]),
        iaa.Multiply((0.8, 1.5)),
        iaa.GaussianBlur(sigma=(0.0, 5.0))
    ])

    # *** This training schedule is an example. Update to your needs ***

    print("Train all layers")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=40,
                augmentation=augmentation,
                layers='all')



############################################################
#  RLE Encoding
############################################################

############################################################
#  RLE Encoding
############################################################

def rle_encode(mask):
    """Encodes a mask in Run Length Encoding (RLE).
    Returns a string of space-separated values.
    """
    assert mask.ndim == 2, "Mask must be of shape [Height, Width]"
    # Flatten it column wise
    m = mask.T.flatten()
    # Compute gradient. Equals 1 or -1 at transition points
    g = np.diff(np.concatenate([[0], m, [0]]), n=1)
    # 1-based indicies of transition points (where gradient != 0)
    rle = np.where(g != 0)[0].reshape([-1, 2]) + 1
    # Convert second index in each pair to lenth
    rle[:, 1] = rle[:, 1] - rle[:, 0]
    return " ".join(map(str, rle.flatten()))

def rle_decode(rle, shape):
    """Decodes an RLE encoded list of space separated
    numbers and returns a binary mask."""
    rle = list(map(int, rle.split()))
    rle = np.array(rle, dtype=np.int32).reshape([-1, 2])
    rle[:, 1] += rle[:, 0]
    rle -= 1
    mask = np.zeros([shape[0] * shape[1]], np.bool)
    for s, e in rle:
        assert 0 <= s < mask.shape[0]
        assert 1 <= e <= mask.shape[0], "shape: {}  s {}  e {}".format(shape, s, e)
        mask[s:e] = 1
    # Reshape and transpose
    mask = mask.reshape([shape[1], shape[0]]).T
    return mask


def mask_to_rle(image_id, mask, scores):
    "Encodes instance masks to submission format."
    assert mask.ndim == 3, "Mask must be [H, W, count]"
    # If mask is empty, return line with image ID only
    if mask.shape[-1] == 0:
        return "{},".format(image_id)
    # Remove mask overlaps
    # Multiply each instance mask by its score order
    # then take the maximum across the last dimension
    order = np.argsort(scores)[::-1] + 1  # 1-based descending
    mask = np.max(mask * np.reshape(order, [1, 1, -1]), -1)
    # Loop over instance masks
    lines = []
    for o in order:
        m = np.where(mask == o, 1, 0)
        # Skip if empty
        if m.sum() == 0.0:
            continue
        rle = rle_encode(m)
        lines.append("{}, {}".format(image_id, rle))
    return "\n".join(lines)


############################################################
#  Detection
############################################################

def detect(model, dataset_dir, subset):
    """Run detection on images in the given directory."""
    print("Running on {}".format(dataset_dir))

    # Create directory
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    submit_dir = "submit_{:%Y%m%dT%H%M%S}".format(datetime.datetime.now())
    submit_dir = os.path.join(RESULTS_DIR, submit_dir)
    os.makedirs(submit_dir)

    # Read dataset
    dataset = WV2Dataset(8)
    dataset.load_wv2(dataset_dir, subset)
    dataset.prepare()
    # Load over images
    submission = []
    for image_id in dataset.image_ids:
        # Load image and run detection
        image = dataset.load_image(image_id)
        # Detect objects
        r = model.detect([image], verbose=0)[0]
        # Encode image to RLE. Returns a string of multiple lines
        source_id = dataset.image_info[image_id]["id"]
        rle = mask_to_rle(source_id, r["masks"], r["scores"])
        submission.append(rle)
        # Save image with masks
        visualize.display_instances(
            image, r['rois'], r['masks'], r['class_ids'],
            dataset.class_names, r['scores'],
            show_bbox=False, show_mask=False,
            title="Predictions")
        plt.savefig("{}/{}.png".format(submit_dir, dataset.image_info[image_id]["id"]))

    # Save to csv file
    submission = "ImageId,EncodedPixels\n" + "\n".join(submission)
    file_path = os.path.join(submit_dir, "submit.csv")
    with open(file_path, "w") as f:
        f.write(submission)
    print("Saved to ", submit_dir)


############################################################
#  Command Line
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Mask R-CNN for fields counting and segmentation')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'preprocess' or 'train' or 'detect. preprocess takes no arguments.'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/dataset/",
                        help='Root directory of the dataset')
    parser.add_argument('--weights', required=False,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--subset', required=False,
                        metavar="Dataset sub-directory",
                        help="Subset of dataset to run prediction on")
    args = parser.parse_args()

if args.command == "preprocess":
    preprocess()
    
else:
    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "detect":
        assert args.subset, "Provide --subset to run prediction on"
    
    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    if args.subset:
        print("Subset: ", args.subset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = WV2Config(8)
    else:
        config = WV2InferenceConfig(8)
    config.display()
    
    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model, args.dataset, args.subset)
    elif args.command == "detect":
        detect(model, args.dataset, args.subset)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'detect'".format(args.command))
