from mrcnn.config import Config
import numpy as np

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
    MAX_GT_INSTANCES = 28 # determined using inspect_crop_data.ipynb

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
