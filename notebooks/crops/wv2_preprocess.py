import random
import os
import shutil
import copy
from skimage import measure
from skimage import morphology as skim
import skimage.io as skio
import warnings
import pandas as pd
import numpy as np

ROOT_DIR = os.path.abspath("../../")
DATASET_DIR = os.path.join(ROOT_DIR, 'data/raw/wv2')
EIGHTCHANNEL_DIR = os.path.join(DATASET_DIR, 'eightchannels')
TRAIN_DIR = os.path.join(DATASET_DIR, 'train')
TEST_DIR = os.path.join(DATASET_DIR, 'test')
WV2_DIR = os.path.join(DATASET_DIR, 'gridded_wv2')
LABELS_DIR = os.path.join(DATASET_DIR, 'gridded_wv2_labels')
CONNECTED_COMP_DIR = os.path.join(DATASET_DIR, 'connected_comp_labels')
OPENED_LABELS_DIR = os.path.join(DATASET_DIR, 'opened_labels')
# Results directory
# Save submission files and test/train split csvs here
RESULTS_DIR = os.path.join(ROOT_DIR, "results/wv2/") 
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
        train_df = pd.DataFrame({'train': train_list})
        test_df = pd.DataFrame({'test': test_list})
        train_df.to_csv(os.path.join(RESULTS_DIR, 'train_ids.csv'))
        test_df.to_csv(os.path.join(RESULTS_DIR, 'test_ids.csv'))
        
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
