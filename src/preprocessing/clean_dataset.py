import logging
import os
import nibabel as nib
from PIL import Image
import numpy as np
from tqdm import tqdm
import tifffile
import time
import pickle
from src.utils.config_loader import load_config
from src.utils.imgTools import knockNeuropilOut


def main(logger, config):
    # get variables 
    outputID = config['data']['preprocessing']['id']
    neuropil = config['data']['neuropil']
    remove_neuropil = config['data']['preprocessing']['remove_neuropil']
    isolate_neuropil = config['data']['preprocessing']['isolate_neuropil']
    basepoints = config['data']['preprocessing']['base_points']
    
    ROOT_PATH = config['paths']['root']
    path_data = f'{ROOT_PATH}/Paul_LFM_Data'
    path_output = f'{ROOT_PATH}/imgs4DL/{outputID}'
    path_baseFrame = f'{path_output}/BaseFrames/'
    path_entireRecs = f'{ROOT_PATH}/imgs4DL/KO_niftis/'
    path_pixelPlot = f'{ROOT_PATH}/LogTransformedPixelActivities'

    if isolate_neuropil:
        outputID = f'{outputID}_{neuropil}_isolated'
    elif remove_neuropil:
        outputID = f'{outputID}_KO_{neuropil}'

    niftis = []
    for root, dirs, files in os.walk(path_data):
        for nii in files[1:]:
            if not nii.startswith('._'):
                niftis.append(root + '/' + nii)
    niftis = sorted(niftis)

    with open(config['paths']['peakIDs_Times_All'], 'rb') as file:
        idsAll, times_all = pickle.load(file) # I am not using times_all

    for path in [path_output, path_baseFrame]:
        if not os.path.exists(path):
            os.makedirs(path)

    file_names = []
    start_time = time.time()
    nii_start_indx = 0 # in case it crashed use this to not start from the beginning


    for idx, nifti in tqdm(enumerate(niftis[nii_start_indx:])):
        idx = idx+nii_start_indx
        file_name = nifti.split('/')[-1].split('.')[0]
        nii_recNR = file_name.split('_')[-1]
        






if __name__ == "__main__":
    config = load_config()
    logger = setup_logger(task_name=config["run_id"], log_dir="logs/slurm")
    main(logger, config)