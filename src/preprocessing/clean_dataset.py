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
# from src.utils.imgTools import knockNeuropilOut
from src.utils.logger import setup_logger
from src.utils.imgTools import load_and_normNIFTI



def main(logger, config):
    # get variables 
    neuropil = config['data']['preprocessing']['neuropil']
    remove_neuropil = config['data']['preprocessing']['remove_neuropil']
    isolate_neuropil = config['data']['preprocessing']['isolate_neuropil']
    use_aligned_template = config['data']['preprocessing'].get('use_aligned_neuropil_template', False)
    basepoints = config['data']['preprocessing']['base_points']
    timesID = config['data']['preprocessing']['times']

    if isolate_neuropil:
        outputID = f'{config['data']['preprocessing']['method_ch']}_{timesID}_{config['data']['preprocessing']['neuropil']}'
    elif remove_neuropil:
        outputID = f'{config['data']['preprocessing']['method_ch']}_{timesID}_KO_{config['data']['preprocessing']['neuropil']}'
    else:
        outputID = f'{config['data']['preprocessing']['method_ch']}_{timesID}'
    
    # Add template suffix if using aligned template
    if use_aligned_template:
        outputID = f'{outputID}-template'
    # print(f'Output ID: {outputID}')
    
    ROOT_PATH = config['paths']['data_root']
    path_data = f'{ROOT_PATH}/Paul_LFM_Data'
    path_output = f'/localscratch/aabdel/imgs4DL/{outputID}'
    path_baseFrame = f'{path_output}/BaseFrames/'
    # path_entireRecs = f'{ROOT_PATH}/imgs4DL/KO_niftis/'
    # path_pixelPlot = f'{ROOT_PATH}/LogTransformedPixelActivities'


    niftis = []
    for root, dirs, files in os.walk(path_data):
        for nii in files[1:]:
            if not nii.startswith('._'):
                niftis.append(root + '/' + nii)
    niftis = sorted(niftis)

    recordings_file = os.environ.get('RECORDINGS_FILE')
    recording_filter = None
    if recordings_file:
        with open(recordings_file) as f:
            recording_filter = {line.strip() for line in f if line.strip()}
        logger.info(f'Recording filter active: {len(recording_filter)} recordings from {recordings_file}')

    with open(config['paths']['peakIDs_Times_All'], 'rb') as file:
        id_times_dict = pickle.load(file)

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

        if recording_filter and file_name not in recording_filter:
            continue

        if file_name in id_times_dict.keys():
            logger.info(f'Processing {file_name} ({nifti})')
            output_dir = f'{path_output}/{file_name}'
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            rec_data = load_and_normNIFTI(nifti, substract_Baseline=True, t_base=[0, 250], save_baseframe=False, knockOutNeuropil=remove_neuropil, isolate_neuropil=isolate_neuropil, neuropil_name=neuropil, use_aligned_template=use_aligned_template)
            if timesID == 'logTs':
                times = id_times_dict[file_name]
            elif timesID == 'allTs':
                times = np.arange(0, rec_data.shape[3])

            for t in times:
                meanZ = np.nanmean(rec_data[:,:,:,t], axis=2)
                meanZ = np.nan_to_num(meanZ, nan=0.0)
                tiff_name = f'{file_name}_{t}.tiff'
                tifffile.imwrite(f'{output_dir}/{tiff_name}', meanZ)
        






if __name__ == "__main__":
    config = load_config()
    logger = setup_logger(task_name=config["run_id"], log_dir="logs/slurm")
    main(logger, config)