import numpy as np
import nibabel as nib
from PIL import Image

def load_and_normNIFTI(file_path, substract_Baseline = False, t_base = [0,250], save_baseframe = False, knockOutNeuropil = False, neuropil_name = 'AL'):
    file_name = file_path.split('/')[-1].split('.')[0]
    data_array = np.rot90(nib.load(file_path).get_fdata(), k=3)
    min_val = np.min(data_array)
    max_val = np.max(data_array)
    normalized_data = (data_array - min_val) / (max_val - min_val)

    if substract_Baseline:
        baseline_yxzt = normalized_data[:,:,:,t_base[0]:t_base[1]]
        baseline_yxz_meant = np.mean(baseline_yxzt, axis=3)
        baseFrame = np.mean(baseline_yxz_meant, axis=2)
        if save_baseframe:
            Image.fromarray(baseFrame).save(os.path.join(CONFIG["path_baseFrame"], f"{file_name}_BaseFrame.tiff"))

        baseFrameExtended = baseFrame[:, :, np.newaxis, np.newaxis]
        normalized_data = normalized_data - baseFrameExtended
        normalized_data[normalized_data<0] = 0


    if knockOutNeuropil:
        recNr = file_path.split('_')[-1].split('.')[0]
        mask_path = f"/projects/lab-data/Collaboration/Gruenwald_Kadow/Neuropils12_Masks/Neuropils12Registered_{recNr}.nii"
        neuropil_names = np.array(['AL', 'MB', 'PENP', 'VLNP', 'CX', 'GNG', 'LX', 'SNP', 'INP', 'LH', 'OL', 'VMNP'])
        idx_neuropil = np.where(neuropil_names == neuropil_name)[0]

        neuropil_binary = nib.load(mask_path).get_fdata()[:, :, :, idx_neuropil]
        neuropil_binary = np.transpose(neuropil_binary, (1, 0, 2, 3))  # just to match data_array shape

        # Broadcast aka. "stretch" the binary mask to match the shape of data_array
        neuropil_binary_broadcasted = np.broadcast_to(neuropil_binary, normalized_data.shape)

        # Set elements in data_array to NaN where neuropil_binary is 1
        normalized_data[neuropil_binary_broadcasted == 1] = np.nan

    return normalized_data