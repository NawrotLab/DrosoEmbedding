import numpy as np
import nibabel as nib
from PIL import Image
import os
import json
from src.utils.config_loader import load_config

CONFIG = load_config()



def load_and_normNIFTI(file_path, substract_Baseline = False, t_base = [0,250], save_baseframe = False, knockOutNeuropil = False, isolate_neuropil = False, neuropil_name = 'AL', use_aligned_template = False):
    file_name = file_path.split('/')[-1].split('.')[0]
    recNr = file_name.split('_')[-1]
    data_array = np.rot90(nib.load(file_path).get_fdata(), k=3)
    min_val = np.min(data_array)
    max_val = np.max(data_array)
    normalized_data = (data_array - min_val) / (max_val - min_val)
    
    neuropil_names = np.array(['AL', 'MB', 'PENP', 'VLNP', 'CX', 'GNG', 'LX', 'SNP', 'INP', 'LH', 'OL', 'VMNP'])    
    idx_neuropil = np.where(neuropil_names == neuropil_name)[0]
    
    # Initialize variables for template case
    original_shape = None  # Store original shape for reverting
    recording_shift = None
    output_shape = None
    output_origin_offset = None
    template_path = None  # Store template path for mask loading

    if substract_Baseline:
        baseline_yxzt = normalized_data[:,:,:,t_base[0]:t_base[1]]
        baseline_yxz_meant = np.mean(baseline_yxzt, axis=3)
        baseFrame = np.mean(baseline_yxz_meant, axis=2)
        if save_baseframe:
            Image.fromarray(baseFrame).save(os.path.join(CONFIG["path_baseFrame"], f"{file_name}_BaseFrame.tiff"))

        baseFrameExtended = baseFrame[:, :, np.newaxis, np.newaxis]
        normalized_data = normalized_data - baseFrameExtended
        normalized_data[normalized_data<0] = 0
    
    # Handle aligned template case - shift data to template space before processing
    if use_aligned_template:
        template_dir = "/projects/lab-data/Collaboration/Gruenwald_Kadow/neuropil_template"
        template_path = os.path.join(template_dir, "Neuropils12_Template_aligned.nii")
        shifts_path = os.path.join(template_dir, "recording_shifts.json")
        
        # Check if template and shifts files exist
        if not os.path.exists(template_path) or not os.path.exists(shifts_path):
            raise FileNotFoundError(f"Aligned template files not found. Template: {template_path}, Shifts: {shifts_path}")
        
        # Load shifts metadata
        with open(shifts_path, 'r') as f:
            shifts_metadata = json.load(f)
        
        # Get shift for this recording
        if recNr not in shifts_metadata['recordings']:
            raise ValueError(f"Recording {recNr} not found in shifts metadata")
        
        # Get shift for this recording, output shape, and output origin offset
        recording_shift = np.array(shifts_metadata['recordings'][recNr]['shift'])
        output_shape = np.array(shifts_metadata['output_shape'])
        output_origin_offset = np.array(shifts_metadata['output_origin_offset'])
        
        # Store original shape for reverting later
        original_shape = normalized_data.shape[:3]  # (y, x, z) without time dimension
        
        # Shift data to align with template space
        shift_y, shift_x, shift_z = recording_shift.astype(int)
        aligned_data = np.zeros((output_shape[0], output_shape[1], output_shape[2], normalized_data.shape[3]), 
                               dtype=normalized_data.dtype)
        
        # Calculate target start and end positions
        tgt_y_start = output_origin_offset[0] + shift_y
        tgt_x_start = output_origin_offset[1] + shift_x
        tgt_z_start = output_origin_offset[2] + shift_z
        tgt_y_end = tgt_y_start + normalized_data.shape[0]
        tgt_x_end = tgt_x_start + normalized_data.shape[1]
        tgt_z_end = tgt_z_start + normalized_data.shape[2]
        
        # Calculate source start and end positions
        src_y_start = max(0, -tgt_y_start)
        src_y_end = min(normalized_data.shape[0], output_shape[0] - tgt_y_start)
        src_x_start = max(0, -tgt_x_start)
        src_x_end = min(normalized_data.shape[1], output_shape[1] - tgt_x_start)
        src_z_start = max(0, -tgt_z_start)
        src_z_end = min(normalized_data.shape[2], output_shape[2] - tgt_z_start)
        
        # Calculate valid target start and end positions
        tgt_y_start_valid = max(0, tgt_y_start)
        tgt_y_end_valid = min(output_shape[0], tgt_y_end)
        tgt_x_start_valid = max(0, tgt_x_start)
        tgt_x_end_valid = min(output_shape[1], tgt_x_end)
        tgt_z_start_valid = max(0, tgt_z_start)
        tgt_z_end_valid = min(output_shape[2], tgt_z_end)
        
        # Adjust source if target was clipped
        if tgt_y_start_valid != tgt_y_start:
            src_y_start += (tgt_y_start_valid - tgt_y_start)
        if tgt_x_start_valid != tgt_x_start:
            src_x_start += (tgt_x_start_valid - tgt_x_start)
        if tgt_z_start_valid != tgt_z_start:
            src_z_start += (tgt_z_start_valid - tgt_z_start)
        
        # Copy data to aligned space
        for t in range(normalized_data.shape[3]):
            if (tgt_y_end_valid > tgt_y_start_valid and 
                tgt_x_end_valid > tgt_x_start_valid and 
                tgt_z_end_valid > tgt_z_start_valid and
                src_y_end > src_y_start and 
                src_x_end > src_x_start and 
                src_z_end > src_z_start):
                aligned_data[tgt_y_start_valid:tgt_y_end_valid, 
                            tgt_x_start_valid:tgt_x_end_valid, 
                            tgt_z_start_valid:tgt_z_end_valid, t] = \
                    normalized_data[src_y_start:src_y_end, 
                                   src_x_start:src_x_end, 
                                   src_z_start:src_z_end, t]
        
        normalized_data = aligned_data
        mask_path = None  # Will use template mask instead
    else:
        mask_path = f"/projects/lab-data/Collaboration/Gruenwald_Kadow/Neuropils12_Masks/Neuropils12Registered_{recNr}.nii"




    if knockOutNeuropil:
        if use_aligned_template:
            # Load template mask (already in aligned/template space)
            # normalized_data is in aligned space with shape (output_shape[0], output_shape[1], output_shape[2], time)
            # Template mask has shape (output_shape[0], output_shape[1], output_shape[2])
            # The template mask is a max projection, so it may cover more than a single recording's neuropil
            # but this is correct - we apply it in aligned space, then extract only the data region when reverting
            template_masks = nib.load(template_path).get_fdata()
            template_mask_3d = template_masks[:, :, :, idx_neuropil[0]]
            # Ensure binary and add time dimension
            neuropil_binary = (template_mask_3d > 0).astype(np.float32)
            neuropil_binary = neuropil_binary[:, :, :, np.newaxis]  # Add time dimension: (y, x, z, 1)
        else:
            neuropil_binary = nib.load(mask_path).get_fdata()[:, :, :, idx_neuropil]
            neuropil_binary = np.transpose(neuropil_binary, (1, 0, 2, 3))  # just to match data_array shape
        
        # Broadcast aka. "stretch" the binary mask to match the shape of data_array
        neuropil_binary_broadcasted = np.broadcast_to(neuropil_binary, normalized_data.shape)
        # Verify shapes match
        if neuropil_binary_broadcasted.shape != normalized_data.shape:
            raise ValueError(f"Mask shape {neuropil_binary_broadcasted.shape} doesn't match data shape {normalized_data.shape}")
        # Set elements in data_array to NaN where neuropil_binary is 1
        normalized_data[neuropil_binary_broadcasted == 1] = np.nan

    if isolate_neuropil:
        if use_aligned_template:
            # Load template mask (already in aligned/template space)
            # normalized_data is in aligned space with shape (output_shape[0], output_shape[1], output_shape[2], time)
            # Template mask has shape (output_shape[0], output_shape[1], output_shape[2])
            template_masks = nib.load(template_path).get_fdata()
            template_mask_3d = template_masks[:, :, :, idx_neuropil[0]]
            # Ensure binary and add time dimension
            neuropil_binary = (template_mask_3d > 0).astype(np.float32)
            neuropil_binary = neuropil_binary[:, :, :, np.newaxis]  # Add time dimension: (y, x, z, 1)
        else:
            neuropil_binary = nib.load(mask_path).get_fdata()[:, :, :, idx_neuropil]
            neuropil_binary = np.transpose(neuropil_binary, (1, 0, 2, 3))
        
        neuropil_binary_broadcasted = np.broadcast_to(neuropil_binary, normalized_data.shape)
        normalized_data[neuropil_binary_broadcasted == 0] = np.nan

    # Revert to original shape if using aligned template
    if use_aligned_template and original_shape is not None:
        # Extract the region that corresponds to the original data location
        # The original data was placed at position (tgt_y_start_valid, tgt_x_start_valid, tgt_z_start_valid)
        # and has size original_shape
        
        # Recalculate the target position (same as when we placed it)
        shift_y, shift_x, shift_z = recording_shift.astype(int)
        tgt_y_start = output_origin_offset[0] + shift_y
        tgt_x_start = output_origin_offset[1] + shift_x
        tgt_z_start = output_origin_offset[2] + shift_z
        
        tgt_y_start_valid = max(0, tgt_y_start)
        tgt_x_start_valid = max(0, tgt_x_start)
        tgt_z_start_valid = max(0, tgt_z_start)
        
        tgt_y_end_valid = min(output_shape[0], tgt_y_start_valid + original_shape[0])
        tgt_x_end_valid = min(output_shape[1], tgt_x_start_valid + original_shape[1])
        tgt_z_end_valid = min(output_shape[2], tgt_z_start_valid + original_shape[2])
        
        # Extract the region corresponding to original data
        # Initialize with NaN to preserve masked regions
        reverted_data = np.full((original_shape[0], original_shape[1], original_shape[2], normalized_data.shape[3]),
                               np.nan, dtype=normalized_data.dtype)
        
        # Calculate how much of the original region is available
        extract_y_size = tgt_y_end_valid - tgt_y_start_valid
        extract_x_size = tgt_x_end_valid - tgt_x_start_valid
        extract_z_size = tgt_z_end_valid - tgt_z_start_valid
        
        # Handle case where original data extends beyond aligned space boundaries
        src_y_start = max(0, -tgt_y_start)
        src_x_start = max(0, -tgt_x_start)
        src_z_start = max(0, -tgt_z_start)
        
        # Copy the data back to original shape (preserving NaN values from mask)
        reverted_data[src_y_start:src_y_start+extract_y_size,
                     src_x_start:src_x_start+extract_x_size,
                     src_z_start:src_z_start+extract_z_size, :] = \
            normalized_data[tgt_y_start_valid:tgt_y_end_valid,
                          tgt_x_start_valid:tgt_x_end_valid,
                          tgt_z_start_valid:tgt_z_end_valid, :]
        
        normalized_data = reverted_data

    return normalized_data

