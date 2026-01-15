import os
import nibabel as nib
import numpy as np
import json
from scipy.ndimage import center_of_mass

neuropil_masks_path = '/projects/lab-data/Collaboration/Gruenwald_Kadow/Neuropils12_Masks'
neuropil_names = ['AL', 'MB', 'PENP', 'VLNP', 'CX', 'GNG', 'LX', 'SNP', 'INP', 'LH', 'OL', 'VMNP']

# Dictionary to store shifts for each recording file
shifts_dict = {}

# First pass: collect all masks and compute their centers of mass
print("\nFirst pass: Computing centers of mass and determining output shape...")
file_info = []  # Store (file_path, recording_id, mask_shape, avg_center) for each recording
all_centers = []  # Collect all centers to compute global reference

for root, dirs, files in os.walk(neuropil_masks_path):
    for file_path in files:
        if file_path.endswith('.nii'):
            full_path = os.path.join(root, file_path)
            recording_id = file_path.split('_')[-1].split('.')[0] if '_' in file_path else file_path.split('.')[0]
            
            neuropil_masks = np.rot90(nib.load(full_path).get_fdata(), k=3)
            
            # Compute center of mass for each neuropil in this recording
            centers = {}
            for neuropil_idz, neuropil_name in enumerate(neuropil_names):
                neuropil_mask = neuropil_masks[:, :, :, neuropil_idz]
                
                # Compute center of mass (returns (y, x, z) in numpy convention)
                com = center_of_mass(neuropil_mask)
                
                # Handle case where mask is empty (all zeros)
                if np.isnan(com).any():
                    # Use geometric center as fallback
                    com = np.array([neuropil_mask.shape[0] / 2, 
                                   neuropil_mask.shape[1] / 2, 
                                   neuropil_mask.shape[2] / 2])
                else:
                    com = np.array(com)
                
                centers[neuropil_name] = com
            
            # Compute average center across all neuropils for this recording
            # This gives us a single center per recording
            all_centers_rec = np.array(list(centers.values()))
            avg_center = np.mean(all_centers_rec, axis=0)
            all_centers.append(avg_center)
            
            # Store info for this recording
            file_info.append((full_path, recording_id, neuropil_masks.shape, avg_center))

# Compute global reference center (average of all recording centers)
# This will be our alignment target
if len(all_centers) > 0:
    global_reference_center = np.mean(np.array(all_centers), axis=0)
    print(f"Global reference center: {global_reference_center}")
else:
    raise ValueError("No masks found!")

# Calculate shifts for each recording
for full_path, recording_id, mask_shape, avg_center in file_info:
    # Shift needed to move this recording's center to the global reference
    # Positive shift means the mask needs to move in the positive direction
    shift = global_reference_center - avg_center
    shifts_dict[recording_id] = {
        'shift': shift.tolist(),
        'original_center': avg_center.tolist(),
        'reference_center': global_reference_center.tolist(),
        'file_path': full_path
    }

# Determine output shape that can accommodate all shifted masks
# We need to find the maximum extent in each dimension after shifting
max_extents = np.zeros(3)  # Maximum position reached in each dimension
min_extents = np.zeros(3)  # Minimum position reached in each dimension

for full_path, recording_id, mask_shape, avg_center in file_info:
    shift = np.array(shifts_dict[recording_id]['shift'])
    mask_shape_3d = mask_shape[:3]
    
    # Calculate where the mask will be placed after shifting
    # The mask's origin (0,0,0) moves to position 'shift'
    # The mask's end moves to position 'shift + mask_shape'
    mask_start = shift
    mask_end = shift + mask_shape_3d
    
    max_extents = np.maximum(max_extents, mask_end)
    min_extents = np.minimum(min_extents, mask_start)

# Calculate required size (add some margin for safety)
margin = 10  # Extra padding on each side
output_shape = (max_extents - min_extents + 2 * margin).astype(int)
output_origin_offset = (-min_extents + margin).astype(int)  # Offset to place reference at center

print(f"\nOutput shape (accommodates all shifted masks): {output_shape}")
print(f"\nShift statistics:")
print(f"  Min extents (after shifting): {min_extents}")
print(f"  Max extents (after shifting): {max_extents}")
print(f"  Extent range: {max_extents - min_extents}")
print(f"  Output origin offset: {output_origin_offset}")
print(f"  Global reference will be at position: {output_origin_offset + global_reference_center}")

# Second pass: align and collect masks
print("\nSecond pass: Aligning masks...")
neuropils_all_recs = {name: [] for name in neuropil_names}

for root, dirs, files in os.walk(neuropil_masks_path):
    for file_path in files:
        if file_path.endswith('.nii'):
            full_path = os.path.join(root, file_path)
            recording_id = file_path.split('_')[-1].split('.')[0] if '_' in file_path else file_path.split('.')[0]
            
            if recording_id not in shifts_dict:
                continue
                
            neuropil_masks = np.rot90(nib.load(full_path).get_fdata(), k=3)
            shift = np.array(shifts_dict[recording_id]['shift'])
            
            for neuropil_idz, neuropil_name in enumerate(neuropil_names):
                neuropil_mask = neuropil_masks[:, :, :, neuropil_idz]
                
                # Apply shift by translating the mask
                # Shift is in (y, x, z) format (numpy convention)
                shift_y, shift_x, shift_z = shift.astype(int)
                
                # Create output array
                aligned_mask = np.zeros(output_shape, dtype=neuropil_mask.dtype)
                
                # Calculate where to place the mask in the output array
                # The mask's origin (0,0,0) should be at position (shift + output_origin_offset)
                tgt_y_start = output_origin_offset[0] + shift_y
                tgt_x_start = output_origin_offset[1] + shift_x
                tgt_z_start = output_origin_offset[2] + shift_z
                
                tgt_y_end = tgt_y_start + neuropil_mask.shape[0]
                tgt_x_end = tgt_x_start + neuropil_mask.shape[1]
                tgt_z_end = tgt_z_start + neuropil_mask.shape[2]
                
                # Calculate valid source and target regions (handle boundaries)
                # Source: what part of the original mask to copy
                src_y_start = max(0, -tgt_y_start)
                src_y_end = min(neuropil_mask.shape[0], output_shape[0] - tgt_y_start)
                src_x_start = max(0, -tgt_x_start)
                src_x_end = min(neuropil_mask.shape[1], output_shape[1] - tgt_x_start)
                src_z_start = max(0, -tgt_z_start)
                src_z_end = min(neuropil_mask.shape[2], output_shape[2] - tgt_z_start)
                
                # Target: where in output array to place it
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
                
                # Copy the shifted mask
                if (tgt_y_end_valid > tgt_y_start_valid and 
                    tgt_x_end_valid > tgt_x_start_valid and 
                    tgt_z_end_valid > tgt_z_start_valid and
                    src_y_end > src_y_start and 
                    src_x_end > src_x_start and 
                    src_z_end > src_z_start):
                    aligned_mask[tgt_y_start_valid:tgt_y_end_valid, 
                                tgt_x_start_valid:tgt_x_end_valid, 
                                tgt_z_start_valid:tgt_z_end_valid] = \
                        neuropil_mask[src_y_start:src_y_end, 
                                     src_x_start:src_x_end, 
                                     src_z_start:src_z_end]
                
                neuropils_all_recs[neuropil_name].append(aligned_mask)

# Convert lists to numpy arrays and compute maximum projection for each neuropil
print(f"\nNumber of recordings: {len(neuropils_all_recs['AL'])}")
neuropil_template = None
for neuropil_name in neuropil_names:
    if len(neuropils_all_recs[neuropil_name]) > 0:
        # Stack all recordings for this neuropil: shape will be (n_recordings, height, width, depth)
        neuropil_array = np.array(neuropils_all_recs[neuropil_name])
        print(f"{neuropil_name}: {neuropil_array.shape}")
        
        # Maximum projection across all recordings: shape will be (height, width, depth)
        neuropil_max = np.max(neuropil_array, axis=0)
        
        # Add dimension for concatenation: shape becomes (height, width, depth, 1)
        neuropil_max = neuropil_max[:, :, :, np.newaxis]
        
        # Concatenate along the last axis to create template
        if neuropil_template is None:
            neuropil_template = neuropil_max
        else:
            neuropil_template = np.concatenate((neuropil_template, neuropil_max), axis=3)

print(f"\nFinal template shape: {neuropil_template.shape}")

# Save the template
output_dir = "/projects/lab-data/Collaboration/Gruenwald_Kadow/neuropil_template"
os.makedirs(output_dir, exist_ok=True)
template_path = os.path.join(output_dir, "Neuropils12_Template_aligned.nii")
nib.save(nib.Nifti1Image(neuropil_template, np.eye(4)), template_path)
print(f"Template saved to: {template_path}")

# Save shifts to JSON file along with metadata
shifts_path = os.path.join(output_dir, "recording_shifts.json")
shifts_metadata = {
    'output_shape': output_shape.tolist(),
    'output_origin_offset': output_origin_offset.tolist(),
    'global_reference_center': global_reference_center.tolist(),
    'recordings': shifts_dict
}
with open(shifts_path, 'w') as f:
    json.dump(shifts_metadata, f, indent=2)
print(f"Shifts saved to: {shifts_path}")

print("\nDone!")

