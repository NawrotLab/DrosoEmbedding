import os
import nibabel as nib
import numpy as np

neuropil_masks_path = '/projects/lab-data/Collaboration/Gruenwald_Kadow/Neuropils12_Masks'
neuropil_names = ['AL', 'MB', 'PENP', 'VLNP', 'CX', 'GNG', 'LX', 'SNP', 'INP', 'LH', 'OL', 'VMNP']

target_shape = [96, 137, 30] # max 0, max 1, min 2

neuropils_all_recs = {name: [] for name in neuropil_names}
for root, dirs, files in os.walk(neuropil_masks_path):
    for file_path in files:
        if file_path.endswith('.nii'):
            neuropil_masks = np.rot90(nib.load(os.path.join(root, file_path)).get_fdata(), k=3)
            for neuropil_idz, neuropil_name in enumerate(neuropil_names):
                neuropil_mask = neuropil_masks[:,:,:,neuropil_idz]
                # Pad first two axes (0 and 1) to match target shape
                pad_axis0 = max(0, target_shape[0] - neuropil_mask.shape[0])
                pad_axis1 = max(0, target_shape[1] - neuropil_mask.shape[1])
                neuropil_mask = np.pad(neuropil_mask, 
                                      ((0, pad_axis0), (0, pad_axis1), (0, 0)), 
                                      mode='constant', constant_values=0)
                
                # Adjust last axis (axis 2) to match target shape
                current_z = neuropil_mask.shape[2]
                target_z = target_shape[2]
                if current_z > target_z:
                    # Determine which side has less activity to crop from
                    planes_to_crop = current_z - target_z
                    # Calculate activity (sum of non-zero values) in first and last planes
                    first_planes_activity = np.sum(neuropil_mask[:, :, :planes_to_crop])
                    last_planes_activity = np.sum(neuropil_mask[:, :, -planes_to_crop:])
                    
                    if first_planes_activity <= last_planes_activity:
                        # Crop from beginning
                        neuropil_mask = neuropil_mask[:, :, planes_to_crop:]
                    else:
                        # Crop from end
                        neuropil_mask = neuropil_mask[:, :, :target_z]
                elif current_z < target_z:
                    # Pad if too small
                    neuropil_mask = np.pad(neuropil_mask, 
                                          ((0, 0), (0, 0), (0, target_z - current_z)), 
                                          mode='constant', constant_values=0)
                
                neuropils_all_recs[neuropil_name].append(neuropil_mask)

# Convert lists to numpy arrays and compute maximum projection for each neuropil
print(len(neuropils_all_recs["AL"]))
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

print(f"Final template shape: {neuropil_template.shape}")
nib.save(nib.Nifti1Image(neuropil_template, np.eye(4)), os.path.join("/projects/lab-data/Collaboration/Gruenwald_Kadow/neuropil_template/Neuropils12_Template_allRecs.nii"))
# nib.save(nib.Nifti1Image(neuropil_template, np.eye(4)), os.path.join(neuropil_masks_path, 'Neuropils12_Template.nii'))
