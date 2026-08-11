# Generates source images for Figure 1, panel B (example input sequences).
# Output is saved directly to src/src_imgs/RawImages.png (whitespace-trimmed
# via savefig bbox_inches='tight') for use in run_figure_overview.py.
import pickle
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import sys
from src.utils.imgTools import load_and_normNIFTI
from src.utils.config_loader import load_config
import matplotlib.image as mpimg




def load_and_process_data(pkl_file, lfm_path, seed=777):

    """Load and process the data for odor, taste, and combination conditions"""

    # Filter and group recordings
    rec_names = list(pkl_file.keys())
    filtered = [r for r in rec_names if not r.split('_')[0][-1].endswith('C')]
    
    # Group by condition
    conditions = {
        'Odor': [r for r in filtered if r[1] == 'O'],
        'Taste': [r for r in filtered if r[1] == 'T'],
        'Combi_M': [r for r in filtered if r[1:3] == 'MM'],
        'Combi_C': [r for r in filtered if r[1:3] == 'MC']
    }
    
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Select random recordings
    selected = {cond: random.choice(recs) for cond, recs in conditions.items()}
    print('selected recordings: ', selected)
    
    # Load and process data
    data = {}
    for cond, rec in selected.items():
        # Load data
        rec_path = f'{lfm_path}/{rec}.nii'
        rec_data = load_and_normNIFTI(
            rec_path,
            substract_Baseline=True,
            t_base=[0, 200],
            save_baseframe=False,
            knockOutNeuropil=False,
            neuropil_name='AL'
        )
        print(f'Loaded{rec_path}.')
        
        # Process slices
        meanZ = np.mean(rec_data, axis=2)
        
        data[cond] = {
            'recording': rec,
            'data': rec_data,
            'meanZ': meanZ
        }
    
    return data

def get_largest_gap(pickle_file, recording_name):
    """Calculate the largest gap in the recording"""
    stim_times = sorted(pickle_file[recording_name])
    gaps = [stim_times[i+1] - stim_times[i] for i in range(len(stim_times)-1)]
    max_gap_idx = gaps.index(max(gaps))
    start_gap = stim_times[max_gap_idx]
    end_gap = stim_times[max_gap_idx + 1]
    return [start_gap, end_gap]

def get_time_points(pkl, data):
    """Calculate time points for each condition"""
    time_points = {}
    for cond in list(data.keys()):
        tS = get_largest_gap(pkl, data[cond]['recording'])[-1]
        time_points[cond] = [tS - 20, tS, tS+10, tS+20, tS+30, tS+40]
    return time_points



def plot_meanZ(data, time_points, output_path, bg_path):
    """
    Plots a figure with as many rows as data.keys(), each row has a background PNG
    with 3 images overlaid (side by side).
    """
    n_rows = len(data.keys())

    # Load background image (PNG)
    bg_img = mpimg.imread(bg_path)

    fig, axes = plt.subplots(n_rows, 1, figsize=(10, 2 * n_rows))  # one axis per row
    if n_rows == 1:
        axes = [axes]

    for row_idx, key in enumerate(data.keys()):
        meanZ_data = data[key]['meanZ']
        images = [
            meanZ_data[:, :, time_points[key][0]],
            meanZ_data[:, :, time_points[key][1]],
            meanZ_data[:, :, time_points[key][2]],
            meanZ_data[:, :, time_points[key][3]],
            meanZ_data[:, :, time_points[key][4]],
            meanZ_data[:, :, time_points[key][5]],
        ]

        ax = axes[row_idx]

        # Plot the background over the full width (x=0 to 3, y=0 to 1)
        ax.imshow(bg_img, extent=[0, 6, 0, 1], zorder=0)

        # [xmin, xmax, ymin, ymax]
        overlays_extents = [
            [0.11, 0.81, 0.145, 0.82],
            [1.48, 2.18, 0.142, 0.82],
            [2.4, 3.1, 0.142, 0.82],
            [3.3, 4.0, 0.142, 0.82],
            [4.2, 4.9, 0.142, 0.82],
            [5.15, 5.85, 0.142, 0.82],   
        ]




        # Overlay each image in its respective "column"
        for i, img in enumerate(images):
            ax.imshow(img, extent=overlays_extents[i], alpha=0.8, cmap='magma', zorder=1)

        ax.set_xlim(0, 6)
        ax.set_ylim(0, 1)
        ax.axis('off')

    plt.text(0.3, 4.8, f"Pre-stimulus", ha='left', va='bottom', fontsize=10)
    plt.text(1.5, 4.8, f"image sequence (dt = 1 s)", ha='left', va='bottom', fontsize=10)


    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)

def main():
    """Main function to run the visualization"""
    # Configuration
    config = load_config()
    paths = config['paths']

    LFM_PATH = os.path.join(paths['data_root'], 'Paul_LFM_Data')
    PKL_FILE = paths['peakIDs_Times_All']
    OUTPUT_PATH = os.path.join(paths['src_imgs_dir'], 'RawImages.png')
    bg_path = os.path.join(paths['src_imgs_dir'], 'RawImages_blank_B.png')

    # Load pickle file
    with open(PKL_FILE, 'rb') as f:
        pkl = pickle.load(f)
    
    # Load and process data
    data = load_and_process_data(pkl, LFM_PATH)
    
    # Get time points
    time_points = get_time_points(pkl, data)
    
    # Create plots
    # plot_brain_stacks(data, time_points, OUTPUT_PATH)
    
    # Save additional plot
    plot_meanZ(data, time_points, OUTPUT_PATH, bg_path)


if __name__ == "__main__":
    main()
