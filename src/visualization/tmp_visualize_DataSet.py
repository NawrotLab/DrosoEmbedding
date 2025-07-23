import pickle
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import sys
from src.utils.imgTools import load_and_normNIFTI


def load_and_process_data(pkl_file, lfm_path, seed=777):
    """Load and process the data for odor, taste, and combination conditions"""

    # Filter and group recordings
    rec_names = list(pkl_file.keys())
    filtered = [r for r in rec_names if not r.split('_')[0][-1].endswith('C')]
    
    # Group by condition
    conditions = {
        'Odor': [r for r in filtered if r[1] == 'O'],
        'Taste': [r for r in filtered if r[1] == 'T'],
        'Combi': [r for r in filtered if r[1] == 'M']
    }
    
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Select random recordings
    selected = {cond: random.choice(recs) for cond, recs in conditions.items()}
    
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
        anterior = np.mean(rec_data[:, :, 0:16, :], axis=2)
        posterior = np.mean(rec_data[:, :, 16:, :], axis=2)
        
        data[cond] = {
            'recording': rec,
            'data': rec_data,
            'slices': [meanZ, anterior, posterior]
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
    for cond in ['Odor', 'Taste', 'Combi']:
        tS = get_largest_gap(pkl, data[cond]['recording'])[-1]
        time_points[cond] = [tS - 15, tS-5, tS+5]
    return time_points

def plot_stack(ax, img, cmap='viridis', alpha=0.7):
    """Plot a brain slice image"""
    ax.imshow(img, cmap=cmap, alpha=alpha)
    ax.axis('off')

def plot_brain_stacks(data, time_points, output_path):
    """Create a 3x3 grid plot showing mean slices at different time points"""
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    
    # Plot each condition's slices at different time points
    for i, cond in enumerate(['Odor', 'Taste', 'Combi']):
        # Get the data and time points for this condition
        data_array = data[cond]['data']
        times = time_points[cond]  # [tS-20, tS-10, tS]
        print('Plotting: ', cond, ' at times: ', times)
        # For each time point
        for j, t in enumerate(times):
            # Get the slice at this time point
            anterior = np.mean(data_array[:, :, 0:11, t], axis=2)
            middle = np.mean(data_array[:, :, 11:22, t], axis=2)
            posterior = np.mean(data_array[:, :, 22:, t], axis=2)
            
            # Plot the slices
            plot_stack(axes[0, j], anterior)
            plot_stack(axes[1, j], middle)
            plot_stack(axes[2, j], posterior)
            
            # Add labels
            if j == 0:
                # Add condition labels on left side
                axes[0, j].set_ylabel('Anterior', fontsize=12)
                axes[1, j].set_ylabel('Middle', fontsize=12)
                axes[2, j].set_ylabel('Posterior', fontsize=12)
            
            # Add time point labels on top row
            if i == 0:
                if j == 0:
                    axes[i, j].set_title('tS-15', fontsize=12)
                elif j == 1:
                    axes[i, j].set_title('tS-5', fontsize=12)
                else:
                    axes[i, j].set_title('tS+5', fontsize=12)
            
            # Add condition label on top row
            if j == 0:
                axes[i, j].text(-0.2, 0.5, cond, 
                              transform=axes[i, j].transAxes,
                              fontsize=12, rotation=90)
    
    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = plt.colorbar(axes[0, 0].images[0], cax=cbar_ax)
    cbar.set_label('Activity', rotation=270, labelpad=15)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()



def plot_data_stacks(
    data, time_points, output_path,
    stack_shift=3,  # pixels to shift background images
    stack_alpha=0.3, main_alpha=1.0,
    figsize=(12, 6), border=1, gamma=1.6
):
    """
    Plot stacks of images for each key and time point, as in the schematic.
    Background slices (1 and 2) are faint and shifted up/right; all images have a white border.
    Uses per-row (per-condition) normalization.
    """
    keys = list(data.keys())
    n_rows = len(keys)
    n_cols = max(len(time_points[k]) for k in keys)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)

    for row, key in enumerate(keys):
        # --- Compute row (condition) min and max ---
        row_min, row_max = np.inf, -np.inf
        for arr in data[key]['slices']:
            row_min = min(row_min, np.nanmin(arr))
            row_max = max(row_max, np.nanmax(arr))

        def normalize_row(img):
            if row_max > row_min:
                return (img - row_min) / (row_max - row_min)
            else:
                return np.zeros_like(img)

        gamma = gamma  # values < 1 brighten, values > 1 darken
       
        def adjust_gamma(img, gamma=gamma):
            return np.clip(img, 0, 1) ** (1/gamma)

        times = time_points[key]
        for col, t in enumerate(times):
            ax = axes[row, col]
            slices = data[key]['slices']
            # Main image
            base_img = normalize_row(slices[0][:,:,t])
            base_img = adjust_gamma(base_img)
            h, w = base_img.shape
            # Create canvas large enough for all shifts and border
            canvas = np.ones((h + stack_shift*2 + 2*border, w + stack_shift*2 + 2*border, 3))
            # Stack background slices
            for i, s in enumerate([2, 1]):
                img = normalize_row(slices[s][:,:,t])
                bordered = np.pad(img, border, mode='constant', constant_values=1)
                canvas[
                    stack_shift*(2-i):stack_shift*(2-i)+h+2*border,
                    stack_shift*(2-i):stack_shift*(2-i)+w+2*border,
                    :
                ] *= bordered[..., None] * stack_alpha + (1-stack_alpha)
            # Main slice (on top, centered)
            bordered_main = np.pad(base_img, border, mode='constant', constant_values=1)
            canvas[
                stack_shift:stack_shift+h+2*border,
                stack_shift:stack_shift+w+2*border,
                :
            ] = bordered_main[..., None]
            ax.imshow(canvas.squeeze(), cmap='viridis')
            ax.axis('off')
            if row == 0:
                ax.set_title(f"t={t}", fontsize=14)
        axes[row,0].text(-0.2, 0.5, key, va='center', ha='right', rotation=90, fontsize=18, transform=axes[row,0].transAxes)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()






def main():
    """Main function to run the visualization"""
    # Configuration
    LFM_PATH = '/projects/lab-data/Collaboration/Gruenwald_Kadow/Paul_LFM_Data'
    PKL_FILE = 'pickles/IDs_logTs.pickle'
    OUTPUT_PATH = 'src/src_imgs/brain_stacks.png'

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
    plot_data_stacks(data, time_points, OUTPUT_PATH)
    print('sth')


if __name__ == "__main__":
    main()