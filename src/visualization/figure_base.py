"""
Base figure utilities for shared functionality across figure generation scripts.

This module provides common utilities for setting up figures, managing layouts,
and handling shared configuration across different figure types.
"""

import os
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from typing import Dict, List, Tuple, Optional, Any
import yaml


class FigureConfig:
    """Configuration constants for figure generation."""
    
    # Base results directory
    BASE_RESULTS_DIR = os.path.join('results', '_chkpt_finals')
    
    # Target journal page width — change here to retarget all figures at once
    # Full-page / double-column width: Nature, eLife, PNAS all land at 6.9–7.1"
    PAGE_WIDTH = 7.0  # inches

    # Font sizes — single source of truth for all publication figures
    FONT_SIZES = {
        'panel_label':   12,   # a, b, c… panel letters (bold)
        'title':         14,   # structural row/column headers
        'subplot_title': 11,   # individual subplot titles
        'label':         11,   # xlabel / ylabel
        'tick':          10,   # xtick / ytick labels
        'legend':        11,   # legend text
        'annotation':    10,   # in-plot text, chance-level labels, accuracy numbers
        'colorbar':      11,   # colorbar tick labels and title
        'heatmap_cell':  10,   # text inside heatmap cells
    }
    
    # Spacing
    SPACING = {
        'hspace': 0.1,
        'wspace': 0.15,
        'title_pad': 0.05
    }
    
    # Task configurations
    TASK_CONFIG = {
        'MetabolicState_2': {
            'name': 'i. State',
            'chance_level': 50.0,
            'baseline': 54.1,
            'y_range': (70, 100)
        },
        'State_Modality_6': {
            'name': 'ii. State, Modality',
            'chance_level': 100/6,
            'baseline': 27.8,
            'y_range': (70, 100)
        },
        'State_Modality_Valence_16': {
            'name': 'iii. State, Modality, Valence',
            'chance_level': 100/16,
            'baseline': 10.0,
            'y_range': (60, 90)
        }
    }


class FigureBase:
    """Base class for figure generation with common functionality."""
    
    def __init__(self, style_file: str = "styles", base_results_dir: Optional[str] = None):
        """
        Initialize the figure base.
        
        Args:
            style_file: Name of the style file to load (without .yaml extension)
            base_results_dir: Override the default results directory
        """
        self.style_file = style_file
        self.base_results_dir = base_results_dir or FigureConfig.BASE_RESULTS_DIR
        self.styles = self._load_styles()
        self._setup_matplotlib()
    
    def _load_styles(self) -> Dict[str, Any]:
        """Load styles from YAML file."""
        styles_path = os.path.join('src', 'visualization', f'{self.style_file}.yaml')
        try:
            with open(styles_path, 'r') as f:
                return yaml.safe_load(f)['styles']
        except FileNotFoundError:
            print(f"Warning: Style file {styles_path} not found. Using default styles.")
            return {}
    
    def _setup_matplotlib(self):
        """Set up common matplotlib parameters."""
        apply_style()
    
    def create_figure(self, figsize: Tuple[float, float], **kwargs) -> plt.Figure:
        """Create a new figure with standard settings."""
        return plt.figure(figsize=figsize, **kwargs)
    
    def save_figure(self, fig: plt.Figure, out_path: str, **kwargs):
        """
        Save figure with standard settings.
        
        Args:
            fig: Matplotlib figure to save
            out_path: Output path for the figure
            **kwargs: Additional arguments for plt.savefig
        """
        # Set default save parameters
        save_kwargs = {
            'dpi': 300,
            'bbox_inches': 'tight',
            'pad_inches': 0.1
        }
        save_kwargs.update(kwargs)
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        
        # Save the figure
        fig.savefig(out_path, **save_kwargs)
        plt.close(fig)
        print(f"Figure saved to {out_path}")
    
    def add_column_titles(self, fig: plt.Figure, titles: List[str], 
                         positions: Optional[List[float]] = None, 
                         y_pos: float = 0.93):
        """
        Add column titles to a figure.
        
        Args:
            fig: Matplotlib figure
            titles: List of title strings
            positions: X positions for titles (if None, evenly spaced)
            y_pos: Y position for titles
        """
        if positions is None:
            # Evenly space titles
            n_titles = len(titles)
            positions = [i / (n_titles - 1) for i in range(n_titles)] if n_titles > 1 else [0.5]
        
        for pos, title in zip(positions, titles):
            fig.text(pos, y_pos, title, ha='center', va='center', 
                    fontsize=FigureConfig.FONT_SIZES['title'], 
                    weight='bold', transform=fig.transFigure)
    
    def add_row_titles(self, fig: plt.Figure, titles: List[str], 
                      positions: Optional[List[float]] = None,
                      x_pos: float = 0.05):
        """
        Add row titles to a figure.
        
        Args:
            fig: Matplotlib figure
            titles: List of title strings
            positions: Y positions for titles (if None, evenly spaced)
            x_pos: X position for titles
        """
        if positions is None:
            # Evenly space titles
            n_titles = len(titles)
            positions = [i / (n_titles - 1) for i in range(n_titles)] if n_titles > 1 else [0.5]
        
        for pos, title in zip(positions, titles):
            fig.text(x_pos, pos, title, ha='left', va='center',
                    fontsize=FigureConfig.FONT_SIZES['title'],
                    weight='bold', rotation=90, transform=fig.transFigure)
    
    def create_colorbar(self, fig: plt.Figure, cmap: str = 'Blues', 
                       vmin: float = 0, vmax: float = 100,
                       label: str = 'Prediction Percentage',
                       position: Tuple[float, float, float, float] = (0.1, 0.88, 0.49, 0.02)):
        """
        Create a horizontal colorbar for the figure.
        
        Args:
            fig: Matplotlib figure
            cmap: Colormap name
            vmin: Minimum value for colormap
            vmax: Maximum value for colormap
            label: Label for the colorbar
            position: (left, bottom, width, height) in figure coordinates
        """
        # Create colorbar axes
        cbar_ax = fig.add_axes(position)
        
        # Create colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        cbar = plt.colorbar(sm, cax=cbar_ax, orientation='horizontal')
        cbar.set_label(label, labelpad=10, fontsize=FigureConfig.FONT_SIZES['label'])
        cbar.ax.xaxis.set_label_position('top')
        cbar.set_ticks([0, 25, 50, 75, 100])
        
        return cbar


def load_styles(style_file: str = "styles") -> Dict[str, Any]:
    """
    Load styles from YAML file.
    
    Args:
        style_file: Name of the style file (without .yaml extension)
        
    Returns:
        Dictionary of styles
    """
    styles_path = os.path.join('src', 'visualization', f'{style_file}.yaml')
    try:
        with open(styles_path, 'r') as f:
            return yaml.safe_load(f)['styles']
    except FileNotFoundError:
        print(f"Warning: Style file {styles_path} not found. Using empty styles.")
        return {}


def get_task_config(task_name: str) -> Dict[str, Any]:
    """
    Get configuration for a specific task.
    
    Args:
        task_name: Name of the task
        
    Returns:
        Dictionary with task configuration
    """
    return FigureConfig.TASK_CONFIG.get(task_name, {
        'name': task_name,
        'chance_level': 0.0,
        'y_range': (60, 100)
    })


# Module-level aliases for direct import
FONT_SIZES = FigureConfig.FONT_SIZES
PAGE_WIDTH = FigureConfig.PAGE_WIDTH


def apply_style():
    """Call once at the top of each figure script before any plotting."""
    plt.rcParams.update({
        'font.family':     'Arial',
        'font.size':        FigureConfig.FONT_SIZES['annotation'],
        'axes.titlesize':   FigureConfig.FONT_SIZES['subplot_title'],
        'axes.labelsize':   FigureConfig.FONT_SIZES['label'],
        'xtick.labelsize':  FigureConfig.FONT_SIZES['tick'],
        'ytick.labelsize':  FigureConfig.FONT_SIZES['tick'],
        'legend.fontsize':  FigureConfig.FONT_SIZES['legend'],
        'figure.titlesize': FigureConfig.FONT_SIZES['title'],
    })
