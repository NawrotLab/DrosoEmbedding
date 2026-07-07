import os
import torch
from torch.utils.data import Dataset
import numpy as np
from scipy.interpolate import interp1d
import tifffile
from torchvision import transforms
from torchvision.transforms import functional as TF

class CustomDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None, 
                 resize_img=(128, 128), seq_length=1, seq_steps=1, 
                 allTs_path=None, augment=False, num_augmentations=5, 
                 cache_images=False, preload_to_ram=False):
        self.image_paths = image_paths
        self.labels = labels
        self.img_size = resize_img
        self.seq_length = seq_length
        self.seq_steps = seq_steps
        self.directory_path = allTs_path
        self.augment = augment
        self.num_augmentations = num_augmentations if augment else 0
        self.cache_images = cache_images

        # Base transform for converting to tensor and resizing
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(resize_img)
        ]) if transform else None

        if self.seq_length > 1 and not self.directory_path:
            raise ValueError("directory_path must be provided if seq_length > 1")

        # Filter valid sequences
        self.image_paths, self.labels = self._filter_valid_sequences()

        # Preload all data to RAM if requested
        self.preloaded_data = None
        if preload_to_ram:
            self.preloaded_data = [
                (self._load_and_preprocess_sequence(idx) if seq_length > 1 else self._load_and_preprocess_image(img_path), label)
                for idx, (img_path, label) in enumerate(zip(self.image_paths, self.labels))
            ]
            # Convert to tensor once
            self.preloaded_data = [
                (torch.stack(imgs).float() if isinstance(imgs, list) else imgs.float(), torch.tensor(label)) 
                for imgs, label in self.preloaded_data
            ]
        
        # Optional: Cache images to RAM (for single images only)
        self.cached_samples = None
        if cache_images:
            self.cached_samples = [
                (self._load_sequence(idx) if seq_length > 1 else self._load_image(img_path), label) 
                for idx, (img_path, label) in enumerate(zip(self.image_paths, self.labels))
            ]

    def __len__(self):
        return len(self.image_paths) * (1 + self.num_augmentations)

    def __getitem__(self, idx):
        num_original_samples = len(self.image_paths)
        original_sample_idx = idx // (self.num_augmentations + 1)
        augmentation_idx = idx % (self.num_augmentations + 1)

        # If data is preloaded to RAM
        if self.preloaded_data is not None:
            img, label = self.preloaded_data[original_sample_idx]
        # If using cached samples
        elif self.cached_samples is not None:
            img, label = self.cached_samples[original_sample_idx]
        else:
            label = self.labels[original_sample_idx]
            if self.seq_length == 1:
                img = self._load_image(self.image_paths[original_sample_idx])
            else:
                img = self._load_and_preprocess_sequence(original_sample_idx)

        # Apply augmentation if needed
        if augmentation_idx > 0 and self.augment:
            img = self._apply_augmentations(img)

        return img, label

    def _load_and_preprocess_image(self, img_path):
        """Load and preprocess a single image."""
        try:
            img = tifffile.imread(img_path)
            if self.transform:
                img = self.transform(img)
            # Convert to float32 immediately
            img = img.float()
            return img
        except Exception as e:
            raise ValueError(f"Error loading image {img_path}: {e}")

    def _load_and_preprocess_sequence(self, idx):
        """Load and preprocess a sequence of images."""
        img_paths = self._get_sequence_paths(idx)
        if not img_paths:
            # Fallback to a different sequence if this one is invalid
            return self._load_sequence((idx + 1) % len(self.image_paths))
        
        # Load, transform, and preprocess all images in the sequence
        images = []
        for path in img_paths:
            img = tifffile.imread(path)
            if self.transform:
                img = self.transform(img)
            # Convert to float32 immediately
            img = img.float()
            images.append(img)
        return torch.stack(images)  # Always return a tensor

    def _get_sequence_paths(self, idx):
        """Get paths for all images in a sequence."""
        sequence_paths = []
        img_path = self.image_paths[idx]
        root_directory = self.directory_path
        filename = img_path.split('/')[-2]
        try:
            start_idx = int(img_path.split('_')[-1].split('.')[0])  # Extract starting frame index
        except ValueError:
            print(f"Warning: Could not parse frame index from {img_path}")
            return []
        end_idx = start_idx + (self.seq_length - 1) * self.seq_steps + 1

        for i in range(start_idx, end_idx, self.seq_steps):
            file_path = os.path.join(root_directory, filename, f"{filename}_{i}.tiff")
            if os.path.exists(file_path):
                sequence_paths.append(file_path)
            else:
                print(f"Missing file: {file_path}")
                return []
        return sequence_paths if len(sequence_paths) == self.seq_length else []

    def _apply_augmentations(self, img):
        """Apply augmentations to either single image or sequence."""
        if img.ndimension() == 3:  # Single image
            return self._augment_image(img)
        elif img.ndimension() == 4:  # Sequence of images
            return torch.stack([self._augment_image(frame) for frame in img])
        return img

    def _augment_image(self, img):
        """Apply augmentations to a single image."""
        # Add noise
        noise = torch.randn_like(img) * 0.05
        img = torch.clamp(img + noise, 0, 1)

        # Adjust brightness and contrast
        brightness_factor = np.random.uniform(0.9, 1.1)
        contrast_factor = np.random.uniform(0.9, 1.1)
        img = TF.adjust_brightness(img, brightness_factor)
        img = TF.adjust_contrast(img, contrast_factor)

        return img

    def _time_warp(self, img, time_stretch=0.05):
        """Apply time warping to sequence data."""
        if img.ndimension() < 3:
            return img  # Time warp only applies to sequences
        t = np.arange(img.shape[-1])
        stretch = np.random.uniform(1 - time_stretch, 1 + time_stretch)
        t_new = np.linspace(0, len(t) - 1, len(t)) * stretch
        t_new = np.clip(t_new, 0, len(t) - 1)
        interp = interp1d(t, img.cpu().numpy(), kind='linear', axis=-1, fill_value="extrapolate")
        warped_img = interp(t_new)
        return torch.tensor(warped_img).float()

    def _filter_valid_sequences(self):
        """Filter out invalid sequences."""
        valid_paths, valid_labels = [], []
        for idx, img_path in enumerate(self.image_paths):
            if self.seq_length == 1:
                valid_paths.append(img_path)
                valid_labels.append(self.labels[idx])
            else:
                img_paths = self._get_sequence_paths(idx)
                if img_paths:
                    valid_paths.append(img_path)
                    valid_labels.append(self.labels[idx])
        return valid_paths, valid_labels

    def get_sample_shape(self):
        """Get the shape of a single sample."""
        sample, _ = self[0]
        return sample.shape


def compute_mean_frame_shape(paths, n_sample=200, seed=0, logger=None):
    """Return (mean_H, mean_W) of raw (pre-transform) TIFF frames.

    Reads a random subset of *paths* without applying any transform so the
    natural image dimensions are measured.  Use the result to derive an
    aspect-correct upscaling target for GradCAM maps, e.g.
        brain_shape = (128, round(128 * mean_W / mean_H))

    Parameters
    ----------
    paths     : list[str]  – image file paths (e.g. correct_paths from GradCAM)
    n_sample  : int        – how many frames to sample (capped at len(paths))
    seed      : int        – RNG seed for reproducible sampling
    """
    def _log(msg):
        if logger: logger.info(msg)
        else: print(msg)

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(paths), size=min(n_sample, len(paths)), replace=False)
    heights, widths = [], []
    for i in idx:
        img = tifffile.imread(paths[i])
        # tifffile returns (H, W) for grayscale or (H, W, C) for colour
        heights.append(img.shape[0])
        widths.append(img.shape[1])
    mean_H = float(np.mean(heights))
    mean_W = float(np.mean(widths))
    _log(f"compute_mean_frame_shape: sampled {len(idx)} frames → "
         f"mean H={mean_H:.1f}, mean W={mean_W:.1f} "
         f"(aspect ratio {mean_W/mean_H:.3f})")
    return mean_H, mean_W