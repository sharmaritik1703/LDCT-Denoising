import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def load_images(directory_path):
    """
    Load all .npy images from the given directory in sequence (e.g., img1_, img2_, etc.)
    and reshape them to (B, H, W) by removing the channel dimension (C=1).
    
    Args:
        directory_path (str): The path to the directory containing .npy images.
    
    Returns:
        np.ndarray: A numpy array containing all the images loaded in sequence with shape (B, H, W).
    """
    # Get a list of all .npy files in the directory
    npy_files = [f for f in os.listdir(directory_path) if f.endswith('.npy')]
    # Sort files based on the sequence
    npy_files.sort()
    
    # Load all images into a numpy array
    images = [np.load(os.path.join(directory_path, file)) for file in npy_files]
    images = np.array(images)  # Shape: (B, C, H, W)
    
    # Remove the channel dimension (C=1) to get (B, H, W)
    if images.shape[1] == 1:  # Ensure channel dimension exists before squeezing
        images = images.squeeze(axis=1)
    
    return images
    

def visualize_pairs(low_dose_images, high_dose_images, dataset_name='Train', n_images=8):
    """
    Visualizes pairs of low-dose and high-dose images, randomly selecting n_images.
    Assumes both are numpy arrays or tensors of shape [N, H, W]
    """
    # Ensure the number of images does not exceed the dataset size
    n_images = min(n_images, len(low_dose_images), len(high_dose_images))
    
    # Randomly select indices without replacement
    random_indices = np.random.choice(len(low_dose_images), n_images, replace=False)
    
    # Select random images
    selected_low = low_dose_images[random_indices]
    selected_high = high_dose_images[random_indices]

    fig, axes = plt.subplots(2, n_images, figsize=(2 * n_images, 4))
    fig.suptitle(f'{dataset_name} Set: Low-dose vs High-dose', fontsize=16)

    for i, idx in enumerate(random_indices):
        axes[0, i].imshow(selected_low[i], cmap='gray')
        axes[0, i].axis('off')
        axes[0, i].set_title(f'Low {idx+1}', fontsize=10)

        axes[1, i].imshow(selected_high[i], cmap='gray')
        axes[1, i].axis('off')
        axes[1, i].set_title(f'High {idx+1}', fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


class PairedDataset(Dataset):
    def __init__(self, ld_array, hd_array, normalize=True):
        """
        ld_image, hd_image: numpy arrays of shape (N, H, W), will be resized to (N, target_size, target_size)
        """
        assert len(ld_array) == len(hd_array), "Low-dose and HD image arrays must have the same number of samples"
        self.ld_array  = ld_array
        self.hd_array  = hd_array
        self.normalize = normalize

    def __len__(self):
        return len(self.ld_array)

    def minmax_normalize(self, img):
        img_min = torch.min(img)
        img_max = torch.max(img)
        return (img - img_min) / (img_max - img_min + 1e-3)

    def __getitem__(self, idx):
        # print(f"Loading sample {idx}")
        ld_np = self.ld_array[idx]
        hd_np = self.hd_array[idx]

        # ld = torch.tensor(ld_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
        ld = torch.tensor(ld_np.astype(np.float32), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        hd = torch.tensor(hd_np.astype(np.float32), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        # hd = torch.tensor(hd_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Add batch and channel dims

        # Remove batch dimension, keep channel dimension
        ld = ld.squeeze(0)  # Shape: (1, 256, 256)
        hd = hd.squeeze(0)  # Shape: (1, 256, 256)

        if self.normalize:
            ld = self.minmax_normalize(ld)
            hd = self.minmax_normalize(hd)

        return ld, hd


def visualize_from_dataloader(dataloader, dataset_name='DataLoader', n_batches=2):
    """
    Visualizes images from a dataloader by randomly selecting n_batches.
    Assumes dataloader has batch_size=4, so n_batches=2 will show 8 images total.
    
    Args:
        dataloader: PyTorch DataLoader object
        dataset_name (str): Name for the plot title
        n_batches (int): Number of batches to visualize (default=2 for 8 images)
    """
    # Convert dataloader to list for random sampling
    all_batches = list(dataloader)
    
    if len(all_batches) < n_batches:
        print(f"Warning: Only {len(all_batches)} batches available, showing all.")
        n_batches = len(all_batches)
    
    # Randomly select batches
    random_batch_indices = np.random.choice(len(all_batches), n_batches, replace=False)
    
    # Collect images from selected batches
    all_ld_images = []
    all_hd_images = []
    
    for batch_idx in random_batch_indices:
        ld_batch, hd_batch = all_batches[batch_idx]
        # Convert to numpy and remove channel dimension for visualization
        ld_batch_np = ld_batch.squeeze(1).numpy()  # Shape: (batch_size, H, W)
        hd_batch_np = hd_batch.squeeze(1).numpy()  # Shape: (batch_size, H, W)
        
        all_ld_images.extend(ld_batch_np)
        all_hd_images.extend(hd_batch_np)
    
    # Convert to numpy arrays
    all_ld_images = np.array(all_ld_images)
    all_hd_images = np.array(all_hd_images)
    
    n_images = len(all_ld_images)
    
    fig, axes = plt.subplots(2, n_images, figsize=(2 * n_images, 4))
    fig.suptitle(f'{dataset_name}: Low-dose vs High-dose ({n_images} images)', fontsize=16)
    
    for i in range(n_images):
        axes[0, i].imshow(all_ld_images[i], cmap='gray')
        axes[0, i].axis('off')
        axes[0, i].set_title(f'Low {i+1}', fontsize=10)
        
        axes[1, i].imshow(all_hd_images[i], cmap='gray')
        axes[1, i].axis('off')
        axes[1, i].set_title(f'High {i+1}', fontsize=10)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

