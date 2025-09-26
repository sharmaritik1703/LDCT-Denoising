import torch
import matplotlib.pyplot as plt
import random
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Function to load a model checkpoint
def load_model(checkpoint_path, model):
    """
    Loads the model weights from a checkpoint file.
    
    Args:
        checkpoint_path (str): Path to the checkpoint file.
        model (torch.nn.Module): The model instance to load the weights into.

    Returns:
        torch.nn.Module: The model with loaded weights.
    """
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return model


# Evaluate the model on the test set
def evaluate_on_test_set(model, test_loader, scale=0.25):
    """
    Evaluates the model on the test set and computes SSIM and PSNR.

    Args:
        model (torch.nn.Module): The trained model.
        test_loader (DataLoader): DataLoader for the test set.

    Prints:
        Average test loss, SSIM, and PSNR.
    """
    device = 'cuda'
    model = model.to(device)
    model.eval()

    # Initialize torchmetrics for GPU-based SSIM and PSNR calculations
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)

    test_loss = 0.0
    test_ssim = 0.0
    test_psnr = 0.0
    criterion = torch.nn.L1Loss()  # Using L1 loss

    with torch.no_grad():
        for inputs, targets in test_loader:
            # Move data to the same device as the model
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)

            l1_loss = criterion(outputs, targets) 
            ssim_loss = 1 - ssim_metric(outputs, targets)  # SSIM loss is 1 - SSIM

            loss = l1_loss + scale * ssim_loss  # Combine L1 loss and SSIM loss
            test_loss += loss.item() * inputs.size(0)

            # Calculate SSIM and PSNR on GPU using torchmetrics
            test_ssim += ssim_metric(outputs, targets).item() * inputs.size(0)
            test_psnr += psnr_metric(outputs, targets).item() * inputs.size(0)

    # Compute average test loss, SSIM, and PSNR
    avg_test_loss = test_loss / len(test_loader.dataset)
    avg_test_ssim = test_ssim / len(test_loader.dataset)
    avg_test_psnr = test_psnr / len(test_loader.dataset)
    print(f"Test Loss: {avg_test_loss:.4f} | Test SSIM: {avg_test_ssim:.4f} | Test PSNR: {avg_test_psnr:.4f}")



def visualize_predictions(model, data_loader, num_samples=9):
    """
    Visualizes the input, output, and ground truth images produced by the trained network.

    Args:
        model (torch.nn.Module): The trained model.
        data_loader (DataLoader): DataLoader for the dataset (can be test or validation set).
        num_samples (int): The number of samples to visualize.

    Returns:
        None
    """
    model.eval()  # Set the model to evaluation mode
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    # Get a batch of data
    inputs, targets = next(iter(data_loader))
    inputs, targets = inputs.to(device), targets.to(device)

    # Forward pass to get the outputs
    with torch.no_grad():
        outputs = model(inputs)

    # Move data to CPU for visualization
    inputs = inputs.detach().cpu().numpy()
    outputs = outputs.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()

    # Randomly select indices for visualization
    indices = random.sample(range(inputs.shape[0]), min(num_samples, inputs.shape[0]))

    # Plot the results
    plt.figure(figsize=(18, 12))  # Increased figure size for larger subplots
    for idx, i in enumerate(indices):
        input_img = inputs[i].transpose(1, 2, 0)  # Convert from (C, H, W) to (H, W, C)
        output_img = outputs[i].transpose(1, 2, 0)
        target_img = targets[i].transpose(1, 2, 0)

        # Remove single-channel grayscale dimension if present
        if input_img.shape[2] == 1:
            input_img = input_img[:, :, 0]
            output_img = output_img[:, :, 0]
            target_img = target_img[:, :, 0]

        # Input image
        plt.subplot(num_samples, 3, 3 * idx + 1)
        plt.imshow(input_img, cmap='gray' if len(input_img.shape) == 2 else None)
        plt.title('Input')
        plt.axis('off')

        # Output image
        plt.subplot(num_samples, 3, 3 * idx + 2)
        plt.imshow(output_img, cmap='gray' if len(output_img.shape) == 2 else None)
        plt.title('Output')
        plt.axis('off')

        # Target (Ground Truth) image
        plt.subplot(num_samples, 3, 3 * idx + 3)
        plt.imshow(target_img, cmap='gray' if len(target_img.shape) == 2 else None)
        plt.title('Target (Ground Truth)')
        plt.axis('off')

    plt.tight_layout()
    plt.show()
