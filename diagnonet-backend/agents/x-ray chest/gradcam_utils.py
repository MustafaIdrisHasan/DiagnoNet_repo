import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
    
    def generate_cam(self, input_tensor, class_idx):
        # Forward pass
        model_output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = torch.argmax(model_output, dim=1).item()
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass
        class_score = model_output[:, class_idx]
        class_score.backward(retain_graph=True)
        
        # Generate CAM
        gradients = self.gradients[0]  # [C, H, W]
        activations = self.activations[0]  # [C, H, W]
        
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(1, 2))  # [C]
        
        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # Apply ReLU
        cam = F.relu(cam)
        
        # Normalize
        cam = cam / torch.max(cam)
        
        return cam.detach().cpu().numpy()

def get_gradcam(img, model, pathology_name):
    """
    Generate Grad-CAM heatmap for a given image and pathology
    
    Args:
        img: Input image (numpy array)
        model: TorchXRayVision model
        pathology_name: Name of the pathology (e.g., "Pneumonia")
    
    Returns:
        numpy array: Grad-CAM heatmap
    """
    # Find the target layer (last convolutional layer)
    target_layer = None
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            target_layer = module
    
    if target_layer is None:
        raise ValueError("Could not find a convolutional layer in the model")
    
    # Create GradCAM instance
    grad_cam = GradCAM(model, target_layer)
    
    # Preprocess image
    from preprocess_module import preprocess
    input_tensor = preprocess(img)
    
    # Get class index
    class_idx = model.pathologies.index(pathology_name)
    
    # Generate CAM
    cam = grad_cam.generate_cam(input_tensor, class_idx)
    
    # Resize to original image size
    cam_resized = cv2.resize(cam, (img.shape[1], img.shape[0]))
    
    return cam_resized

def plot_gradcam(img, heatmap, alpha=0.4):
    """
    Plot Grad-CAM overlay on image
    
    Args:
        img: Original image
        heatmap: Grad-CAM heatmap
        alpha: Transparency of overlay
    """
    # Ensure image is grayscale for display
    if len(img.shape) == 3:
        img_display = img.mean(axis=2)
    else:
        img_display = img
    
    # Normalize image for display
    img_display = (img_display - img_display.min()) / (img_display.max() - img_display.min())
    
    # Create a custom colormap for the heatmap
    colors = ['blue', 'cyan', 'yellow', 'red']
    n_bins = 256
    cmap = LinearSegmentedColormap.from_list('gradcam', colors, N=n_bins)
    
    # Display the image
    plt.imshow(img_display, cmap='gray')
    
    # Overlay the heatmap
    plt.imshow(heatmap, cmap=cmap, alpha=alpha)
    
    plt.axis('off')

def save_gradcam_visualization(img, heatmap, prediction_prob, filename, pathology_name="Pneumonia"):
    """
    Save a complete Grad-CAM visualization with original image, heatmap, and overlay
    
    Args:
        img: Original image
        heatmap: Grad-CAM heatmap
        prediction_prob: Model prediction probability
        filename: Output filename
        pathology_name: Name of the pathology
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    if len(img.shape) == 3:
        img_display = img.mean(axis=2)
    else:
        img_display = img
    
    axes[0].imshow(img_display, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Heatmap only
    axes[1].imshow(heatmap, cmap='hot')
    axes[1].set_title('Grad-CAM Heatmap')
    axes[1].axis('off')
    
    # Overlay
    axes[2].imshow(img_display, cmap='gray')
    axes[2].imshow(heatmap, cmap='hot', alpha=0.5)
    axes[2].set_title(f'{pathology_name} Overlay\nProb: {prediction_prob:.3f}')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    plt.close()
    
    return filename

def ensure_gradcam_dir():
    """
    Ensure the gradcam_visualizations directory exists
    """
    if not os.path.exists("gradcam_visualizations"):
        os.makedirs("gradcam_visualizations")
        print("📁 Created gradcam_visualizations directory")

def get_gradcam_output_path(input_path, suffix="", pathology=""):
    """
    Generate a standardized output path for Grad-CAM visualizations
    
    Args:
        input_path: Path to the original image
        suffix: Additional suffix for the filename
        pathology: Name of the pathology being visualized
    
    Returns:
        str: Full path to save the Grad-CAM visualization
    """
    ensure_gradcam_dir()
    
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    
    if pathology:
        filename = f"{base_name}_gradcam_{pathology}"
    else:
        filename = f"{base_name}_gradcam"
    
    if suffix:
        filename += f"_{suffix}"
    
    filename += ".png"
    
    return os.path.join("gradcam_visualizations", filename) 