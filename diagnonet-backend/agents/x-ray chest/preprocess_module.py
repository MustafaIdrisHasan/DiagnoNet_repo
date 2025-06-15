import torch
import torchvision.transforms as T
import torchxrayvision as xrv
import numpy as np

def preprocess(img):
    """
    Preprocess an image for TorchXRayVision model inference.
    
    Args:
        img: Input image array (can be RGB or grayscale)
        
    Returns:
        torch.Tensor: Preprocessed tensor ready for model inference
    """
    # Normalize the image
    img = xrv.datasets.normalize(img, 255)
    
    # Convert RGB to grayscale if needed
    if img.ndim == 3:
        img = img.mean(2)
    
    # Apply transformations
    tfm = T.Compose([
        xrv.datasets.XRayCenterCrop(),
        xrv.datasets.XRayResizer(224)
    ])
    
    # Apply transforms and convert to tensor
    img_transformed = tfm(img[None, ...])
    return torch.from_numpy(img_transformed).unsqueeze(0).float() 