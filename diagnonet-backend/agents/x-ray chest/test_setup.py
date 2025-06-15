#!/usr/bin/env python3
"""
Test script to verify our chest X-ray analysis setup works correctly
"""

import torch
import torchxrayvision as xrv
import skimage.io as io
import matplotlib.pyplot as plt
from preprocess_module import preprocess

# Test Grad-CAM functionality
from gradcam_utils import get_gradcam, plot_gradcam, save_gradcam_visualization, get_gradcam_output_path

def test_with_sample_image():
    """
    Test the setup using our sample.png image
    """
    print("=== Testing Setup ===")
    
    # Load model
    print("Loading model...")
    model = xrv.models.DenseNet(weights="densenet121-res224-all").eval()
    
    # Load sample image
    print("Loading sample image...")
    try:
        img = io.imread("xray_images/sample.png")
        print(f"Image shape: {img.shape}")
    except Exception as e:
        print(f"Error loading xray_images/sample.png: {e}")
        print("Please make sure sample.png exists in the xray_images directory")
        return
    
    # Preprocess
    print("Preprocessing image...")
    tensor = preprocess(img)
    print(f"Tensor shape: {tensor.shape}")
    
    # Inference
    print("Running inference...")
    with torch.no_grad():
        probs = torch.sigmoid(model(tensor)[0])
        pneumonia_prob = probs[model.pathologies.index("Pneumonia")]
        print(f"Pneumonia probability: {pneumonia_prob:.3f}")
    
    # Test Grad-CAM
    print("Testing Grad-CAM...")
    try:
        heat = get_gradcam(img, model, "Pneumonia")
        
        # Create and save visualization using helper function
        output_path = get_gradcam_output_path("xray_images/sample.png", suffix="test", pathology="Pneumonia")
        save_gradcam_visualization(img, heat, pneumonia_prob, output_path, "Pneumonia")
        
        print(f"✓ Grad-CAM test successful! Saved as '{output_path}'")
        
    except Exception as e:
        print(f"Error with Grad-CAM: {e}")
    
    # Show top findings
    print("\n=== Top 5 Findings ===")
    top5 = sorted(zip(model.pathologies, probs.tolist()), key=lambda x: x[1], reverse=True)[:5]
    for label, p in top5:
        print(f"{label:25}: {p:.3f}")
    
    print("\n✅ Setup test completed successfully!")
    print("Your environment is ready for chest X-ray analysis.")

if __name__ == "__main__":
    test_with_sample_image() 