import glob
import torch
import torchxrayvision as xrv
import skimage.io as io
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from preprocess_module import preprocess

# For Grad-CAM visualization
from gradcam_utils import get_gradcam, save_gradcam_visualization, get_gradcam_output_path

def test_accuracy_with_gradcam():
    """
    Test model accuracy on pediatric chest X-ray dataset and generate Grad-CAM visualizations.
    """
    print("Loading model...")
    model = xrv.models.DenseNet(weights="densenet121-res224-all").eval()
    
    print("Finding test images...")
    files = glob.glob('paed/chest_xray/test/*/*.jpeg')
    
    if not files:
        print("No test images found. Please ensure the dataset is downloaded and extracted to 'paed/' directory.")
        print("Expected structure: paed/chest_xray/test/PNEUMONIA/ and paed/chest_xray/test/NORMAL/")
        return
    
    print(f"Found {len(files)} test images")
    
    y_true, y_pred = [], []
    gradcam_count = 0
    max_gradcam = 10  # Limit Grad-CAM generation to first 10 images to avoid too many files
    
    for i, f in enumerate(files):
        if i % 100 == 0:
            print(f"Processing image {i+1}/{len(files)}")
            
        # Determine true label
        label = 1 if 'PNEUMONIA' in f else 0
        
        # Load and preprocess image
        img = io.imread(f)
        tensor = preprocess(img)
        
        # Inference
        with torch.no_grad():
            p = torch.sigmoid(model(tensor)[0])[model.pathologies.index("Pneumonia")]
        
        y_true.append(label)
        y_pred.append(int(p > 0.5))  # simple 0.5 threshold
        
        # Generate Grad-CAM for first few images
        if gradcam_count < max_gradcam and label == 1:  # Only for pneumonia cases
            print(f"Generating Grad-CAM for {f}")
            try:
                # Get Grad-CAM heatmap
                heat = get_gradcam(img, model, "Pneumonia")
                
                # Save the visualization using helper function
                output_filename = get_gradcam_output_path(f, suffix="accuracy_test", pathology="Pneumonia")
                save_gradcam_visualization(img, heat, p, output_filename, "Pneumonia")
                
                gradcam_count += 1
                print(f"Saved Grad-CAM visualization: {output_filename}")
                
            except Exception as e:
                print(f"Error generating Grad-CAM for {f}: {e}")
    
    # Calculate and print accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\n=== RESULTS ===")
    print(f"Total images processed: {len(files)}")
    print(f"Accuracy on pediatric test set: {accuracy:.4f}")
    print(f"Grad-CAM visualizations generated: {gradcam_count}")
    
    # Additional metrics
    from sklearn.metrics import classification_report, confusion_matrix
    print(f"\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Normal', 'Pneumonia']))
    print(f"\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

if __name__ == "__main__":
    test_accuracy_with_gradcam() 