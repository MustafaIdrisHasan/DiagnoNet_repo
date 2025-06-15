# Chest X-Ray Analysis with TorchXRayVision

This project implements chest X-ray analysis using TorchXRayVision with custom Grad-CAM visualization for explainable AI diagnostics.

## 🚀 Quick Start

### 1. Environment Setup with uv

```bash
# Install uv (if not already installed)
pipx install uv

# Create and activate virtual environment
uv venv cxr_env

# Windows (PowerShell)
.\cxr_env\Scripts\Activate.ps1

# macOS/Linux
source cxr_env/bin/activate

# Install dependencies
uv pip install torch torchvision torchxrayvision scikit-image scikit-learn matplotlib opencv-python pydicom kaggle grad-cam pathlib
```

### 2. Basic Usage

#### Quick Test with Sample Image
```bash
python quick_test.py
```

#### Command Line Interface
```bash
# Basic usage with DenseNet (224x224)
python run_cxr.py --image xray_images/sample.png --topk 8

# Enhanced usage with ResNet-50 (512x512) - default, better small-lesion detection
python run_cxr.py --image xray_images/sample.png --model resnet --topk 8

# With Monte-Carlo dropout uncertainty estimation
python run_cxr.py --image xray_images/sample.png --mc-dropout 20

# Export results to JSON with base64 Grad-CAM
python run_cxr.py --image xray_images/sample.png --json-out results.json --mc-dropout 10
```

#### Test Setup and Grad-CAM
```bash
python test_setup.py
```

## 📊 Dataset Setup (Kaggle)

### 1. Kaggle API Setup
1. Go to [https://www.kaggle.com/account](https://www.kaggle.com/account)
2. Scroll to 'API' section and click 'Create New API Token'
3. Download `kaggle.json` file
4. Place it in your Kaggle config directory:
   - **Windows**: `C:\Users\{username}\.kaggle\kaggle.json`
   - **Linux/Mac**: `~/.kaggle/kaggle.json`

### 2. Download Dataset
```bash
# Run the setup script (guides you through the process)
python setup_kaggle_dataset.py

# Or manually:
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
# Extract to 'paed' directory (done automatically by setup script)
```

### 3. Run Accuracy Testing
```bash
python accuracy_test.py
```

## 📁 Project Structure

```
.
├── README.md                    # This file
├── requirements.txt            # Dependencies (optional)
│
├── xray_images/                # Original X-ray images
│   ├── sample.png              # Sample chest X-ray
│   ├── trial_image1.jpeg       # Test X-ray 1
│   ├── trial_image2.jpeg       # Test X-ray 2
│   └── Chest-X-ray-left-pneumothorax.png
│
├── gradcam_visualizations/     # AI explanation heatmaps
│   ├── sample_gradcam_Emphysema.png
│   ├── sample_gradcam_Pneumonia.png
│   ├── trial_image1_gradcam_Pneumonia.png
│   └── test_gradcam_sample.png
│
├── quick_test.py              # Basic inference test
├── run_cxr.py                 # Enhanced CLI tool with Grad-CAM
├── test_setup.py              # Setup verification
├── analyze_my_xray.py         # User-friendly analysis script
├── preprocess_module.py       # Image preprocessing utilities
├── gradcam_utils.py           # Custom Grad-CAM implementation
├── accuracy_test.py           # Dataset evaluation with Grad-CAM
├── setup_kaggle_dataset.py    # Kaggle dataset setup helper
│
└── paed/                      # Kaggle dataset (after download)
    └── chest_xray/
        ├── test/
        │   ├── PNEUMONIA/
        │   └── NORMAL/
        └── train/
            ├── PNEUMONIA/
            └── NORMAL/
```

## 🔬 Features

### 1. **Multi-pathology Detection with Enhanced Models**
- **DenseNet-121** @ 224×224px (original)
- **ResNet-50** @ 512×512px (NEW - default, better small-lesion detection)
- Detects 18+ chest pathologies including:
  - Pneumonia
  - Pneumothorax
  - Emphysema
  - Infiltration
  - Lung Opacity
  - Fracture
  - And more...

### 2. **Advanced Explainable AI**
- **Grad-CAM++** implementation (enhanced attribution)
- **Lung-mask preprocessing** using PSPNet segmentation
- Visual heatmaps showing model attention
- Overlay visualizations for clinical interpretation
- Masked heatmaps focus only on lung regions

### 3. **Uncertainty Quantification**
- **Monte-Carlo Dropout** for prediction uncertainty
- Run multiple forward passes to estimate confidence intervals
- Get mean ± standard deviation for all pathology predictions
- Useful for clinical decision support

### 4. **Advanced Export Capabilities**
- **JSON export** with structured results
- **Base64-encoded Grad-CAM** images for API integration
- Top pathology probabilities with uncertainty
- Pneumonia and Pneumothorax always reported
- Ready for downstream LLM processing

### 5. **Flexible Input Formats**
- PNG, JPEG, DICOM support
- Automatic preprocessing and normalization
- Handles RGB and grayscale images

### 6. **Comprehensive Evaluation**
- Accuracy testing on pediatric dataset
- Classification reports and confusion matrices
- Batch processing capabilities

## 📈 Example Results

### Sample Analysis Output
```
Pneumonia                : 0.515  ±0.032
Pneumothorax             : 0.089  ±0.015

Generating Grad-CAM++ for: Emphysema

Top 5 findings:
Emphysema                : 0.623  ±0.045
Infiltration             : 0.562  ±0.028
Fracture                 : 0.530  ±0.041
Pneumonia                : 0.515  ±0.032
Lung Opacity             : 0.515  ±0.037

Saved heat-map → xray_images/sample_gradcam_Emphysema.png
Wrote JSON → results.json
```

### Grad-CAM Visualization
The system generates three-panel visualizations:
1. **Original Image**: Raw chest X-ray
2. **Heatmap**: Grad-CAM attention map
3. **Overlay**: Combined visualization with prediction confidence

## 🛠 Scripts Description

| Script | Purpose | Usage |
|--------|---------|-------|
| `quick_test.py` | Basic model inference | `python quick_test.py` |
| `run_cxr.py` | Enhanced CLI with Grad-CAM | `python run_cxr.py --image xray_images/image.png --topk 5` |
| `test_setup.py` | Verify installation | `python test_setup.py` |
| `accuracy_test.py` | Dataset evaluation | `python accuracy_test.py` |
| `setup_kaggle_dataset.py` | Dataset download helper | `python setup_kaggle_dataset.py` |

## 🔧 Technical Details

### Model Architecture
- **Base Models**: 
  - DenseNet-121 @ 224×224px (compact, faster)
  - ResNet-50 @ 512×512px (default, better detail detection)
- **Input Processing**: Lung-masked grayscale images
- **Pre-training**: Multi-dataset chest X-ray training
- **Output**: 18+ pathology probabilities with uncertainty estimates
- **Explainability**: Grad-CAM++ with lung-region masking

### Preprocessing Pipeline
```python
# Image normalization
img = xrv.datasets.normalize(img, 255)

# RGB to grayscale conversion
if img.ndim == 3:
    img = img.mean(2)

# Center crop and resize (size depends on model)
transforms = [
    xrv.datasets.XRayCenterCrop(),
    xrv.datasets.XRayResizer(side)  # 224 for DenseNet, 512 for ResNet
]

# Lung segmentation and masking
with torch.no_grad():
    lung_mask = (SEG_MODEL(tensor)[:, [4,5]].sum(1, keepdim=True) > 0.5).float()
tensor *= lung_mask  # Apply mask to focus on lung regions
```

### Grad-CAM++ Implementation
- **Enhanced Attribution**: Grad-CAM++ provides better localization than vanilla Grad-CAM
- **Lung-Masked Heatmaps**: Combined with PSPNet segmentation for clinical focus  
- **Multi-Class Support**: Targets last convolutional layer for any pathology
- **High-Resolution**: Works with both 224×224 and 512×512 input sizes
- **Export Ready**: Base64 encoding for API/web integration

## 🚨 Requirements

- Python 3.8+
- PyTorch
- TorchXRayVision
- scikit-image, scikit-learn
- matplotlib, opencv-python
- **grad-cam** (for Grad-CAM++ implementation)
- **pathlib** (for JSON export functionality)
- Optional: CUDA for GPU acceleration

## 📝 Notes

- **Dataset Size**: ~1.15GB (5,863 X-ray images)
- **Processing Time**: ~1-2 seconds per image (CPU)
- **Memory Usage**: ~2-4GB during batch processing
- **Grad-CAM Generation**: Limited to first 10 pneumonia cases to avoid file clutter

## 🆕 Enhanced Features (Latest Update)

### What's New in This Version:

1. **Higher-Resolution Backbone**: 
   - ResNet-50 @ 512×512px is now the default (vs. 224×224px)
   - Better detection of small lesions and subtle findings
   - Switch to DenseNet with `--model densenet` for faster processing

2. **Lung-Mask Preprocessing**:
   - PSPNet-based lung segmentation before analysis
   - Model focuses only on lung regions, ignoring tubes/text/artifacts
   - Cleaner, more clinically relevant heatmaps

3. **Grad-CAM++ Visualization**:
   - Superior attribution quality vs. vanilla Grad-CAM
   - Better localization of pathology regions
   - Combined with lung masking for focused attention

4. **Monte-Carlo Dropout Uncertainty**:
   - Run multiple forward passes with `--mc-dropout N`
   - Get confidence intervals: `Pneumonia: 0.515 ±0.032`
   - Essential for clinical decision support

5. **JSON Export for API Integration**:
   - Structured output with `--json-out results.json`
   - Base64-encoded Grad-CAM images included
   - Ready for downstream processing/LLM integration

### Example Enhanced Workflow:
```bash
# Full-featured analysis with uncertainty and export
python run_cxr.py \
  --image patient_xray.png \
  --model resnet \
  --mc-dropout 20 \
  --json-out analysis_results.json
```

### Clinical Benefits:
- **Higher Accuracy**: 512×512 resolution captures more detail
- **Focused Analysis**: Lung masking reduces false positives
- **Confidence Estimates**: Monte-Carlo dropout provides uncertainty
- **Integration Ready**: JSON export for electronic health records

## 🤝 Contributing

Feel free to improve this implementation:
1. Add more visualization options
2. Implement additional evaluation metrics
3. Add support for more image formats
4. Optimize processing speed

## 📄 License

This project uses TorchXRayVision and follows their licensing terms. Please refer to the original repository for detailed license information.

---

**Note**: This is an educational/research tool. Always consult qualified medical professionals for actual diagnostic decisions. 