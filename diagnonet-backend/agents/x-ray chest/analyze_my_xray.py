import torchxrayvision as xrv, torch, skimage.io as io, torchvision.transforms as T
from gradcam_utils import get_gradcam, save_gradcam_visualization, get_gradcam_output_path
import os

# ── CHANGE THIS LINE TO YOUR IMAGE PATH ──────────────────────────────────────
IMAGE_PATH = "xray_images/sample.png"  # ← Change this to your X-ray image path
# Examples:
# IMAGE_PATH = "xray_images/my_chest_xray.jpg"
# IMAGE_PATH = "C:\\Users\\YourName\\Downloads\\xray.png"

print(f"Analyzing X-ray: {IMAGE_PATH}")
print("=" * 50)

# ── 1. Load your image ───────────────────────────────────────────────────────
try:
    img = io.imread(IMAGE_PATH)
    print(f"✓ Image loaded successfully! Size: {img.shape}")
except Exception as e:
    print(f"❌ Error loading image: {e}")
    print("Make sure the file path is correct and the file exists.")
    exit()

# ── 2. Pre-process exactly as the model expects ─────────────────────────────
# Keep original image for gradcam
original_img = img.copy()

img = xrv.datasets.normalize(img, 255)          # scale pixels
if img.ndim == 3:                               # RGB → gray
    img = img.mean(2)
img = T.Compose([xrv.datasets.XRayCenterCrop(),
                 xrv.datasets.XRayResizer(224)])(img[None, ...])
tensor = torch.from_numpy(img).unsqueeze(0).float()   # shape (1,1,224,224)

# ── 3. Load pre-trained medical AI model ────────────────────────────────────
print("Loading AI model...")
model = xrv.models.DenseNet(weights="densenet121-res224-all").eval()

# ── 4. AI Analysis ──────────────────────────────────────────────────────────
print("Running AI analysis...")
with torch.no_grad():
    probs = torch.sigmoid(model(tensor)[0]).tolist()

# ── 5. Show ALL findings (sorted by probability) ────────────────────────────
print("\n🔍 AI ANALYSIS RESULTS:")
print("=" * 50)
all_findings = sorted(zip(model.pathologies, probs), key=lambda x: x[1], reverse=True)

for i, (condition, probability) in enumerate(all_findings, 1):
    confidence = probability * 100
    if confidence > 30:  # Only show conditions with >30% confidence
        status = "🔴 HIGH" if confidence > 60 else "🟡 MODERATE" if confidence > 40 else "🟢 LOW"
        print(f"{i:2d}. {condition:25}: {confidence:5.1f}% {status}")

# ── 6. Generate Grad-CAM for the most suspicious finding ─────────────────────
print(f"\n🔍 GENERATING VISUAL EXPLANATION...")
print("=" * 50)

try:
    # Find the highest-confidence condition
    highest_condition = max(all_findings, key=lambda x: x[1])
    condition_name, condition_prob = highest_condition
    
    if condition_prob > 0.3:  # Only generate if confidence > 30%
        print(f"Creating Grad-CAM heatmap for: {condition_name} ({condition_prob*100:.1f}%)")
        
        # Generate heatmap (use original image for gradcam)
        heat = get_gradcam(original_img, model, condition_name)
        
        # Save visualization using helper function
        output_path = get_gradcam_output_path(IMAGE_PATH, suffix="analysis", pathology=condition_name)
        save_gradcam_visualization(original_img, heat, condition_prob, output_path, condition_name)
        
        print(f"✅ Visual explanation saved: {output_path}")
        print("This shows WHERE the AI detected the condition in red/yellow areas.")
    else:
        print("⚠️  No high-confidence findings - skipping visualization.")
        
except Exception as e:
    print(f"❌ Error generating visualization: {e}")

print("\n⚠️  IMPORTANT DISCLAIMER:")
print("This is an AI tool for educational purposes only.")
print("Always consult a qualified medical professional for diagnosis.")
print("Do not use this for actual medical decisions.")

print(f"\n✅ Analysis complete for: {IMAGE_PATH}") 