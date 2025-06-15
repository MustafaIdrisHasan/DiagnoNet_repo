import torchxrayvision as xrv, torch, skimage.io as io, torchvision.transforms as T

# ── 1. Load image ───────────────────────────────────────────────────────────────
img = io.imread("xray_images/sample.png")

# ── 2. Pre-process exactly as the model expects ────────────────────────────────
img = xrv.datasets.normalize(img, 255)          # scale pixels
if img.ndim == 3:                               # RGB → gray
    img = img.mean(2)
img = T.Compose([xrv.datasets.XRayCenterCrop(),
                 xrv.datasets.XRayResizer(224)])(img[None, ...])
tensor = torch.from_numpy(img).unsqueeze(0).float()   # shape (1,1,224,224)

# ── 3. Load pre-trained DenseNet-121 ───────────────────────────────────────────
model = xrv.models.DenseNet(weights="densenet121-res224-all").eval()

# ── 4. Inference ───────────────────────────────────────────────────────────────
with torch.no_grad():
    probs = torch.sigmoid(model(tensor)[0]).tolist()

# ── 5. Show top-5 findings ─────────────────────────────────────────────────────
top5 = sorted(zip(model.pathologies, probs), key=lambda x: x[1], reverse=True)[:5]
for label, p in top5:
    print(f"{label:25}: {p:.3f}") 