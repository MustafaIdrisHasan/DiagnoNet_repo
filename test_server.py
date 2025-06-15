import requests
import os

def test_xray_endpoint():
    """Test the /analyze-xray endpoint with a sample image"""
    
    # Check if test image exists
    test_image_path = "diagnonet-backend/agents/x-ray chest/test_xray.png"
    if not os.path.exists(test_image_path):
        print(f"❌ Test image not found: {test_image_path}")
        return
    
    # Prepare the request
    url = "http://localhost:8000/analyze-xray"
    
    with open(test_image_path, "rb") as f:
        files = {"xray_image": ("test_xray.png", f, "image/png")}
        data = {"topk": 5, "mc_dropout": 0}
        
        print("🚀 Testing X-ray analysis endpoint...")
        print(f"📤 Uploading: {test_image_path}")
        
        try:
            response = requests.post(url, files=files, data=data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                print("✅ Success! Server response:")
                print(f"   📊 Success: {result.get('success')}")
                
                analysis = result.get('analysis', {})
                if analysis:
                    print(f"   🎯 Top Finding: {analysis.get('top_label')} ({analysis.get('top_prob'):.1%})")
                    print(f"   📋 Pathologies found: {len(analysis.get('pathologies', []))}")
                    print(f"   🖼️  Grad-CAM generated: {'Yes' if analysis.get('gradcam_b64') else 'No'}")
                    print(f"   🩺 Clinical explanation: {'Yes' if analysis.get('clinical_explanation') else 'No'}")
                    
                    if analysis.get('gradcam_path'):
                        print(f"   📁 Grad-CAM saved to: {analysis['gradcam_path']}")
                
            else:
                print(f"❌ Error {response.status_code}: {response.text}")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Request failed: {e}")

if __name__ == "__main__":
    test_xray_endpoint() 