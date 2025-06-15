from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import json
import time
import asyncio

app = FastAPI(title="DiagnoNet Backend - Simple", version="1.0.0")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "DiagnoNet Backend API - Simple Version"}

@app.post("/analyze-xray")
async def analyze_xray(xray_image: UploadFile = File(...)):
    """
    Simplified X-ray analysis that returns mock results for testing
    """
    try:
        # Validate file type
        if not xray_image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Simulate processing time
        await asyncio.sleep(2)
        
        # Return mock analysis results that match the frontend expectations
        mock_result = {
            "success": True,
            "analysis": {
                "xray": {
                    "pathologies": [
                        ["No Finding", 0.85],
                        ["Infiltration", 0.12],
                        ["Atelectasis", 0.08],
                        ["Effusion", 0.05],
                        ["Pneumonia", 0.03]
                    ],
                    "top_label": "No Finding",
                    "top_prob": 0.85,
                    "clinical_explanation": "Based on the AI analysis with 85% confidence, this chest X-ray shows no significant abnormal findings. The lung fields appear clear with normal cardiac silhouette and no evidence of infiltrates, consolidation, or effusion. This suggests a healthy chest radiograph with no immediate pathological concerns.",
                    "gradcam_file": None,
                    "gradcam_b64": None
                },
                "vitals": {},
                "symptoms": {}
            },
            "filename": xray_image.filename
        }
        
        return JSONResponse(content=mock_result)
            
    except Exception as e:
        print(f"Error in analyze_xray: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    import asyncio
    uvicorn.run(app, host="0.0.0.0", port=8000) 