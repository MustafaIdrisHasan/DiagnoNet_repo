from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import subprocess
import tempfile
import os
import json
import shutil
from pathlib import Path

app = FastAPI(title="DiagnoNet Backend", version="1.0.0")

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
    return {"message": "DiagnoNet Backend API"}

@app.post("/analyze-xray")
async def analyze_xray(xray_image: UploadFile = File(...)):
    """
    Analyze a chest X-ray image using the run_cxr.py script
    """
    try:
        # Validate file type
        if not xray_image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Create temporary file to save the uploaded image
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{xray_image.filename.split('.')[-1]}") as temp_file:
            # Save uploaded file to temp location
            shutil.copyfileobj(xray_image.file, temp_file)
            temp_image_path = temp_file.name
        
        try:
            # Path to the run_cxr.py script
            script_path = Path(__file__).parent / "agents" / "x-ray chest" / "run_cxr.py"
            
            # Run the X-ray analysis script
            result = subprocess.run([
                "python", 
                str(script_path),
                "--image", temp_image_path,
                "--topk", "10",
                "--json-out", f"{temp_image_path}_analysis.json"
            ], capture_output=True, text=True, timeout=300)  # 5 minute timeout
            
            if result.returncode != 0:
                raise HTTPException(status_code=500, detail=f"X-ray analysis failed: {result.stderr}")
            
            # Try to read the JSON output
            json_output_path = f"{temp_image_path}_analysis.json"
            if os.path.exists(json_output_path):
                with open(json_output_path, 'r') as f:
                    analysis_result = json.load(f)
                # Clean up the JSON file
                os.unlink(json_output_path)
            else:
                # Fallback: parse the stdout for results
                analysis_result = {
                    "message": "Analysis completed",
                    "stdout": result.stdout,
                    "raw_output": result.stdout
                }
            
            return JSONResponse(content={
                "success": True,
                "analysis": analysis_result,
                "filename": xray_image.filename
            })
            
        finally:
            # Clean up temporary image file
            if os.path.exists(temp_image_path):
                os.unlink(temp_image_path)
    
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=408, detail="X-ray analysis timed out")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
