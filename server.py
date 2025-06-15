from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn, os, json, psycopg2
from psycopg2.extras import Json
import sys
sys.path.append("diagnonet-backend/agents/x-ray chest")
from run_cxr import analyze_xray  # your existing function

app = FastAPI()
app.add_middleware(CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Database helper
def save_to_db(image_name: str, report: dict, gradcam_path: str):
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        print("Warning: DATABASE_URL not set, skipping database save")
        return
    
    try:
        conn = psycopg2.connect(db_url)
        cur  = conn.cursor()
        cur.execute("""
          CREATE TABLE IF NOT EXISTS analyses (
            id SERIAL PRIMARY KEY,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            image_name TEXT NOT NULL,
            report JSONB NOT NULL,
            gradcam BYTEA
          );
        """)
        
        # Read gradcam image if it exists
        gradcam_bytes = None
        if gradcam_path and os.path.exists(gradcam_path):
            with open(gradcam_path, "rb") as f:
                gradcam_bytes = f.read()
        
        cur.execute("""
          INSERT INTO analyses (image_name, report, gradcam)
          VALUES (%s, %s, %s)
        """, (image_name, Json(report), psycopg2.Binary(gradcam_bytes) if gradcam_bytes else None))
        conn.commit()
        cur.close()
        conn.close()
        print(f"Successfully saved analysis for {image_name} to database")
    except Exception as e:
        print(f"Database save failed: {e}")

@app.get("/")
async def root():
    return {"message": "DiagnoNet X-ray Analysis Server", "status": "running"}

@app.post("/analyze-xray")
async def analyze_xray_endpoint(
    xray_image: UploadFile = File(...),
    mc_dropout: int = 0,
    topk: int = 5
):
    # 1. save upload to temporary location
    upload_dir = "diagnonet-backend/agents/x-ray chest/temp_uploads"
    os.makedirs(upload_dir, exist_ok=True)
    tmp = os.path.join(upload_dir, xray_image.filename)
    
    try:
        with open(tmp, "wb") as f:
            f.write(await xray_image.read())
        
        # 2. call analysis
        try:
            res = analyze_xray(tmp, topk=topk, mc_dropout=mc_dropout)
        except Exception as e:
            raise HTTPException(500, f"X-ray analysis failed: {str(e)}")
        
        # 3. persist to database
        gradcam_path = res.get("gradcam_path")
        save_to_db(xray_image.filename, res, gradcam_path)
        
        # 4. respond
        return {"success": True, "analysis": {"xray": res}, "filename": xray_image.filename}
        
    except Exception as e:
        raise HTTPException(500, f"Server error: {str(e)}")
    finally:
        # Clean up uploaded file
        if os.path.exists(tmp):
            os.remove(tmp)
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 