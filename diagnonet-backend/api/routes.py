from fastapi import APIRouter, HTTPException, File, UploadFile, Form
from pydantic import BaseModel, Field
from typing import Dict, Optional
from agents.vitals_agent import VitalsAgent
from agents.xray_agent import XRayAgent
from agents.supervisor_agent import SupervisorAgent, SupervisorInput, SupervisorOutput
import logging
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize the agents
vitals_agent = VitalsAgent()
xray_agent = XRayAgent()
supervisor_agent = SupervisorAgent()

class VitalsPayload(BaseModel):
    """Pydantic model for vital signs input"""
    systolic_bp: float = Field(..., ge=50, le=300, description="Systolic blood pressure (mmHg)")
    diastolic_bp: float = Field(..., ge=30, le=200, description="Diastolic blood pressure (mmHg)")
    heart_rate: float = Field(..., ge=30, le=250, description="Heart rate (beats per minute)")
    temperature: float = Field(..., ge=90, le=110, description="Body temperature (Fahrenheit)")
    respiratory_rate: float = Field(..., ge=5, le=50, description="Respiratory rate (breaths per minute)")
    oxygen_saturation: float = Field(..., ge=70, le=100, description="Oxygen saturation (%)")
    
    class Config:
        schema_extra = {
            "example": {
                "systolic_bp": 120,
                "diastolic_bp": 80,
                "heart_rate": 72,
                "temperature": 98.6,
                "respiratory_rate": 16,
                "oxygen_saturation": 98
            }
        }

class HealthResponse(BaseModel):
    """Health check response model"""
    status: str
    message: str

@router.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "DiagnoNet BioGPT Medical AI",
        "version": "2.0.0",
        "endpoints": {
            "health": "/health",
            "biogpt_analysis": "/biogpt-analysis"
        }
    }

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        # Quick vitals agent test
        test_vitals = {
            "systolic_bp": 120,
            "diastolic_bp": 80,
            "heart_rate": 72,
            "temperature": 98.6,
            "respiratory_rate": 16,
            "oxygen_saturation": 98
        }
        
        vitals_agent.analyze_vitals(test_vitals)
        
        return HealthResponse(
            status="healthy",
            message="BioGPT Medical AI System Operational"
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"System health check failed: {str(e)}"
        )

@router.post("/biogpt-analysis", response_model=SupervisorOutput)
async def biogpt_medical_analysis(
    vitals_data: str = Form(..., description="JSON string of vital signs data"),
    xray_image: Optional[UploadFile] = File(None, description="Optional X-ray image file")
):
    """
    BioGPT Medical Analysis Endpoint with X-ray Support
    
    Advanced AI-powered medical diagnosis using Microsoft's BioGPT model.
    Analyzes vital signs and optional X-ray images for comprehensive medical assessment.
    
    Features:
    - BioGPT medical AI model
    - Multi-agent intelligent fusion
    - X-ray image analysis
    - Medical-grade reasoning
    - Confidence scoring
    """
    try:
        logger.info(f"BioGPT analysis requested with X-ray: {xray_image is not None}")

        # Parse vitals data from form
        try:
            vitals_dict = json.loads(vitals_data)
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=400,
                detail="Invalid vitals data format. Must be valid JSON."
            )

        # Validate vital signs relationships
        if vitals_dict['systolic_bp'] <= vitals_dict['diastolic_bp']:
            raise HTTPException(
                status_code=400,
                detail="Systolic blood pressure must be higher than diastolic blood pressure"
            )

        # Step 1: Analyze vitals
        vitals_result = vitals_agent.analyze_vitals(vitals_dict)
        logger.info(f"Vitals analysis complete: {vitals_result['primary_diagnosis']}")

        # Step 2: Convert to supervisor format
        vitals_agent_output = supervisor_agent.create_vitals_agent_output(vitals_result)

        # Step 3: Analyze X-ray if provided
        xray_agent_output = None
        if xray_image:
            try:
                # Validate file type
                if not xray_image.content_type.startswith('image/'):
                    raise HTTPException(
                        status_code=400,
                        detail="File must be an image (PNG, JPG, or JPEG)"
                    )
                
                # Read image data
                image_data = await xray_image.read()
                
                # Analyze X-ray
                xray_result = xray_agent.analyze_xray(image_data, xray_image.filename)
                xray_agent_output = supervisor_agent.create_xray_agent_output(xray_result)
                
                logger.info(f"X-ray analysis complete: {xray_result['primary_finding']}")
                
            except Exception as e:
                logger.error(f"X-ray analysis failed: {str(e)}")
                # Create error output for X-ray analysis
                xray_agent_output = supervisor_agent.create_xray_agent_output({
                    'primary_finding': 'X-ray analysis failed',
                    'findings': [f'Error: {str(e)}'],
                    'confidence': 0.0,
                    'technical_quality': 'error'
                })

        # Step 4: Create placeholder for symptoms agent (not implemented)
        primary_diagnosis = vitals_result['primary_diagnosis']
        symptoms_output = supervisor_agent.create_placeholder_agent_output(
            "symptoms",
            diagnosis=primary_diagnosis,
            confidence=0.6
        )

        # Step 5: Create supervisor input
        supervisor_input = SupervisorInput(
            vitals=vitals_agent_output,
            symptoms=symptoms_output,
            xray=xray_agent_output  # Will be None if no X-ray provided
        )

        # Step 6: Run BioGPT analysis
        supervisor_result = supervisor_agent.analyze(supervisor_input)

        logger.info(f"BioGPT analysis complete. Final diagnosis: {supervisor_result.final_diagnosis}")

        return supervisor_result

    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid input: {str(e)}"
        )
    except Exception as e:
        logger.error(f"BioGPT analysis error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Medical analysis failed: {str(e)}"
        )
