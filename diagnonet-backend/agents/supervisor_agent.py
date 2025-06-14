import json
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
import logging
from datetime import datetime
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentOutput(BaseModel):
    """Schema for individual agent outputs"""
    diagnosis: str = Field(..., description="Primary diagnosis from the agent")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score (0.0-1.0)")
    explanation: str = Field(..., description="Reasoning behind the diagnosis")

class SupervisorInput(BaseModel):
    """Schema for supervisor agent input"""
    xray: Optional[AgentOutput] = Field(None, description="X-ray agent output")
    symptoms: Optional[AgentOutput] = Field(None, description="Symptoms agent output") 
    vitals: Optional[AgentOutput] = Field(None, description="Vitals agent output")

class SupervisorOutput(BaseModel):
    """Schema for supervisor agent output"""
    final_diagnosis: str = Field(..., description="Final integrated diagnosis")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Overall confidence score")
    model_used: str = Field(..., description="AI model used for analysis")
    reasoning: str = Field(..., description="Detailed medical reasoning")
    agent_agreement: str = Field(..., description="Level of agreement between agents")
    recommendations: List[str] = Field(..., description="Medical recommendations")
    severity: str = Field(..., description="Severity assessment")

class SupervisorAgent:
    """
    Supervisor agent that integrates outputs from multiple medical diagnosis agents
    using BioGPT for intelligent fusion and final diagnosis.
    """

    def __init__(self):
        self.confidence_threshold = 0.70    # Threshold for model escalation
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Initialize BioGPT model lazily (load on first use)
        self.model_name = "microsoft/biogpt"
        self.model = None
        self.tokenizer = None
        self.generator = None
        self.model_loading = False

        logger.info("BioGPT Supervisor initialized (model will load on first use)")

    def _load_biogpt_model(self):
        """Load BioGPT model on demand"""
        if self.model is not None or self.model_loading:
            return

        try:
            self.model_loading = True
            logger.info("Loading BioGPT model (this may take a few minutes on first run)...")

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )

            # Create text generation pipeline
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if torch.cuda.is_available() else -1,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )

            logger.info(f"BioGPT model loaded successfully on {self.device}")

        except Exception as e:
            logger.error(f"Failed to load BioGPT model: {str(e)}")
            logger.info("Will use rule-based analysis instead")
            self.model = None
            self.tokenizer = None
            self.generator = None
        finally:
            self.model_loading = False

    def analyze(self, agent_outputs: SupervisorInput) -> SupervisorOutput:
        """
        Perform intelligent fusion of agent outputs and return final diagnosis
        
        Args:
            agent_outputs: Outputs from specialized agents (x-ray, symptoms, vitals)
            
        Returns:
            SupervisorOutput: Final integrated diagnosis with confidence and reasoning
        """
        try:
            # Validate inputs
            available_agents = self._get_available_agents(agent_outputs)
            if not available_agents:
                raise ValueError("No agent outputs provided")
            
            # Detect agent disagreement
            disagreement_level = self._detect_disagreement(agent_outputs)
            
            # Determine analysis approach
            use_advanced = self._should_escalate(agent_outputs, disagreement_level)
            model_to_use = "BioGPT-Advanced" if use_advanced else "BioGPT-Standard"

            logger.info(f"Using analysis: {model_to_use}, Disagreement level: {disagreement_level}")

            # Generate analysis prompt
            prompt = self._create_analysis_prompt(agent_outputs, available_agents, disagreement_level)

            # Get BioGPT analysis
            ai_response = self._get_biogpt_analysis(prompt, use_advanced)

            # Parse and structure the response
            result = self._parse_ai_response(ai_response, model_to_use, disagreement_level)
            
            return result
            
        except Exception as e:
            logger.error(f"Supervisor analysis failed: {str(e)}")
            return self._create_fallback_response(agent_outputs, str(e))
    
    def _get_available_agents(self, agent_outputs: SupervisorInput) -> List[str]:
        """Get list of agents that provided outputs"""
        available = []
        if agent_outputs.xray:
            available.append("xray")
        if agent_outputs.symptoms:
            available.append("symptoms")
        if agent_outputs.vitals:
            available.append("vitals")
        return available
    
    def _detect_disagreement(self, agent_outputs: SupervisorInput) -> str:
        """
        Detect level of disagreement between agents
        
        Returns:
            str: "low", "moderate", or "high" disagreement level
        """
        diagnoses = []
        confidences = []
        
        if agent_outputs.xray:
            diagnoses.append(agent_outputs.xray.diagnosis.lower())
            confidences.append(agent_outputs.xray.confidence)
        if agent_outputs.symptoms:
            diagnoses.append(agent_outputs.symptoms.diagnosis.lower())
            confidences.append(agent_outputs.symptoms.confidence)
        if agent_outputs.vitals:
            diagnoses.append(agent_outputs.vitals.diagnosis.lower())
            confidences.append(agent_outputs.vitals.confidence)
        
        if len(diagnoses) <= 1:
            return "low"
        
        # Check for unique diagnoses
        unique_diagnoses = len(set(diagnoses))
        total_diagnoses = len(diagnoses)
        
        # Check confidence variance
        avg_confidence = sum(confidences) / len(confidences)
        confidence_variance = sum((c - avg_confidence) ** 2 for c in confidences) / len(confidences)
        
        # Determine disagreement level
        if unique_diagnoses == total_diagnoses:  # All different
            return "high"
        elif unique_diagnoses > 1 and (confidence_variance > 0.1 or avg_confidence < 0.6):
            return "moderate"
        else:
            return "low"
    
    def _should_escalate(self, agent_outputs: SupervisorInput, disagreement_level: str) -> bool:
        """
        Determine if case should be escalated to more powerful model
        
        Returns:
            bool: True if escalation is needed
        """
        # Escalate if high disagreement
        if disagreement_level == "high":
            return True
        
        # Escalate if any agent has low confidence
        confidences = []
        if agent_outputs.xray:
            confidences.append(agent_outputs.xray.confidence)
        if agent_outputs.symptoms:
            confidences.append(agent_outputs.symptoms.confidence)
        if agent_outputs.vitals:
            confidences.append(agent_outputs.vitals.confidence)
        
        if confidences and min(confidences) < self.confidence_threshold:
            return True
        
        return False
    
    def _create_analysis_prompt(self, agent_outputs: SupervisorInput, available_agents: List[str], disagreement_level: str) -> str:
        """Create detailed prompt for AI analysis"""
        
        prompt = f"""You are a senior medical supervisor AI analyzing outputs from multiple specialized diagnostic agents. Your task is to provide a final, integrated medical diagnosis.

AGENT OUTPUTS:
"""
        
        if agent_outputs.xray:
            prompt += f"""
X-RAY AGENT:
- Diagnosis: {agent_outputs.xray.diagnosis}
- Confidence: {agent_outputs.xray.confidence:.2f}
- Explanation: {agent_outputs.xray.explanation}
"""
        
        if agent_outputs.symptoms:
            prompt += f"""
SYMPTOMS AGENT:
- Diagnosis: {agent_outputs.symptoms.diagnosis}
- Confidence: {agent_outputs.symptoms.confidence:.2f}
- Explanation: {agent_outputs.symptoms.explanation}
"""
        
        if agent_outputs.vitals:
            prompt += f"""
VITALS AGENT:
- Diagnosis: {agent_outputs.vitals.diagnosis}
- Confidence: {agent_outputs.vitals.confidence:.2f}
- Explanation: {agent_outputs.vitals.explanation}
"""
        
        prompt += f"""
ANALYSIS CONTEXT:
- Available agents: {', '.join(available_agents)}
- Disagreement level: {disagreement_level}
- Timestamp: {datetime.now().isoformat()}

INSTRUCTIONS:
1. Analyze all available agent outputs for consistency and reliability
2. Consider the confidence scores and explanations from each agent
3. Identify any conflicting diagnoses and resolve them using medical knowledge
4. Provide a final integrated diagnosis with your reasoning
5. Assess the overall confidence in your final diagnosis
6. Determine severity level (LOW, MODERATE, HIGH, CRITICAL)
7. Provide specific medical recommendations

Please respond in the following JSON format:
{{
    "final_diagnosis": "Primary medical condition",
    "confidence": 0.85,
    "reasoning": "Detailed medical reasoning integrating all agent inputs...",
    "agent_agreement": "Description of how agents agreed/disagreed",
    "recommendations": ["Recommendation 1", "Recommendation 2", "..."],
    "severity": "MODERATE"
}}

Focus on medical accuracy, patient safety, and clear reasoning."""
        
        return prompt

    def _get_biogpt_analysis(self, prompt: str, use_advanced: bool = False) -> str:
        """Get analysis from BioGPT model"""
        try:
            # Load model on first use
            if self.generator is None:
                self._load_biogpt_model()

            # If model still not available, use rule-based fallback
            if self.generator is None:
                logger.info("BioGPT model not available, using rule-based analysis")
                return self._get_rule_based_analysis(prompt)

            # Prepare the prompt for BioGPT
            medical_prompt = f"""Medical Analysis Task:
{prompt}

Based on the above medical data, provide a comprehensive diagnosis analysis in JSON format:
{{
    "final_diagnosis": "Primary medical condition",
    "confidence": 0.85,
    "reasoning": "Detailed medical reasoning...",
    "agent_agreement": "Assessment of agent agreement",
    "recommendations": ["Recommendation 1", "Recommendation 2"],
    "severity": "MODERATE"
}}

Analysis:"""

            # Generate response using BioGPT
            max_length = 800 if use_advanced else 500
            temperature = 0.1 if use_advanced else 0.3

            response = self.generator(
                medical_prompt,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                num_return_sequences=1,
                return_full_text=False
            )

            generated_text = response[0]['generated_text']
            logger.info(f"BioGPT generated {len(generated_text)} characters")

            return generated_text

        except Exception as e:
            logger.error(f"BioGPT analysis failed: {str(e)}")
            logger.info("Falling back to rule-based analysis")
            return self._get_rule_based_analysis(prompt)

    def _get_rule_based_analysis(self, prompt: str) -> str:
        """Fallback rule-based analysis when BioGPT is not available"""
        logger.info("Using rule-based analysis fallback")

        # Extract key information from prompt
        if "Normal" in prompt:
            diagnosis = "Normal vital signs"
            confidence = 0.8
            severity = "LOW"
        elif "High" in prompt or "Elevated" in prompt:
            diagnosis = "Elevated vital signs requiring attention"
            confidence = 0.7
            severity = "MODERATE"
        elif "Low" in prompt or "Decreased" in prompt:
            diagnosis = "Decreased vital signs requiring monitoring"
            confidence = 0.7
            severity = "MODERATE"
        else:
            diagnosis = "Vital signs analysis completed"
            confidence = 0.6
            severity = "MODERATE"

        return f"""{{
    "final_diagnosis": "{diagnosis}",
    "confidence": {confidence},
    "reasoning": "Analysis based on vital signs patterns and clinical guidelines. Rule-based assessment used as BioGPT model fallback.",
    "agent_agreement": "Single agent analysis - no disagreement detected",
    "recommendations": ["Monitor vital signs", "Consult healthcare provider if symptoms persist", "Follow up as needed"],
    "severity": "{severity}"
}}"""

    def _parse_ai_response(self, ai_response: str, model_used: str, disagreement_level: str) -> SupervisorOutput:
        """Parse AI response and create structured output"""
        try:
            # Try to extract JSON from the response
            json_start = ai_response.find('{')
            json_end = ai_response.rfind('}') + 1

            if json_start != -1 and json_end != -1:
                json_str = ai_response[json_start:json_end]
                parsed_data = json.loads(json_str)
            else:
                # Fallback parsing if JSON not found
                parsed_data = self._fallback_parse(ai_response)

            return SupervisorOutput(
                final_diagnosis=parsed_data.get("final_diagnosis", "Unable to determine diagnosis"),
                confidence=float(parsed_data.get("confidence", 0.5)),
                model_used=model_used,
                reasoning=parsed_data.get("reasoning", "Analysis completed with limited information"),
                agent_agreement=parsed_data.get("agent_agreement", f"Disagreement level: {disagreement_level}"),
                recommendations=parsed_data.get("recommendations", ["Consult healthcare provider", "Monitor symptoms"]),
                severity=parsed_data.get("severity", "MODERATE")
            )

        except Exception as e:
            logger.error(f"Failed to parse AI response: {str(e)}")
            return self._create_fallback_response(None, f"Parsing error: {str(e)}")

    def _fallback_parse(self, response: str) -> Dict[str, Any]:
        """Fallback parsing when JSON extraction fails"""
        return {
            "final_diagnosis": "Analysis completed - see reasoning for details",
            "confidence": 0.6,
            "reasoning": response[:500] + "..." if len(response) > 500 else response,
            "agent_agreement": "Unable to assess agreement",
            "recommendations": ["Consult healthcare provider for detailed evaluation"],
            "severity": "MODERATE"
        }

    def _create_fallback_response(self, agent_outputs: Optional[SupervisorInput], error_msg: str) -> SupervisorOutput:
        """Create fallback response when analysis fails"""

        # Try to extract some basic info from agent outputs
        diagnoses = []
        avg_confidence = 0.5

        if agent_outputs:
            confidences = []
            if agent_outputs.xray:
                diagnoses.append(agent_outputs.xray.diagnosis)
                confidences.append(agent_outputs.xray.confidence)
            if agent_outputs.symptoms:
                diagnoses.append(agent_outputs.symptoms.diagnosis)
                confidences.append(agent_outputs.symptoms.confidence)
            if agent_outputs.vitals:
                diagnoses.append(agent_outputs.vitals.diagnosis)
                confidences.append(agent_outputs.vitals.confidence)

            if confidences:
                avg_confidence = sum(confidences) / len(confidences)

        primary_diagnosis = diagnoses[0] if diagnoses else "Unable to determine diagnosis"

        return SupervisorOutput(
            final_diagnosis=primary_diagnosis,
            confidence=avg_confidence,
            model_used="fallback",
            reasoning=f"Analysis failed: {error_msg}. Using available agent data for basic assessment.",
            agent_agreement="Unable to assess due to analysis failure",
            recommendations=[
                "⚠️ Analysis system encountered an error",
                "Consult healthcare provider immediately",
                "Do not rely solely on this automated assessment"
            ],
            severity="MODERATE"
        )

    def create_vitals_agent_output(self, vitals_result: Dict) -> AgentOutput:
        """
        Convert vitals agent output to supervisor input format

        Args:
            vitals_result: Output from VitalsAgent.analyze_vitals()

        Returns:
            AgentOutput: Formatted output for supervisor
        """
        return AgentOutput(
            diagnosis=vitals_result.get("primary_diagnosis", "Unknown"),
            confidence=vitals_result.get("confidence", 0.5),
            explanation=f"Vitals analysis: {vitals_result.get('primary_diagnosis', 'Unknown')} "
                       f"based on vital signs. Top conditions: "
                       f"{', '.join([c['condition'] for c in vitals_result.get('top_conditions', [])[:2]])}"
        )

    def create_xray_agent_output(self, xray_result: Dict) -> AgentOutput:
        """
        Convert X-ray agent output to supervisor input format

        Args:
            xray_result: Output from XRayAgent.analyze_xray()

        Returns:
            AgentOutput: Formatted output for supervisor
        """
        return AgentOutput(
            diagnosis=xray_result.get("primary_finding", "Unknown"),
            confidence=xray_result.get("confidence", 0.5),
            explanation=f"X-ray analysis: {xray_result.get('primary_finding', 'Unknown')}. "
                       f"Findings: {', '.join(xray_result.get('findings', [])[:2])}. "
                       f"Image quality: {xray_result.get('technical_quality', 'unknown')}"
        )

    def create_placeholder_agent_output(self, agent_type: str, diagnosis: str = "Normal", confidence: float = 0.7) -> AgentOutput:
        """
        Create placeholder output for agents not yet implemented

        Args:
            agent_type: Type of agent ("symptoms")
            diagnosis: Placeholder diagnosis
            confidence: Placeholder confidence

        Returns:
            AgentOutput: Placeholder output
        """
        explanations = {
            "symptoms": f"Symptoms analysis indicates {diagnosis}. This is a placeholder response as the symptoms agent is not yet fully implemented."
        }

        return AgentOutput(
            diagnosis=diagnosis,
            confidence=confidence,
            explanation=explanations.get(agent_type, f"{agent_type} analysis placeholder")
        )
