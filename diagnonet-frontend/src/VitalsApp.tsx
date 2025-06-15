import React, { useState, ChangeEvent } from 'react';
import { motion } from 'framer-motion';
import { Heart, Thermometer, Activity, Gauge, Wind, Droplets, AlertCircle, Loader2, Brain, Upload, X, FileImage, CheckCircle } from 'lucide-react';

interface VitalsData {
  systolic_bp: number;
  diastolic_bp: number;
  heart_rate: number;
  temperature: number;
  respiratory_rate: number;
  oxygen_saturation: number;
}

interface XRayFile {
  file: File;
  preview: string;
  name: string;
  size: number;
}

interface PatientData {
  name: string;
  age: string;
  height: string;
  symptoms: string;
  medicalHistory: string;
}

interface DiagnosisResult {
  final_diagnosis: string;
  confidence: number;
  model_used: string;
  reasoning: string;
  agent_agreement: string;
  recommendations: string[];
  severity: string;
}

interface XRayAnalysisResult {
  success: boolean;
  analysis: {
    xray?: {
      pathologies: [string, number][];
      top_label: string;
      top_prob: number;
      clinical_explanation?: string;
      gradcam_file?: string;
      gradcam_b64?: string;
    };
    vitals?: any;
    symptoms?: any;
  };
  filename: string;
}

const VitalsApp: React.FC = () => {
  const [patientData, setPatientData] = useState<PatientData>({
    name: '',
    age: '',
    height: '',
    symptoms: '',
    medicalHistory: ''
  });

  const [vitals, setVitals] = useState<VitalsData>({
    systolic_bp: 120,
    diastolic_bp: 80,
    heart_rate: 72,
    temperature: 98.6,
    respiratory_rate: 16,
    oxygen_saturation: 98
  });

  const [result, setResult] = useState<DiagnosisResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [xrayFile, setXrayFile] = useState<XRayFile | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [xrayAnalysisResult, setXrayAnalysisResult] = useState<XRayAnalysisResult | null>(null);
  const [xrayAnalyzing, setXrayAnalyzing] = useState(false);

  const handlePatientDataChange = (field: keyof PatientData, value: string) => {
    setPatientData(prev => ({
      ...prev,
      [field]: value
    }));
  };

  const handleInputChange = (field: keyof VitalsData, value: string) => {
    const numValue = parseFloat(value);
    if (!isNaN(numValue)) {
      setVitals(prev => ({
        ...prev,
        [field]: numValue
      }));
    }
  };

  const handleXRayUpload = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    setUploadError(null);
    setXrayAnalysisResult(null);

    if (!file) {
      setXrayFile(null);
      return;
    }

    // Validate file type
    const allowedTypes = ['image/png', 'image/jpeg', 'image/jpg'];
    if (!allowedTypes.includes(file.type)) {
      setUploadError('Please upload a PNG, JPG, or JPEG image file.');
      return;
    }

    // Validate file size (10MB limit)
    const maxSize = 10 * 1024 * 1024; // 10MB
    if (file.size > maxSize) {
      setUploadError(`File size (${(file.size / 1024 / 1024).toFixed(1)}MB) exceeds 10MB limit.`);
      return;
    }

    // Create preview
    const reader = new FileReader();
    reader.onload = (e) => {
      const xrayFileData = {
        file,
        preview: e.target?.result as string,
        name: file.name,
        size: file.size
      };
      setXrayFile(xrayFileData);
    };
    reader.readAsDataURL(file);
  };

  const processXRayAnalysis = async () => {
    if (!xrayFile) {
      setUploadError('Please upload an X-ray image first.');
      return;
    }

    setXrayAnalyzing(true);
    setUploadError(null);
    setXrayAnalysisResult(null);

    try {
      const formData = new FormData();
      formData.append('xray_image', xrayFile.file);
      formData.append('topk', '5');
      formData.append('mc_dropout', '0');

      const response = await fetch('http://localhost:8000/analyze-xray', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to analyze X-ray');
      }

      const analysisData: XRayAnalysisResult = await response.json();
      setXrayAnalysisResult(analysisData);
    } catch (err) {
      setUploadError(err instanceof Error ? err.message : 'X-ray analysis failed');
    } finally {
      setXrayAnalyzing(false);
    }
  };

  const removeXRayFile = () => {
    setXrayFile(null);
    setUploadError(null);
  };

  const submitVitals = async () => {
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      // Create FormData for multipart request
      const formData = new FormData();
      formData.append('vitals_data', JSON.stringify(vitals));
      
      // Add X-ray image if provided
      if (xrayFile) {
        formData.append('xray_image', xrayFile.file);
      }

      const response = await fetch('http://localhost:8000/biogpt-analysis', {
        method: 'POST',
        body: formData, // No Content-Type header needed for FormData
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to analyze vitals');
      }

      const data: DiagnosisResult = await response.json();
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-center py-8 px-4"
      >
        <div className="flex items-center justify-center mb-4">
          <Brain className="w-8 h-8 mr-3 text-blue-600" />
          <h1 className="text-4xl font-bold text-gray-800">DiagnoNet</h1>
        </div>
        <p className="text-gray-600 text-lg">BioGPT Medical AI Analysis</p>
        <p className="text-sm text-gray-500 mt-1">Advanced AI-powered vital signs diagnosis</p>
      </motion.div>

      <div className="max-w-3xl mx-auto px-4 pb-8 space-y-6">

        {/* Section 1: Patient Demographics & Symptoms */}
        <motion.section
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="card"
        >
          <h2 className="card-title flex items-center mb-6">
            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="w-6 h-6 mr-2 text-green-600">
              <path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/>
              <circle cx="9" cy="7" r="4"/>
              <path d="m22 21-3-3m0 0a5.5 5.5 0 1 0-7.78-7.78 5.5 5.5 0 0 0 7.78 7.78Z"/>
            </svg>
            Patient Information & Symptoms
          </h2>

          <div className="space-y-6">
            {/* Patient Name */}
            <div>
              <label>Patient Name</label>
              <input
                type="text"
                value={patientData.name}
                onChange={(e) => handlePatientDataChange('name', e.target.value)}
                placeholder="Enter full name"
              />
            </div>

            {/* Patient Age & Height */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label>Age (years)</label>
                <input
                  type="number"
                  value={patientData.age}
                  onChange={(e) => handlePatientDataChange('age', e.target.value)}
                  placeholder="e.g., 45"
                  min="0"
                  max="120"
                />
              </div>
              <div>
                <label>Height (cm)</label>
                <input
                  type="number"
                  value={patientData.height}
                  onChange={(e) => handlePatientDataChange('height', e.target.value)}
                  placeholder="e.g., 170"
                  min="30"
                  max="250"
                />
              </div>
            </div>

            {/* Symptoms Input */}
            <div>
              <label>Current Symptoms</label>
              <textarea
                value={patientData.symptoms}
                onChange={(e) => handlePatientDataChange('symptoms', e.target.value)}
                placeholder="Describe current symptoms (e.g., chest pain, shortness of breath, fever, cough, fatigue...)"
                rows={4}
              />
              <p className="text-xs text-gray-500 mt-1">Be as detailed as possible about symptoms, duration, and severity</p>
            </div>

            {/* Medical History Input */}
            <div>
              <label>Medical History & Background</label>
              <textarea
                value={patientData.medicalHistory}
                onChange={(e) => handlePatientDataChange('medicalHistory', e.target.value)}
                placeholder="Previous medical conditions, medications, allergies, family history, recent procedures..."
                rows={4}
              />
              <p className="text-xs text-gray-500 mt-1">Include relevant medical history, current medications, and known allergies</p>
            </div>

            {/* Progress Indicator */}
            <div className="flex items-center justify-between pt-4 border-t border-gray-200">
              <span className="text-sm text-gray-600">Step 1 of 3: Patient Information</span>
              <div className="flex space-x-2">
                <div className="progress-dot progress-dot-complete"></div>
                <div className="progress-dot progress-dot-inactive"></div>
                <div className="progress-dot progress-dot-inactive"></div>
              </div>
            </div>
          </div>
        </motion.section>

        {/* Section 2: Vital Signs Input */}
        <motion.section
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="card"
        >
          <h2 className="card-title flex items-center mb-6">
            <Activity className="w-6 h-6 mr-2 text-blue-600" />
            Vital Signs Measurement
          </h2>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
            {/* Blood Pressure */}
            <div>
              <label>
                <Gauge className="w-4 h-4 inline mr-1" />
                Systolic BP (mmHg)
              </label>
              <input
                type="number"
                value={vitals.systolic_bp}
                onChange={(e) => handleInputChange('systolic_bp', e.target.value)}
                min="50" max="300"
              />
            </div>
            <div>
              <label>
                Diastolic BP (mmHg)
              </label>
              <input
                type="number"
                value={vitals.diastolic_bp}
                onChange={(e) => handleInputChange('diastolic_bp', e.target.value)}
                min="30" max="200"
              />
            </div>

            {/* Heart Rate */}
            <div>
              <label>
                <Heart className="w-4 h-4 inline mr-1 text-red-500" />
                Heart Rate (BPM)
              </label>
              <input
                type="number"
                value={vitals.heart_rate}
                onChange={(e) => handleInputChange('heart_rate', e.target.value)}
                min="30" max="250"
              />
            </div>

            {/* Temperature */}
            <div>
              <label>
                <Thermometer className="w-4 h-4 inline mr-1 text-orange-500" />
                Temperature (°F)
              </label>
              <input
                type="number"
                step="0.1"
                value={vitals.temperature}
                onChange={(e) => handleInputChange('temperature', e.target.value)}
                min="90" max="110"
              />
            </div>

            {/* Respiratory Rate */}
            <div>
              <label>
                <Wind className="w-4 h-4 inline mr-1 text-blue-500" />
                Respiratory Rate (breaths/min)
              </label>
              <input
                type="number"
                value={vitals.respiratory_rate}
                onChange={(e) => handleInputChange('respiratory_rate', e.target.value)}
                min="5" max="50"
              />
            </div>

            {/* Oxygen Saturation */}
            <div>
              <label>
                <Droplets className="w-4 h-4 inline mr-1 text-cyan-500" />
                Oxygen Saturation (%)
              </label>
              <input
                type="number"
                value={vitals.oxygen_saturation}
                onChange={(e) => handleInputChange('oxygen_saturation', e.target.value)}
                min="70" max="100"
              />
            </div>
          </div>

          <motion.button
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            onClick={submitVitals}
            disabled={loading}
            className="btn-primary w-full"
          >
            {loading ? (
              <>
                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                BioGPT Analyzing{xrayFile ? ' (with X-ray)' : ''}...
              </>
            ) : (
              <>
                <Brain className="w-4 h-4 mr-2" />
                Run BioGPT Medical Analysis{xrayFile ? ' + X-ray' : ''}
              </>
            )}
          </motion.button>

          {/* Progress Indicator */}
          <div className="flex items-center justify-between pt-4 border-t border-gray-200">
            <span className="text-sm text-gray-600">Step 2 of 3: Vital Signs</span>
            <div className="flex space-x-2">
              <div className="progress-dot progress-dot-complete"></div>
              <div className="progress-dot progress-dot-active"></div>
              <div className="progress-dot progress-dot-inactive"></div>
            </div>
          </div>
        </motion.section>

        {/* Section 3: X-Ray Analysis Agent */}
        <motion.section
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="card"
        >
          <h2 className="card-title flex items-center mb-6">
            <FileImage className="w-6 h-6 mr-2 text-green-600" />
            X-Ray Analysis Agent
          </h2>

          <div className="space-y-4">
            {/* Upload Area */}
            <div className="upload-area">
              {!xrayFile ? (
                <div>
                  <Upload className="w-12 h-12 mx-auto text-gray-400 mb-4" />
                  <p className="text-lg font-medium text-gray-700 mb-2">Upload Chest X-Ray Image</p>
                  <p className="text-sm text-gray-500 mb-4">
                    Optional: Upload a chest X-ray for enhanced AI analysis
                  </p>
                  <input
                    type="file"
                    accept=".png,.jpg,.jpeg,image/png,image/jpeg"
                    onChange={handleXRayUpload}
                    className="hidden"
                    id="xray-upload"
                  />
                  <label
                    htmlFor="xray-upload"
                    className="file-upload-label"
                  >
                    <Upload className="w-4 h-4 mr-2" />
                    Choose X-Ray Image
                  </label>
                  <p className="text-xs text-gray-400 mt-2">
                    Supports PNG, JPG, JPEG • Max 10MB
                  </p>
                </div>
              ) : (
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-3">
                      <CheckCircle className="w-6 h-6 text-green-500" />
                      <div className="text-left">
                        <p className="font-medium text-gray-700">{xrayFile.name}</p>
                        <p className="text-sm text-gray-500">
                          {(xrayFile.size / 1024 / 1024).toFixed(1)}MB
                        </p>
                      </div>
                    </div>
                    <button
                      onClick={removeXRayFile}
                      className="p-1 text-gray-400 hover:text-red-500 transition-colors"
                    >
                      <X className="w-5 h-5" />
                    </button>
                  </div>

                  {/* Image Preview */}
                  <div className="image-preview">
                    <img
                      src={xrayFile.preview}
                      alt="X-ray preview"
                    />
                  </div>

                  {/* Process X-ray Button */}
                  <motion.button
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                    onClick={processXRayAnalysis}
                    disabled={xrayAnalyzing}
                    className="btn-primary w-full"
                  >
                    {xrayAnalyzing ? (
                      <>
                        <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                        Processing X-ray with AI...
                      </>
                    ) : (
                      <>
                        <Brain className="w-4 h-4 mr-2" />
                        Process X-ray & Generate Grad-CAM
                      </>
                    )}
                  </motion.button>

                  {/* Analysis Status */}
                  {xrayAnalysisResult && (
                    <div className="flex items-center justify-center space-x-2 text-sm text-green-600">
                      <CheckCircle className="w-4 h-4" />
                      <span>X-ray analysis completed successfully</span>
                    </div>
                  )}

                  {/* Grad-CAM Visualization - Show immediately after processing */}
                  {xrayAnalysisResult?.analysis?.xray?.gradcam_b64 && (
                    <motion.div
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      className="mt-4"
                    >
                      <div className="bg-gradient-to-br from-orange-50 to-red-50 p-4 rounded-lg border border-orange-200">
                        <h4 className="font-semibold text-orange-800 mb-3 flex items-center">
                          <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="w-5 h-5 mr-2">
                            <rect width="18" height="18" x="3" y="3" rx="2" ry="2"/>
                            <circle cx="9" cy="9" r="2"/>
                            <path d="m21 15-3.086-3.086a2 2 0 0 0-2.828 0L6 21"/>
                          </svg>
                          🔥 AI Heat Map - Regions of Interest
                        </h4>
                        <div className="bg-white p-3 rounded-lg border border-orange-100 shadow-sm">
                          <img 
                            src={`data:image/png;base64,${xrayAnalysisResult.analysis.xray.gradcam_b64}`}
                            alt="Grad-CAM heat map visualization"
                            className="w-full max-w-2xl mx-auto rounded-lg shadow-md"
                            style={{ maxHeight: '500px', objectFit: 'contain' }}
                          />
                          <div className="mt-3 pt-3 border-t border-orange-100 text-center">
                            <p className="text-sm font-medium text-orange-800 mb-1">
                              🎯 Focus Areas for: {xrayAnalysisResult.analysis.xray.top_label}
                            </p>
                            <p className="text-xs text-orange-700">
                              Red/Yellow areas indicate regions where the AI detected abnormalities
                            </p>
                            <p className="text-xs text-orange-600 mt-1">
                              Confidence: {(xrayAnalysisResult.analysis.xray.top_prob * 100).toFixed(1)}% • Generated using Grad-CAM++
                            </p>
                          </div>
                        </div>
                      </div>
                    </motion.div>
                  )}
                </div>
              )}
            </div>

            {/* X-ray Analysis Results Section - Comprehensive Display */}
            {xrayAnalysisResult && xrayAnalysisResult.analysis && xrayAnalysisResult.analysis.xray && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="mt-6 space-y-6"
              >
                {/* Primary Findings Card */}
                <div className="bg-gradient-to-r from-blue-50 to-indigo-50 border border-blue-200 rounded-lg p-6">
                  <h3 className="font-bold text-blue-900 text-lg mb-4 flex items-center">
                    <Brain className="w-6 h-6 mr-2" />
                    🏥 Primary AI Analysis Results
                  </h3>
                  
                  {/* Top Finding Highlight */}
                  <div className="bg-white rounded-lg p-4 border border-blue-100 shadow-sm mb-4">
                    <div className="flex items-center justify-between mb-3">
                      <h4 className="text-lg font-semibold text-gray-800">🎯 Primary Finding</h4>
                      <div className="text-right">
                        <span className="text-2xl font-bold text-blue-600">
                          {(xrayAnalysisResult.analysis.xray.top_prob * 100).toFixed(1)}%
                        </span>
                        <p className="text-xs text-gray-500">Confidence</p>
                      </div>
                    </div>
                    <p className="text-xl font-bold text-gray-900 mb-2">
                      {xrayAnalysisResult.analysis.xray.top_label}
                    </p>
                    {xrayAnalysisResult.analysis.xray.top_prob > 0.5 ? (
                      <div className="flex items-center text-sm text-orange-600">
                        <AlertCircle className="w-4 h-4 mr-1" />
                        <span>Significant finding detected - requires medical review</span>
                      </div>
                    ) : (
                      <div className="flex items-center text-sm text-green-600">
                        <CheckCircle className="w-4 h-4 mr-1" />
                        <span>Low probability finding</span>
                      </div>
                    )}
                  </div>

                  {/* Detailed Pathology Breakdown */}
                  <div className="bg-white rounded-lg p-4 border border-blue-100 shadow-sm">
                    <h4 className="font-semibold text-gray-800 mb-3 flex items-center">
                      📊 Detailed Pathology Analysis
                      <span className="ml-2 text-xs bg-blue-100 text-blue-800 px-2 py-1 rounded">
                        Top 8 Findings
                      </span>
                    </h4>
                    <div className="space-y-3">
                      {xrayAnalysisResult.analysis.xray.pathologies.slice(0, 8).map(([pathology, probability], index) => (
                        <div key={index} className="flex items-center justify-between bg-gray-50 p-3 rounded-lg border">
                          <div className="flex items-center space-x-3">
                            <span className={`flex-shrink-0 w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold ${
                              index === 0 ? 'bg-red-100 text-red-800' :
                              index < 3 ? 'bg-orange-100 text-orange-800' :
                              'bg-gray-100 text-gray-800'
                            }`}>
                              {index + 1}
                            </span>
                            <span className="font-medium text-gray-700">{pathology}</span>
                          </div>
                          <div className="flex items-center space-x-3">
                            <div className="w-24 bg-gray-200 rounded-full h-2.5">
                              <div 
                                className={`h-2.5 rounded-full ${
                                  probability > 0.7 ? 'bg-red-500' :
                                  probability > 0.4 ? 'bg-orange-500' :
                                  probability > 0.2 ? 'bg-yellow-500' :
                                  'bg-green-500'
                                }`}
                                style={{ width: `${Math.max(probability * 100, 5)}%` }}
                              ></div>
                            </div>
                            <span className="text-sm font-semibold text-gray-600 w-12 text-right">
                              {(probability * 100).toFixed(1)}%
                            </span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
                
                {/* Ollama Clinical Analysis - Enhanced Section */}
                {xrayAnalysisResult.analysis.xray.clinical_explanation && 
                 !xrayAnalysisResult.analysis.xray.clinical_explanation.includes('❌') && (
                  <div className="bg-gradient-to-r from-purple-50 to-blue-50 border border-purple-200 rounded-lg p-6">
                    <h3 className="font-bold text-purple-900 text-lg mb-4 flex items-center">
                      <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="w-6 h-6 mr-2">
                        <path d="M12 8V4H8"/>
                        <rect width="16" height="12" x="4" y="8" rx="2"/>
                        <path d="M2 14h2"/>
                        <path d="M20 14h2"/>
                        <path d="M15 13v2"/>
                        <path d="M9 13v2"/>
                      </svg>
                      🤖 Clinical AI Interpretation
                    </h3>
                    <div className="bg-white rounded-lg p-5 border border-purple-100 shadow-sm">
                      <div className="prose prose-sm max-w-none">
                        <p className="text-gray-800 leading-relaxed whitespace-pre-wrap text-base">
                          {xrayAnalysisResult.analysis.xray.clinical_explanation}
                        </p>
                      </div>
                      <div className="flex items-center justify-between mt-4 pt-4 border-t border-purple-100">
                        <div className="flex items-center space-x-2">
                          <div className="w-2 h-2 bg-purple-500 rounded-full animate-pulse"></div>
                          <span className="text-sm text-purple-700 font-medium">
                            Powered by Ollama LLM (llama3.2:latest)
                          </span>
                        </div>
                        <span className="text-xs text-purple-600 bg-purple-100 px-2 py-1 rounded">
                          AI-Generated Clinical Interpretation
                        </span>
                      </div>
                    </div>
                  </div>
                )}
              </motion.div>
            )}

            {/* Upload Error */}
            {uploadError && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="alert alert-error"
              >
                <div className="flex items-center">
                  <AlertCircle className="w-4 h-4 mr-2" />
                  <span className="text-sm">{uploadError}</span>
                </div>
              </motion.div>
            )}

            {/* X-ray Analysis Info */}
            <div className="alert alert-success">
              <h3 className="font-medium mb-2">AI X-Ray Analysis Features</h3>
              <ul className="text-sm space-y-1">
                <li>• Automated chest X-ray interpretation</li>
                <li>• Detection of common abnormalities</li>
                <li>• Integration with vital signs analysis</li>
                <li>• Enhanced diagnostic accuracy</li>
              </ul>
            </div>

            {/* Medical Disclaimer for X-ray */}
            <div className="alert alert-warning">
              <div className="flex items-center mb-2">
                <AlertCircle className="w-4 h-4 mr-2" />
                <span className="text-sm font-medium">X-Ray Analysis Disclaimer</span>
              </div>
              <p className="text-xs">
                AI X-ray analysis is experimental and for educational purposes only.
                Always have medical images reviewed by qualified radiologists.
              </p>
            </div>
          </div>
        </motion.section>

        {/* Section 4: Analysis Results */}
        <motion.section
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
          className="card"
        >
          <h2 className="card-title flex items-center mb-6">
            <Brain className="w-6 h-6 mr-2 text-purple-600" />
            AI Diagnosis & Analysis Results
          </h2>

          {error && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="alert alert-error"
            >
              <div className="flex items-center">
                <AlertCircle className="w-5 h-5 mr-2" />
                <span>{error}</span>
              </div>
            </motion.div>
          )}

          {result && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="space-y-4"
            >
              {/* Final Diagnosis */}
              <div className="alert alert-info">
                <h3 className="font-semibold mb-2">Medical Diagnosis</h3>
                <p className="text-lg font-medium">{result.final_diagnosis}</p>
                <div className="flex items-center justify-between mt-2">
                  <p className="text-sm">
                    Confidence: {(result.confidence * 100).toFixed(1)}%
                  </p>
                  <span className={`badge ${
                    result.severity === 'CRITICAL' ? 'badge-critical' :
                    result.severity === 'HIGH' ? 'badge-high' :
                    result.severity === 'MODERATE' ? 'badge-moderate' :
                    'badge-low'
                  }`}>
                    {result.severity}
                  </span>
                </div>
              </div>

              {/* AI Details */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="p-3 bg-gray-50 rounded">
                  <h4 className="text-sm font-medium text-gray-700 mb-1">AI Model</h4>
                  <p className="text-sm text-gray-600">{result.model_used}</p>
                </div>
                <div className="p-3 bg-gray-50 rounded">
                  <h4 className="text-sm font-medium text-gray-700 mb-1">Agent Status</h4>
                  <p className="text-sm text-gray-600">{result.agent_agreement}</p>
                </div>
              </div>

              <div className="p-3 bg-gray-50 rounded">
                <h4 className="text-sm font-medium text-gray-700 mb-1">Medical Reasoning</h4>
                <p className="text-sm text-gray-600">{result.reasoning}</p>
              </div>

              {/* Recommendations */}
              <div>
                <h3 className="font-semibold text-gray-800 mb-2">Recommendations</h3>
                <ul className="space-y-1">
                  {result.recommendations.map((rec, index) => (
                    <li key={index} className="text-sm text-gray-700 flex items-start">
                      <span className="text-blue-500 mr-2">•</span>
                      {rec}
                    </li>
                  ))}
                </ul>
              </div>
            </motion.div>
          )}

          {!result && !error && !loading && (
            <div className="text-center text-gray-500 py-8">
              <Brain className="w-12 h-12 mx-auto mb-4 opacity-50" />
              <p>Enter vital signs above and click "Run BioGPT Medical Analysis"</p>
              <p className="text-xs mt-2 text-gray-400">
                Powered by Microsoft BioGPT Medical AI
              </p>
            </div>
          )}

          {/* Progress Indicator */}
          {result && (
            <div className="flex items-center justify-between pt-6 border-t border-gray-200">
              <span className="text-sm text-gray-600">Step 3 of 3: Analysis Complete</span>
              <div className="flex space-x-2">
                <div className="progress-dot progress-dot-complete"></div>
                <div className="progress-dot progress-dot-complete"></div>
                <div className="progress-dot progress-dot-complete"></div>
              </div>
            </div>
          )}
        </motion.section>

        {/* Medical Disclaimer */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="medical-disclaimer"
        >
          <div className="flex items-center justify-center mb-2">
            <AlertCircle className="w-4 h-4 mr-2 text-amber-600" />
            <span className="text-sm font-medium text-amber-800">Medical Disclaimer</span>
          </div>
          <p className="text-xs text-amber-700">
            This AI system is for educational purposes only. Always consult qualified healthcare professionals for medical advice.
          </p>
        </motion.div>
      </div>
    </div>
  );
};

export default VitalsApp;
