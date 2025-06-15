import React, { useState } from 'react';

const VitalsApp = () => {
  const [xrayFile, setXrayFile] = useState(null);
  const [uploadError, setUploadError] = useState(null);
  const [xrayAnalysisResult, setXrayAnalysisResult] = useState(null);
  const [xrayAnalyzing, setXrayAnalyzing] = useState(false);

  const handleXRayUpload = async (event) => {
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
    reader.onload = async (e) => {
      const xrayFileData = {
        file,
        preview: e.target?.result,
        name: file.name,
        size: file.size
      };
      setXrayFile(xrayFileData);

      // Automatically analyze the X-ray
      setXrayAnalyzing(true);
      try {
        const formData = new FormData();
        formData.append('xray_image', file);

        const response = await fetch('http://localhost:8000/analyze-xray', {
          method: 'POST',
          body: formData,
        });

        if (!response.ok) {
          const errorData = await response.json();
          throw new Error(errorData.detail || 'Failed to analyze X-ray');
        }

        const analysisData = await response.json();
        setXrayAnalysisResult(analysisData);
      } catch (err) {
        setUploadError(err instanceof Error ? err.message : 'X-ray analysis failed');
      } finally {
        setXrayAnalyzing(false);
      }
    };
    reader.readAsDataURL(file);
  };

  const removeXRayFile = () => {
    setXrayFile(null);
    setUploadError(null);
    setXrayAnalysisResult(null);
  };

  return (
    <div style={{ minHeight: '100vh', background: 'linear-gradient(to bottom right, #e3f2fd, #f3e5f5)', padding: '20px' }}>
      {/* Header */}
      <div style={{ textAlign: 'center', paddingBottom: '20px' }}>
        <h1 style={{ fontSize: '2.5rem', fontWeight: 'bold', color: '#333', marginBottom: '10px' }}>
          🧠 DiagnoNet
        </h1>
        <p style={{ color: '#666', fontSize: '1.1rem' }}>AI-Powered Medical Analysis</p>
        <p style={{ color: '#888', fontSize: '0.9rem' }}>Advanced chest X-ray diagnosis with Ollama explanations</p>
      </div>

      <div style={{ maxWidth: '800px', margin: '0 auto' }}>
        {/* X-Ray Upload Section */}
        <div style={{ 
          backgroundColor: 'white', 
          padding: '30px', 
          borderRadius: '12px', 
          boxShadow: '0 4px 6px rgba(0,0,0,0.1)', 
          marginBottom: '20px' 
        }}>
          <h2 style={{ fontSize: '1.5rem', fontWeight: '600', marginBottom: '20px', color: '#333' }}>
            📸 X-Ray Analysis
          </h2>

          {/* Upload Area */}
          <div style={{ 
            border: '2px dashed #ccc', 
            borderRadius: '8px', 
            padding: '40px', 
            textAlign: 'center',
            backgroundColor: '#fafafa'
          }}>
            {!xrayFile ? (
              <div>
                <div style={{ fontSize: '3rem', marginBottom: '15px' }}>📤</div>
                <p style={{ fontSize: '1.1rem', fontWeight: '500', color: '#333', marginBottom: '10px' }}>
                  Upload Chest X-Ray Image
                </p>
                <p style={{ fontSize: '0.9rem', color: '#666', marginBottom: '20px' }}>
                  Upload a chest X-ray for AI analysis with Ollama explanations
                </p>
                <input
                  type="file"
                  accept=".png,.jpg,.jpeg,image/png,image/jpeg"
                  onChange={handleXRayUpload}
                  style={{ display: 'none' }}
                  id="xray-upload"
                />
                <label
                  htmlFor="xray-upload"
                  style={{
                    display: 'inline-block',
                    padding: '12px 24px',
                    backgroundColor: '#007bff',
                    color: 'white',
                    borderRadius: '6px',
                    cursor: 'pointer',
                    fontSize: '1rem',
                    fontWeight: '500'
                  }}
                >
                  📤 Choose X-Ray Image
                </label>
                <p style={{ fontSize: '0.8rem', color: '#999', marginTop: '10px' }}>
                  Supports PNG, JPG, JPEG • Max 10MB
                </p>
              </div>
            ) : (
              <div>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '15px' }}>
                  <div style={{ display: 'flex', alignItems: 'center' }}>
                    <span style={{ color: 'green', fontSize: '1.5rem', marginRight: '10px' }}>✅</span>
                    <div>
                      <p style={{ fontWeight: '500', color: '#333', margin: 0 }}>{xrayFile.name}</p>
                      <p style={{ fontSize: '0.9rem', color: '#666', margin: 0 }}>
                        {(xrayFile.size / 1024 / 1024).toFixed(1)}MB
                      </p>
                    </div>
                  </div>
                  <button
                    onClick={removeXRayFile}
                    style={{
                      background: 'none',
                      border: 'none',
                      fontSize: '1.5rem',
                      cursor: 'pointer',
                      color: '#dc3545'
                    }}
                  >
                    ❌
                  </button>
                </div>

                {/* Image Preview */}
                <div style={{ marginBottom: '15px' }}>
                  <img
                    src={xrayFile.preview}
                    alt="X-ray preview"
                    style={{ maxWidth: '300px', maxHeight: '300px', borderRadius: '8px', border: '1px solid #ddd' }}
                  />
                </div>

                {/* Analysis Status */}
                {xrayAnalyzing ? (
                  <div style={{ color: '#007bff', fontSize: '1rem' }}>
                    <span>🔄</span> Analyzing X-ray image...
                  </div>
                ) : xrayAnalysisResult ? (
                  <div style={{ color: 'green', fontSize: '1rem' }}>
                    <span>✅</span> X-ray analysis completed
                  </div>
                ) : (
                  <div style={{ color: 'green', fontSize: '1rem' }}>
                    <span>✅</span> X-ray image ready for analysis
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Upload Error */}
          {uploadError && (
            <div style={{ 
              backgroundColor: '#ffe6e6', 
              border: '1px solid #ff9999', 
              padding: '10px', 
              borderRadius: '6px', 
              marginTop: '15px',
              color: '#cc0000'
            }}>
              <span>⚠️</span> {uploadError}
            </div>
          )}
        </div>

        {/* X-ray Analysis Results */}
        {xrayAnalysisResult && xrayAnalysisResult.analysis.xray && (
          <div style={{ 
            backgroundColor: '#e3f2fd', 
            border: '1px solid #90caf9', 
            padding: '20px', 
            borderRadius: '12px',
            marginTop: '20px'
          }}>
            <h3 style={{ color: '#1565c0', marginBottom: '15px', fontSize: '1.3rem' }}>
              🧠 X-Ray AI Analysis Results
            </h3>
            
            {/* Top Finding */}
            <div style={{ marginBottom: '15px' }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px' }}>
                <h4 style={{ fontWeight: '500', color: '#333', margin: 0 }}>Primary Finding</h4>
                <span style={{ 
                  fontSize: '0.9rem', 
                  backgroundColor: '#bbdefb', 
                  color: '#1565c0', 
                  padding: '4px 8px', 
                  borderRadius: '4px' 
                }}>
                  {(xrayAnalysisResult.analysis.xray.top_prob * 100).toFixed(1)}% confidence
                </span>
              </div>
              <p style={{ fontSize: '1.2rem', fontWeight: '600', color: '#333', margin: 0 }}>
                {xrayAnalysisResult.analysis.xray.top_label}
              </p>
            </div>

            {/* Clinical Explanation */}
            {xrayAnalysisResult.analysis.xray.clinical_explanation && (
              <div style={{ marginBottom: '15px' }}>
                <h4 style={{ fontWeight: '500', color: '#333', marginBottom: '8px' }}>Clinical Interpretation</h4>
                <div style={{ 
                  backgroundColor: 'white', 
                  padding: '12px', 
                  borderRadius: '6px', 
                  border: '1px solid #e3f2fd' 
                }}>
                  <p style={{ color: '#333', fontSize: '0.95rem', lineHeight: '1.5', margin: 0 }}>
                    {xrayAnalysisResult.analysis.xray.clinical_explanation}
                  </p>
                </div>
              </div>
            )}

            {/* Top Pathologies */}
            <div style={{ marginBottom: '15px' }}>
              <h4 style={{ fontWeight: '500', color: '#333', marginBottom: '10px' }}>Detected Pathologies</h4>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                {xrayAnalysisResult.analysis.xray.pathologies.slice(0, 5).map(([pathology, probability], index) => (
                  <div key={index} style={{ 
                    display: 'flex', 
                    alignItems: 'center', 
                    justifyContent: 'space-between', 
                    backgroundColor: 'white', 
                    padding: '8px 12px', 
                    borderRadius: '6px',
                    border: '1px solid #e3f2fd'
                  }}>
                    <span style={{ fontSize: '0.9rem', color: '#333' }}>{pathology}</span>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <div style={{ 
                        width: '80px', 
                        height: '8px', 
                        backgroundColor: '#e0e0e0', 
                        borderRadius: '4px',
                        overflow: 'hidden'
                      }}>
                        <div 
                          style={{ 
                            width: `${probability * 100}%`, 
                            height: '100%', 
                            backgroundColor: '#1976d2',
                            borderRadius: '4px'
                          }}
                        ></div>
                      </div>
                      <span style={{ fontSize: '0.8rem', color: '#666', minWidth: '40px' }}>
                        {(probability * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Grad-CAM Visualization */}
            {xrayAnalysisResult.analysis.xray.gradcam_b64 && (
              <div style={{ marginBottom: '15px' }}>
                <h4 style={{ fontWeight: '500', color: '#333', marginBottom: '10px' }}>Heat Map Visualization</h4>
                <div style={{ 
                  backgroundColor: 'white', 
                  padding: '10px', 
                  borderRadius: '6px',
                  border: '1px solid #e3f2fd',
                  textAlign: 'center'
                }}>
                  <img 
                    src={`data:image/png;base64,${xrayAnalysisResult.analysis.xray.gradcam_b64}`}
                    alt="Grad-CAM heat map"
                    style={{ maxWidth: '400px', width: '100%', borderRadius: '6px' }}
                  />
                  <p style={{ fontSize: '0.8rem', color: '#666', marginTop: '8px', margin: 0 }}>
                    Heat map showing areas of interest for the AI analysis
                  </p>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Info Section */}
        <div style={{ 
          backgroundColor: '#fff3cd', 
          border: '1px solid #ffeaa7', 
          padding: '15px', 
          borderRadius: '8px', 
          marginTop: '20px',
          textAlign: 'center'
        }}>
          <p style={{ fontSize: '0.9rem', color: '#856404', margin: 0 }}>
            <strong>⚕️ Medical Disclaimer:</strong> This AI analysis is for educational purposes only. 
            Always consult qualified medical professionals for diagnosis and treatment.
          </p>
        </div>
      </div>
    </div>
  );
};

export default VitalsApp; 