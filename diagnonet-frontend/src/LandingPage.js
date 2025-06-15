import React from 'react';
import { useNavigate } from 'react-router-dom';
import '../styles/LandingPage.css';

const LandingPage = () => {
  const navigate = useNavigate();

  const handleStartDiagnosis = () => {
    navigate('/patient-form');
  };

  return (
    <div className="landing-page">
      <div className="medical-grid"></div>
      
      {/* DNA Helix Decorations */}
      <div className="dna-helix" style={{ top: '10%', left: '5%' }}></div>
      <div className="dna-helix" style={{ top: '60%', right: '5%' }}></div>
      
      {/* Medical Icons */}
      <div className="medical-icons">
        <div className="medical-icon heartbeat" style={{ top: '20%', left: '15%' }}>
          <svg width="40" height="40" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M12 21.35l-1.45-1.32C5.4 15.36 2 12.28 2 8.5 2 5.42 4.42 3 7.5 3c1.74 0 3.41.81 4.5 2.09C13.09 3.81 14.76 3 16.5 3 19.58 3 22 5.42 22 8.5c0 3.78-3.4 6.86-8.55 11.54L12 21.35z" fill="#E61A4F" opacity="0.3"/>
          </svg>
        </div>
        
        <div className="medical-icon stethoscope" style={{ top: '70%', left: '80%' }}>
          <svg width="40" height="40" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <circle cx="18" cy="15" r="3" stroke="#FB6E92" strokeWidth="2" fill="none" opacity="0.4"/>
            <path d="M9 9V6a3 3 0 0 1 6 0v3" stroke="#FB6E92" strokeWidth="2" fill="none" opacity="0.4"/>
            <path d="M15 9v6a6 6 0 1 1-12 0V9" stroke="#FB6E92" strokeWidth="2" fill="none" opacity="0.4"/>
          </svg>
        </div>
        
        <div className="medical-icon brain" style={{ top: '40%', right: '20%' }}>
          <svg width="40" height="40" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M12 2C8.5 2 6 4.5 6 8c0 1.5.5 3 1.5 4C6.5 13 6 14.5 6 16c0 3.5 2.5 6 6 6s6-2.5 6-6c0-1.5-.5-3-1.5-4 1-1 1.5-2.5 1.5-4 0-3.5-2.5-6-6-6z" fill="#800080" opacity="0.3"/>
            <circle cx="9" cy="10" r="1" fill="#E61A4F"/>
            <circle cx="15" cy="10" r="1" fill="#E61A4F"/>
            <path d="M9 14 Q12 16 15 14" stroke="#FB6E92" strokeWidth="1.5" fill="none"/>
          </svg>
        </div>
      </div>
      
      {/* Main Content */}
      <div className="landing-content">
        <div className="hero-section fade-in">
          <h1 className="main-title pulse-animation">
            DiagnoNet
          </h1>
          <p className="tagline">
            AI ASSIST, NOT REPLACE
          </p>
          <p className="subtitle">
            Advanced Medical Diagnosis Platform powered by Intelligent Agents
          </p>
          
          <div className="features-grid">
            <div className="feature-card">
              <div className="feature-icon">
                <svg width="50" height="50" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z" fill="#E61A4F"/>
                </svg>
              </div>
              <h3>Vital Agent</h3>
              <p>Comprehensive vital signs analysis</p>
            </div>
            
            <div className="feature-card">
              <div className="feature-icon">
                <svg width="50" height="50" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z" fill="#FB6E92"/>
                </svg>
              </div>
              <h3>Symptom Agent</h3>
              <p>Intelligent symptom pattern recognition</p>
            </div>
            
            <div className="feature-card">
              <div className="feature-icon">
                <svg width="50" height="50" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <rect x="3" y="3" width="18" height="18" rx="2" ry="2" stroke="#800080" strokeWidth="2" fill="none"/>
                  <circle cx="9" cy="9" r="2" fill="#800080"/>
                  <path d="M21 15l-3.086-3.086a2 2 0 0 0-2.828 0L6 21" stroke="#800080" strokeWidth="2"/>
                </svg>
              </div>
              <h3>X-ray Agent</h3>
              <p>Advanced medical imaging analysis</p>
            </div>
          </div>
          
          <button 
            className="btn btn-primary cta-button"
            onClick={handleStartDiagnosis}
          >
            Start Diagnosis
          </button>
        </div>
      </div>
      
      {/* Floating Particles */}
      <div className="particles">
        <div className="particle" style={{ top: '15%', left: '10%', animationDelay: '0s' }}></div>
        <div className="particle" style={{ top: '25%', right: '15%', animationDelay: '1s' }}></div>
        <div className="particle" style={{ bottom: '20%', left: '20%', animationDelay: '2s' }}></div>
        <div className="particle" style={{ bottom: '30%', right: '10%', animationDelay: '3s' }}></div>
      </div>
    </div>
  );
};

export default LandingPage;
