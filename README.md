# DiagnoNet - Advanced Medical AI Diagnosis System

🏥 **BioGPT-powered multi-agent medical analysis platform**

## 🚀 Features

- **Multi-Agent Architecture**: Specialized AI agents for different medical domains
- **BioGPT Integration**: Microsoft's medical language model for advanced reasoning
- **X-Ray Analysis**: Computer vision for chest X-ray interpretation
- **Vital Signs Analysis**: ML-powered vital signs assessment with SHAP explanations
- **React.js Frontend**: Beautiful, responsive medical interface
- **Medical-Grade Validation**: Physiologically accurate input validation

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    DIAGNONET MEDICAL AI SYSTEM                 │
├─────────────────────────────────────────────────────────────────┤
│  Frontend (React.js)  │  API Gateway  │  Multi-Agent System     │
│  ┌─────────────────┐  │  ┌──────────┐ │  ┌────────────────────┐ │
│  │ Patient Forms   │──┼──│ FastAPI  │─┼──│ Vitals Agent       │ │
│  │ X-ray Upload    │  │  │ Routes   │ │  │ X-ray Agent        │ │
│  │ Results Display │  │  │ CORS     │ │  │ Supervisor Agent   │ │
│  └─────────────────┘  │  └──────────┘ │  └────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## 🛠️ Technology Stack

### Backend
- **FastAPI**: High-performance Python web framework
- **BioGPT**: Microsoft's medical language model
- **scikit-learn**: Machine learning with Random Forest
- **SHAP**: Explainable AI for medical interpretability
- **Pillow**: Medical image processing
- **PyTorch**: Deep learning framework

### Frontend
- **React.js 19**: Modern UI framework
- **TypeScript**: Type-safe development
- **Framer Motion**: Smooth animations
- **Lucide React**: Medical icons
- **Custom CSS**: Tailwind-inspired design system

## 📁 Project Structure

```
diagnonet/
├── README.md                   # This file
├── .gitignore                  # Git ignore rules
├── diagnonet-backend/
│   ├── main.py                 # FastAPI application entry point
│   ├── requirements.txt        # Python dependencies
│   ├── api/
│   │   └── routes.py          # API endpoints and medical validation
│   └── agents/
│       ├── vitals_agent.py    # ML-powered vital signs analysis
│       ├── xray_agent.py      # Medical image processing
│       └── supervisor_agent.py # BioGPT orchestration
└── diagnonet-frontend/
    ├── package.json           # Node.js dependencies
    └── src/
        ├── App.tsx           # Main React application
        ├── VitalsApp.tsx     # Medical interface component
        └── index.css         # Complete design system
```

## 🚀 Quick Start

### Backend Setup
```bash
cd diagnonet-backend
pip install -r requirements.txt
python main.py
```

### Frontend Setup
```bash
cd diagnonet-frontend
npm install
npm start
```

## 🏥 Medical AI Agents

### 1. Vitals Agent
- **Random Forest Classifier** with hyperparameter tuning
- **SHAP Explanations** for medical interpretability
- **Feature Engineering**: MAP, Shock Index calculations
- **Medical Validation**: Physiological range checking

### 2. X-Ray Agent
- **Image Processing Pipeline**: 512x512 standardization
- **Feature Extraction**: Symmetry, lung clarity, contrast analysis
- **Medical Findings**: Automated abnormality detection
- **Quality Assessment**: Technical image evaluation

### 3. Supervisor Agent
- **BioGPT Integration**: Microsoft's medical language model
- **Multi-Agent Fusion**: Combines all analysis results
- **Medical Reasoning**: Advanced AI-powered diagnosis
- **Confidence Scoring**: Reliable medical assessments

## 📊 Key Features

- **Medical-Grade Validation**: All inputs validated against physiological ranges
- **Explainable AI**: SHAP values show which vitals contributed to diagnosis
- **Multi-Modal Analysis**: Combines vital signs and X-ray imaging
- **Real-Time Processing**: Fast API responses for clinical workflows
- **Beautiful UI**: Professional medical interface with animations
- **Error Recovery**: Graceful degradation when components fail

## 🔬 Medical Conditions Detected

- Hypertension / Hypotension
- Tachycardia / Bradycardia  
- Fever / Hypothermia
- Respiratory Distress
- Sepsis / Pneumonia
- Heart Failure / Dehydration
- X-ray Abnormalities

## ⚠️ Medical Disclaimer

This AI system is for educational and research purposes only. Always consult qualified healthcare professionals for medical advice. Do not use for actual medical diagnosis without proper clinical validation.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- Microsoft BioGPT team for the medical language model
- scikit-learn community for machine learning tools
- React.js team for the frontend framework
- Medical AI research community

---

**Built with ❤️ for advancing medical AI technology**
