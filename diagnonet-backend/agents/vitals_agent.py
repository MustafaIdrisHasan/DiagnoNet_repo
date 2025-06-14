import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class VitalsAgent:
    """
    Medical diagnosis agent that analyzes vital signs to predict potential conditions.
    Uses machine learning models to assess health risks based on vital measurements.
    """
    
    def __init__(self, model_path: str = None):
        self.model = None
        self.scaler = StandardScaler()
        self.conditions = [
            "Normal", "Hypertension", "Hypotension", "Tachycardia",
            "Bradycardia", "Fever", "Hypothermia", "Respiratory_Distress",
            "Sepsis", "Pneumonia", "Heart_Failure", "Dehydration"
        ]
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            self._create_medical_model()
    
    def _create_medical_model(self):
        """Create a sophisticated medical model based on clinical data patterns"""
        np.random.seed(42)
        n_samples = 5000  # Larger dataset for better training

        # Generate realistic medical data with correlations
        X, y = self._generate_realistic_medical_data(n_samples)

        # Split data for validation
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Scale features
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Use ensemble of models for better accuracy
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced'  # Handle class imbalance
        )

        # Train the model
        self.model.fit(X_train_scaled, y_train)

        # Evaluate model performance
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model trained with accuracy: {accuracy:.3f}")

        # Print feature importance
        feature_names = ['systolic_bp', 'diastolic_bp', 'heart_rate', 'temperature', 'respiratory_rate', 'oxygen_saturation']
        importances = self.model.feature_importances_
        for name, importance in zip(feature_names, importances):
            print(f"{name}: {importance:.3f}")

    def _generate_realistic_medical_data(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate realistic medical data with proper correlations and distributions"""
        X = np.zeros((n_samples, 6))
        y = np.zeros(n_samples, dtype=int)

        # Define condition-specific patterns
        conditions_data = {
            0: {  # Normal
                'weight': 0.4,
                'systolic_bp': (110, 130, 10),
                'diastolic_bp': (70, 85, 8),
                'heart_rate': (65, 85, 10),
                'temperature': (98.0, 99.0, 0.5),
                'respiratory_rate': (14, 18, 2),
                'oxygen_saturation': (97, 100, 1)
            },
            1: {  # Hypertension
                'weight': 0.15,
                'systolic_bp': (140, 180, 15),
                'diastolic_bp': (90, 110, 10),
                'heart_rate': (70, 95, 12),
                'temperature': (98.0, 99.2, 0.6),
                'respiratory_rate': (14, 20, 3),
                'oxygen_saturation': (96, 100, 2)
            },
            2: {  # Hypotension
                'weight': 0.08,
                'systolic_bp': (70, 90, 8),
                'diastolic_bp': (45, 60, 6),
                'heart_rate': (85, 120, 15),
                'temperature': (97.5, 99.0, 0.8),
                'respiratory_rate': (16, 22, 3),
                'oxygen_saturation': (94, 98, 2)
            },
            3: {  # Tachycardia
                'weight': 0.1,
                'systolic_bp': (100, 140, 15),
                'diastolic_bp': (65, 90, 10),
                'heart_rate': (100, 150, 20),
                'temperature': (98.5, 100.5, 1),
                'respiratory_rate': (18, 25, 4),
                'oxygen_saturation': (95, 99, 2)
            },
            4: {  # Bradycardia
                'weight': 0.06,
                'systolic_bp': (100, 130, 12),
                'diastolic_bp': (60, 80, 8),
                'heart_rate': (35, 60, 8),
                'temperature': (97.8, 98.8, 0.4),
                'respiratory_rate': (12, 16, 2),
                'oxygen_saturation': (96, 100, 2)
            },
            5: {  # Fever
                'weight': 0.12,
                'systolic_bp': (105, 135, 12),
                'diastolic_bp': (65, 85, 8),
                'heart_rate': (85, 120, 15),
                'temperature': (100.5, 104.0, 1.2),
                'respiratory_rate': (18, 28, 4),
                'oxygen_saturation': (94, 98, 2)
            },
            6: {  # Hypothermia
                'weight': 0.03,
                'systolic_bp': (85, 115, 10),
                'diastolic_bp': (50, 70, 8),
                'heart_rate': (45, 80, 12),
                'temperature': (92.0, 96.5, 1.5),
                'respiratory_rate': (10, 16, 3),
                'oxygen_saturation': (90, 96, 3)
            },
            7: {  # Respiratory_Distress
                'weight': 0.1,
                'systolic_bp': (95, 125, 12),
                'diastolic_bp': (60, 80, 8),
                'heart_rate': (90, 130, 18),
                'temperature': (98.5, 101.0, 1),
                'respiratory_rate': (22, 35, 5),
                'oxygen_saturation': (80, 94, 4)
            },
            8: {  # Sepsis
                'weight': 0.05,
                'systolic_bp': (70, 100, 12),
                'diastolic_bp': (40, 65, 8),
                'heart_rate': (110, 160, 20),
                'temperature': (101.0, 105.0, 1.5),
                'respiratory_rate': (25, 40, 6),
                'oxygen_saturation': (85, 93, 4)
            },
            9: {  # Pneumonia
                'weight': 0.08,
                'systolic_bp': (100, 130, 12),
                'diastolic_bp': (65, 85, 8),
                'heart_rate': (95, 125, 15),
                'temperature': (100.8, 103.5, 1.2),
                'respiratory_rate': (24, 32, 4),
                'oxygen_saturation': (88, 94, 3)
            },
            10: {  # Heart_Failure
                'weight': 0.06,
                'systolic_bp': (85, 120, 15),
                'diastolic_bp': (55, 80, 10),
                'heart_rate': (80, 110, 12),
                'temperature': (98.0, 99.5, 0.6),
                'respiratory_rate': (20, 30, 4),
                'oxygen_saturation': (90, 96, 3)
            },
            11: {  # Dehydration
                'weight': 0.07,
                'systolic_bp': (90, 115, 10),
                'diastolic_bp': (55, 75, 8),
                'heart_rate': (95, 130, 15),
                'temperature': (98.5, 100.0, 0.8),
                'respiratory_rate': (16, 24, 3),
                'oxygen_saturation': (94, 98, 2)
            }
        }

        # Generate samples for each condition
        sample_idx = 0
        for condition_id, params in conditions_data.items():
            n_condition_samples = int(n_samples * params['weight'])

            for i in range(n_condition_samples):
                if sample_idx >= n_samples:
                    break

                # Generate correlated vital signs
                X[sample_idx, 0] = np.random.normal(params['systolic_bp'][0] +
                                                  (params['systolic_bp'][1] - params['systolic_bp'][0]) * np.random.random(),
                                                  params['systolic_bp'][2])
                X[sample_idx, 1] = np.random.normal(params['diastolic_bp'][0] +
                                                  (params['diastolic_bp'][1] - params['diastolic_bp'][0]) * np.random.random(),
                                                  params['diastolic_bp'][2])
                X[sample_idx, 2] = np.random.normal(params['heart_rate'][0] +
                                                  (params['heart_rate'][1] - params['heart_rate'][0]) * np.random.random(),
                                                  params['heart_rate'][2])
                X[sample_idx, 3] = np.random.normal(params['temperature'][0] +
                                                  (params['temperature'][1] - params['temperature'][0]) * np.random.random(),
                                                  params['temperature'][2])
                X[sample_idx, 4] = np.random.normal(params['respiratory_rate'][0] +
                                                  (params['respiratory_rate'][1] - params['respiratory_rate'][0]) * np.random.random(),
                                                  params['respiratory_rate'][2])
                X[sample_idx, 5] = np.random.normal(params['oxygen_saturation'][0] +
                                                  (params['oxygen_saturation'][1] - params['oxygen_saturation'][0]) * np.random.random(),
                                                  params['oxygen_saturation'][2])

                # Ensure realistic bounds
                X[sample_idx, 0] = np.clip(X[sample_idx, 0], 60, 250)   # systolic BP
                X[sample_idx, 1] = np.clip(X[sample_idx, 1], 30, 150)   # diastolic BP
                X[sample_idx, 2] = np.clip(X[sample_idx, 2], 30, 200)   # heart rate
                X[sample_idx, 3] = np.clip(X[sample_idx, 3], 90, 110)   # temperature
                X[sample_idx, 4] = np.clip(X[sample_idx, 4], 5, 50)     # respiratory rate
                X[sample_idx, 5] = np.clip(X[sample_idx, 5], 70, 100)   # oxygen saturation

                y[sample_idx] = condition_id
                sample_idx += 1

        return X[:sample_idx], y[:sample_idx]

    def _generate_labels(self, X: np.ndarray) -> np.ndarray:
        """Generate labels based on vital sign thresholds"""
        labels = []
        
        for row in X:
            systolic, diastolic, hr, temp, rr, spo2 = row
            
            # Priority-based labeling (most severe condition wins)
            if temp > 100.4:  # Fever
                labels.append(5)
            elif temp < 95:  # Hypothermia
                labels.append(6)
            elif systolic > 140 or diastolic > 90:  # Hypertension
                labels.append(1)
            elif systolic < 90 or diastolic < 60:  # Hypotension
                labels.append(2)
            elif hr > 100:  # Tachycardia
                labels.append(3)
            elif hr < 60:  # Bradycardia
                labels.append(4)
            elif rr > 20 or spo2 < 95:  # Respiratory distress
                labels.append(7)
            else:  # Normal
                labels.append(0)
                
        return np.array(labels)
    
    def analyze_vitals(self, vitals: Dict[str, float]) -> Dict:
        """
        Analyze vital signs and return diagnosis with confidence scores
        
        Args:
            vitals: Dictionary containing vital measurements
                   Expected keys: systolic_bp, diastolic_bp, heart_rate, 
                                temperature, respiratory_rate, oxygen_saturation
        
        Returns:
            Dictionary with diagnosis, confidence, and recommendations
        """
        if not self.model:
            raise ValueError("Model not loaded or trained")
        
        # Extract features in correct order
        features = [
            vitals.get('systolic_bp', 120),
            vitals.get('diastolic_bp', 80),
            vitals.get('heart_rate', 70),
            vitals.get('temperature', 98.6),
            vitals.get('respiratory_rate', 16),
            vitals.get('oxygen_saturation', 98)
        ]
        
        # Scale features
        features_scaled = self.scaler.transform([features])
        
        # Get prediction and probabilities
        prediction = self.model.predict(features_scaled)[0]
        probabilities = self.model.predict_proba(features_scaled)[0]
        
        # Get top 3 predictions
        top_indices = np.argsort(probabilities)[::-1][:3]
        
        results = {
            'primary_diagnosis': self.conditions[prediction],
            'confidence': float(probabilities[prediction]),
            'top_conditions': [
                {
                    'condition': self.conditions[idx],
                    'probability': float(probabilities[idx])
                }
                for idx in top_indices
            ],
            'vital_analysis': self._analyze_individual_vitals(vitals),
            'recommendations': self._get_recommendations(prediction, vitals)
        }
        
        return results
    
    def _analyze_individual_vitals(self, vitals: Dict[str, float]) -> Dict:
        """Analyze each vital sign individually"""
        analysis = {}
        
        # Blood pressure analysis
        systolic = vitals.get('systolic_bp', 120)
        diastolic = vitals.get('diastolic_bp', 80)
        
        if systolic > 140 or diastolic > 90:
            bp_status = "High (Hypertensive)"
        elif systolic < 90 or diastolic < 60:
            bp_status = "Low (Hypotensive)"
        else:
            bp_status = "Normal"
        
        analysis['blood_pressure'] = {
            'status': bp_status,
            'systolic': systolic,
            'diastolic': diastolic
        }
        
        # Heart rate analysis
        hr = vitals.get('heart_rate', 70)
        if hr > 100:
            hr_status = "High (Tachycardia)"
        elif hr < 60:
            hr_status = "Low (Bradycardia)"
        else:
            hr_status = "Normal"
        
        analysis['heart_rate'] = {
            'status': hr_status,
            'value': hr
        }
        
        # Temperature analysis
        temp = vitals.get('temperature', 98.6)
        if temp > 100.4:
            temp_status = "High (Fever)"
        elif temp < 95:
            temp_status = "Low (Hypothermia)"
        else:
            temp_status = "Normal"
        
        analysis['temperature'] = {
            'status': temp_status,
            'value': temp
        }
        
        return analysis
    
    def _get_recommendations(self, prediction: int, vitals: Dict[str, float]) -> List[str]:
        """Get comprehensive medical recommendations based on diagnosis"""
        recommendations = []

        condition = self.conditions[prediction]

        if condition == "Normal":
            recommendations.extend([
                "Continue maintaining healthy lifestyle",
                "Regular health check-ups recommended",
                "Maintain balanced diet and regular exercise"
            ])
        elif condition == "Hypertension":
            recommendations.extend([
                "🚨 URGENT: Monitor blood pressure closely",
                "Reduce sodium intake (<2300mg/day)",
                "Consult cardiologist within 24-48 hours",
                "Avoid strenuous activity until evaluated",
                "Consider DASH diet implementation"
            ])
        elif condition == "Hypotension":
            recommendations.extend([
                "⚠️ Increase fluid intake gradually",
                "Monitor for dizziness, fainting, or weakness",
                "Avoid sudden position changes",
                "Consider compression stockings",
                "Seek medical evaluation if symptoms persist"
            ])
        elif condition == "Tachycardia":
            recommendations.extend([
                "🚨 Monitor heart rate continuously",
                "Avoid caffeine, alcohol, and stimulants",
                "Practice deep breathing and relaxation",
                "Seek immediate medical attention if chest pain occurs",
                "Consider cardiac evaluation"
            ])
        elif condition == "Bradycardia":
            recommendations.extend([
                "⚠️ Monitor for fatigue and dizziness",
                "Avoid medications that slow heart rate",
                "Seek cardiology consultation",
                "Monitor exercise tolerance",
                "Consider pacemaker evaluation if symptomatic"
            ])
        elif condition == "Fever":
            recommendations.extend([
                "🌡️ Stay well hydrated (8-10 glasses water/day)",
                "Rest and avoid strenuous activity",
                "Monitor temperature every 2-4 hours",
                "Consider fever reducers (acetaminophen/ibuprofen)",
                "Seek medical care if fever >103°F or persists >3 days"
            ])
        elif condition == "Hypothermia":
            recommendations.extend([
                "🚨 EMERGENCY: Seek immediate medical attention",
                "Move to warm environment immediately",
                "Remove wet clothing, wrap in blankets",
                "Avoid direct heat application",
                "Monitor consciousness and breathing"
            ])
        elif condition == "Respiratory_Distress":
            recommendations.extend([
                "🚨 URGENT: Seek immediate medical evaluation",
                "Sit upright to ease breathing",
                "Use supplemental oxygen if available",
                "Monitor oxygen saturation closely",
                "Avoid lying flat, consider emergency services"
            ])
        elif condition == "Sepsis":
            recommendations.extend([
                "🚨 EMERGENCY: Call 911 immediately",
                "This is a life-threatening condition",
                "Requires immediate hospital treatment",
                "Monitor for confusion, rapid breathing",
                "Time-sensitive - every hour matters"
            ])
        elif condition == "Pneumonia":
            recommendations.extend([
                "🚨 URGENT: Seek medical attention within hours",
                "Rest and increase fluid intake",
                "Monitor breathing and oxygen levels",
                "Avoid smoking and secondhand smoke",
                "May require antibiotics and hospitalization"
            ])
        elif condition == "Heart_Failure":
            recommendations.extend([
                "🚨 URGENT: Cardiology consultation required",
                "Monitor daily weight (report 2+ lb gain)",
                "Limit sodium intake (<2000mg/day)",
                "Elevate legs when resting",
                "Take medications as prescribed, monitor symptoms"
            ])
        elif condition == "Dehydration":
            recommendations.extend([
                "⚠️ Increase fluid intake immediately",
                "Use oral rehydration solutions",
                "Monitor urine color (should be pale yellow)",
                "Avoid alcohol and caffeine",
                "Seek medical care if unable to keep fluids down"
            ])
        else:
            recommendations.extend([
                "⚠️ Abnormal vital signs detected",
                "Consult healthcare provider for evaluation",
                "Monitor symptoms closely",
                "Seek emergency care if condition worsens"
            ])

        # Add severity-based recommendations
        severity = self._assess_severity(vitals)
        if severity == "CRITICAL":
            recommendations.insert(0, "🚨 CRITICAL: Call emergency services (911) immediately")
        elif severity == "URGENT":
            recommendations.insert(0, "⚠️ URGENT: Seek medical attention within 2-4 hours")

        return recommendations

    def _assess_severity(self, vitals: Dict[str, float]) -> str:
        """Assess overall severity based on vital signs"""
        critical_flags = 0
        urgent_flags = 0

        # Critical thresholds
        if vitals.get('systolic_bp', 120) > 180 or vitals.get('systolic_bp', 120) < 70:
            critical_flags += 1
        if vitals.get('heart_rate', 70) > 150 or vitals.get('heart_rate', 70) < 40:
            critical_flags += 1
        if vitals.get('temperature', 98.6) > 104 or vitals.get('temperature', 98.6) < 95:
            critical_flags += 1
        if vitals.get('oxygen_saturation', 98) < 85:
            critical_flags += 1
        if vitals.get('respiratory_rate', 16) > 35 or vitals.get('respiratory_rate', 16) < 8:
            critical_flags += 1

        # Urgent thresholds
        if vitals.get('systolic_bp', 120) > 160 or vitals.get('systolic_bp', 120) < 90:
            urgent_flags += 1
        if vitals.get('heart_rate', 70) > 120 or vitals.get('heart_rate', 70) < 50:
            urgent_flags += 1
        if vitals.get('temperature', 98.6) > 102 or vitals.get('temperature', 98.6) < 96:
            urgent_flags += 1
        if vitals.get('oxygen_saturation', 98) < 92:
            urgent_flags += 1
        if vitals.get('respiratory_rate', 16) > 25 or vitals.get('respiratory_rate', 16) < 10:
            urgent_flags += 1

        if critical_flags >= 1:
            return "CRITICAL"
        elif urgent_flags >= 2:
            return "URGENT"
        elif urgent_flags >= 1:
            return "MODERATE"
        else:
            return "STABLE"
    
    def save_model(self, path: str):
        """Save the trained model and scaler"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'conditions': self.conditions
        }
        joblib.dump(model_data, path)
    
    def load_model(self, path: str):
        """Load a trained model and scaler"""
        model_data = joblib.load(path)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.conditions = model_data['conditions']
