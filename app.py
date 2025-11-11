import streamlit as st
import requests
import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import io
import base64
import os
import pdfplumber
import docx
import tempfile
from datetime import datetime
import logging
import re
import time
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure the page for cloud deployment
st.set_page_config(
    page_title="MediAI Pro - Advanced Medical Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional medical UI
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2e86ab;
        text-align: center;
        margin-bottom: 2rem;
    }
    .medical-card {
        background-color: #f8f9fa;
        border-radius: 15px;
        padding: 1.5rem;
        border-left: 5px solid #1f77b4;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .urgent-card {
        background-color: #ffe6e6;
        border-left: 5px solid #dc3545;
    }
    .normal-card {
        background-color: #e6ffe6;
        border-left: 5px solid #28a745;
    }
    .chat-container {
        background-color: #f0f2f6;
        border-radius: 15px;
        padding: 1rem;
        margin-bottom: 1rem;
        max-height: 500px;
        overflow-y: auto;
    }
    .user-message {
        background-color: #007bff;
        color: white;
        padding: 1rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        max-width: 80%;
        margin-left: auto;
        word-wrap: break-word;
    }
    .assistant-message {
        background-color: #ffffff;
        color: #333;
        padding: 1rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        max-width: 80%;
        margin-right: auto;
        border: 1px solid #e0e0e0;
        word-wrap: break-word;
    }
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .quick-action-btn {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem;
        margin: 0.2rem;
        width: 100%;
        text-align: center;
        cursor: pointer;
    }
    .api-status-connected {
        background-color: #d4edda;
        color: #155724;
        padding: 0.5rem;
        border-radius: 5px;
        border: 1px solid #c3e6cb;
    }
    .api-status-disconnected {
        background-color: #f8d7da;
        color: #721c24;
        padding: 0.5rem;
        border-radius: 5px;
        border: 1px solid #f5c6cb;
    }
    .rate-limit-warning {
        background-color: #fff3cd;
        color: #856404;
        padding: 0.5rem;
        border-radius: 5px;
        border: 1px solid #ffeaa7;
        margin: 0.5rem 0;
    }
    .emergency-alert {
        background-color: #ffcccc;
        color: #cc0000;
        padding: 1rem;
        border-radius: 10px;
        border: 2px solid #ff6666;
        margin: 1rem 0;
        font-weight: bold;
    }
    .vital-signs {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'patient_data' not in st.session_state:
    st.session_state.patient_data = {}
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {}
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'api_configured' not in st.session_state:
    st.session_state.api_configured = False
if 'analyze_clicked' not in st.session_state:
    st.session_state.analyze_clicked = False
if 'api_key' not in st.session_state:
    st.session_state.api_key = None
if 'patient_context' not in st.session_state:
    st.session_state.patient_context = ""
if 'processed_lab_data' not in st.session_state:
    st.session_state.processed_lab_data = ""
if 'rate_limit_warning' not in st.session_state:
    st.session_state.rate_limit_warning = False
if 'last_api_call' not in st.session_state:
    st.session_state.last_api_call = 0
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'medical_conditions' not in st.session_state:
    st.session_state.medical_conditions = []
if 'current_diagnosis' not in st.session_state:
    st.session_state.current_diagnosis = ""
if 'treatment_plan' not in st.session_state:
    st.session_state.treatment_plan = ""

class MedicalAIAnalyzer:
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.medical_knowledge_base = self._initialize_medical_knowledge()
        # Current available Groq models
        self.available_models = [
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768",
            "gemma2-9b-it"
        ]
        self.current_model = "llama-3.1-8b-instant"
        self.rate_limit_delay = 2
        self.conversation_context = []
    
    def _initialize_medical_knowledge(self):
        """Initialize comprehensive medical knowledge base"""
        return {
            'lab_ranges': {
                'WBC': {'low': 4.0, 'high': 11.0, 'unit': '10^3/μL', 'critical_low': 2.0, 'critical_high': 30.0},
                'RBC': {'low': 4.2, 'high': 5.8, 'unit': '10^6/μL', 'critical_low': 2.5, 'critical_high': 7.0},
                'Hemoglobin': {'low': 12.0, 'high': 16.0, 'unit': 'g/dL', 'critical_low': 7.0, 'critical_high': 20.0},
                'Hematocrit': {'low': 36, 'high': 48, 'unit': '%', 'critical_low': 20, 'critical_high': 60},
                'Platelets': {'low': 150, 'high': 450, 'unit': '10^3/μL', 'critical_low': 50, 'critical_high': 1000},
                'Glucose': {'low': 70, 'high': 100, 'unit': 'mg/dL', 'critical_low': 50, 'critical_high': 400},
                'Creatinine': {'low': 0.6, 'high': 1.3, 'unit': 'mg/dL', 'critical_low': 0.2, 'critical_high': 5.0},
                'BUN': {'low': 7, 'high': 20, 'unit': 'mg/dL', 'critical_low': 5, 'critical_high': 100},
                'ALT': {'low': 7, 'high': 55, 'unit': 'U/L', 'critical_low': 5, 'critical_high': 300},
                'AST': {'low': 8, 'high': 48, 'unit': 'U/L', 'critical_low': 5, 'critical_high': 300},
                'Total Cholesterol': {'low': 0, 'high': 200, 'unit': 'mg/dL', 'critical_low': 0, 'critical_high': 300},
                'LDL': {'low': 0, 'high': 100, 'unit': 'mg/dL', 'critical_low': 0, 'critical_high': 190},
                'HDL': {'low': 40, 'high': 100, 'unit': 'mg/dL', 'critical_low': 20, 'critical_high': 150},
                'Triglycerides': {'low': 0, 'high': 150, 'unit': 'mg/dL', 'critical_low': 0, 'critical_high': 500},
                'Iron': {'low': 60, 'high': 170, 'unit': 'μg/dL', 'critical_low': 40, 'critical_high': 200},
                'Ferritin': {'low': 15, 'high': 150, 'unit': 'ng/mL', 'critical_low': 10, 'critical_high': 500},
                'TIBC': {'low': 250, 'high': 400, 'unit': 'μg/dL', 'critical_low': 200, 'critical_high': 500},
                'Transferrin Saturation': {'low': 20, 'high': 50, 'unit': '%', 'critical_low': 15, 'critical_high': 60},
                'TSH': {'low': 0.4, 'high': 4.0, 'unit': 'mIU/L', 'critical_low': 0.1, 'critical_high': 10.0},
                'T3': {'low': 80, 'high': 200, 'unit': 'ng/dL', 'critical_low': 50, 'critical_high': 300},
                'T4': {'low': 4.5, 'high': 12.0, 'unit': 'μg/dL', 'critical_low': 2.0, 'critical_high': 20.0},
                'Vitamin B12': {'low': 200, 'high': 900, 'unit': 'pg/mL', 'critical_low': 150, 'critical_high': 1500},
                'Vitamin D': {'low': 30, 'high': 100, 'unit': 'ng/mL', 'critical_low': 10, 'critical_high': 150}
            },
            'symptoms_database': {
                'fatigue': {
                    'common_causes': ['Iron deficiency anemia', 'Thyroid disorders', 'Sleep apnea', 'Depression', 'Chronic fatigue syndrome'],
                    'urgent_signs': ['Chest pain', 'Shortness of breath', 'Fainting', 'Severe weakness'],
                    'tests': ['CBC', 'Iron studies', 'Thyroid panel', 'Vitamin B12', 'Vitamin D']
                },
                'headache': {
                    'common_causes': ['Tension headache', 'Migraine', 'Sinusitis', 'Dehydration', 'Eye strain'],
                    'urgent_signs': ['Sudden severe headache', 'Vision changes', 'Confusion', 'Fever with stiff neck'],
                    'tests': ['Blood pressure', 'Neurological exam', 'CT scan if indicated']
                },
                'chest_pain': {
                    'common_causes': ['Acid reflux', 'Anxiety', 'Muscle strain', 'Angina', 'Pulmonary issues'],
                    'urgent_signs': ['Radiating pain', 'Shortness of breath', 'Sweating', 'Nausea'],
                    'tests': ['ECG', 'Cardiac enzymes', 'Chest X-ray', 'Stress test']
                },
                'abdominal_pain': {
                    'common_causes': ['Indigestion', 'Irritable bowel', 'Appendicitis', 'Gallstones', 'UTI'],
                    'urgent_signs': ['Severe pain', 'Fever', 'Vomiting blood', 'Unable to pass stool'],
                    'tests': ['CBC', 'Liver function tests', 'Ultrasound', 'CT scan']
                }
            },
            'medication_database': {
                'iron_deficiency': {
                    'supplements': ['Ferrous sulfate', 'Ferrous gluconate', 'Iron polysaccharide'],
                    'dosing': '65-200 mg elemental iron daily',
                    'duration': '3-6 months',
                    'monitoring': 'Repeat blood tests in 2-3 months'
                },
                'hypertension': {
                    'medications': ['ACE inhibitors', 'Beta blockers', 'Calcium channel blockers', 'Diuretics'],
                    'lifestyle': 'Low salt diet, exercise, weight management',
                    'monitoring': 'Regular blood pressure checks'
                },
                'diabetes': {
                    'medications': ['Metformin', 'Insulin', 'GLP-1 agonists', 'SGLT2 inhibitors'],
                    'monitoring': 'Blood glucose, HbA1c every 3-6 months',
                    'targets': 'HbA1c < 7%, fasting glucose 80-130 mg/dL'
                }
            }
        }

    def call_groq_api(self, messages, model=None, max_tokens=1000, temperature=0.7):
        """Make API call to Groq with comprehensive error handling"""
        if not self.api_key:
            return None
        
        model_to_use = model or self.current_model
        
        # Rate limiting
        current_time = time.time()
        time_since_last_call = current_time - st.session_state.last_api_call
        if time_since_last_call < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - time_since_last_call)
        
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "messages": messages,
            "model": model_to_use,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": 0.9,
            "stream": False
        }
        
        try:
            response = requests.post(url, headers=headers, json=data, timeout=30)
            st.session_state.last_api_call = time.time()
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            elif response.status_code == 429:
                st.session_state.rate_limit_warning = True
                logger.warning("Rate limit exceeded")
                self.rate_limit_delay = 5
                return None
            elif response.status_code == 400:
                error_data = response.json()
                if "decommissioned" in error_data.get("error", {}).get("message", ""):
                    logger.warning(f"Model {model_to_use} deprecated, trying fallback")
                    return self._try_fallback_model(messages, max_tokens, temperature)
                return None
            else:
                logger.error(f"API error: {response.status_code}")
                return None
        except Exception as e:
            logger.error(f"API call failed: {str(e)}")
            return None

    def _try_fallback_model(self, messages, max_tokens, temperature):
        """Try alternative models if primary fails"""
        current_index = self.available_models.index(self.current_model) if self.current_model in self.available_models else 0
        fallback_models = self.available_models[current_index + 1:]
        
        for model in fallback_models:
            try:
                time.sleep(1)
                
                url = "https://api.groq.com/openai/v1/chat/completions"
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                data = {
                    "messages": messages,
                    "model": model,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "top_p": 0.9,
                    "stream": False
                }
                
                response = requests.post(url, headers=headers, json=data, timeout=30)
                st.session_state.last_api_call = time.time()
                
                if response.status_code == 200:
                    result = response.json()
                    logger.info(f"Using fallback model: {model}")
                    self.current_model = model
                    return result["choices"][0]["message"]["content"]
            except Exception as e:
                logger.error(f"Fallback model {model} failed: {str(e)}")
                continue
        
        return None

    def extract_text_from_pdf(self, pdf_file):
        """Extract text from uploaded PDF file"""
        try:
            with pdfplumber.open(pdf_file) as pdf:
                text = ""
                for page in pdf.pages:
                    text += page.extract_text() or ""
            return text
        except Exception as e:
            logger.error(f"PDF extraction failed: {str(e)}")
            return f"Error reading PDF: {str(e)}"

    def extract_text_from_docx(self, docx_file):
        """Extract text from uploaded DOCX file"""
        try:
            doc = docx.Document(docx_file)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return text
        except Exception as e:
            logger.error(f"DOCX extraction failed: {str(e)}")
            return f"Error reading DOCX: {str(e)}"

    def process_uploaded_files(self, uploaded_files):
        """Process all uploaded files and extract text"""
        all_text = ""
        for uploaded_file in uploaded_files:
            if uploaded_file.type == "application/pdf":
                text = self.extract_text_from_pdf(uploaded_file)
                all_text += f"\n--- PDF Content from {uploaded_file.name} ---\n{text}\n"
            elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                text = self.extract_text_from_docx(uploaded_file)
                all_text += f"\n--- DOCX Content from {uploaded_file.name} ---\n{text}\n"
            elif uploaded_file.type == "text/plain":
                text = str(uploaded_file.read(), "utf-8")
                all_text += f"\n--- TXT Content from {uploaded_file.name} ---\n{text}\n"
        return all_text

    def chat_with_medical_ai(self, message, patient_context="", processed_data=""):
        """Advanced medical conversation with AI"""
        if not self.api_key or st.session_state.rate_limit_warning:
            return self._professional_medical_response(message, patient_context, processed_data)
        
        try:
            # Build comprehensive medical context
            system_prompt = f"""You are Dr. MedAI, an experienced physician and medical AI assistant. You provide professional, evidence-based medical guidance.

PATIENT CONTEXT:
{patient_context}

MEDICAL DATA:
{processed_data}

CONVERSATION HISTORY:
{st.session_state.conversation_history[-5:] if st.session_state.conversation_history else 'No previous conversation'}

MEDICAL CONDITIONS IDENTIFIED:
{st.session_state.medical_conditions}

CURRENT DIAGNOSIS CONSIDERATIONS:
{st.session_state.current_diagnosis}

TREATMENT PLAN:
{st.session_state.treatment_plan}

PROFESSIONAL GUIDELINES:

1. CLINICAL ASSESSMENT:
   - Analyze symptoms systematically
   - Consider differential diagnoses
   - Evaluate risk factors
   - Assess urgency level

2. DIAGNOSTIC APPROACH:
   - Suggest appropriate tests based on presentation
   - Interpret lab results in clinical context
   - Consider imaging when indicated
   - Review medication interactions

3. TREATMENT RECOMMENDATIONS:
   - Provide evidence-based treatment options
   - Discuss lifestyle modifications
   - Consider medication management
   - Plan follow-up and monitoring

4. PATIENT EDUCATION:
   - Explain conditions in understandable terms
   - Discuss prognosis and expectations
   - Provide self-management strategies
   - Emphasize preventive care

5. SAFETY PROTOCOLS:
   - Identify red flags requiring immediate care
   - Discuss when to seek emergency help
   - Review medication safety
   - Consider contraindications

RESPONSE STYLE:
- Be professional yet compassionate
- Use medical terminology appropriately
- Provide clear explanations
- Offer practical next steps
- Always emphasize doctor consultation for definitive care

CURRENT QUERY: {message}"""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message}
            ]

            response = self.call_groq_api(messages, max_tokens=1200, temperature=0.7)
            
            if response:
                # Update conversation history
                st.session_state.conversation_history.append(f"Patient: {message}")
                st.session_state.conversation_history.append(f"Doctor: {response}")
                
                # Extract potential medical conditions from response
                self._extract_medical_insights(response)
                
                return response
            else:
                return self._professional_medical_response(message, patient_context, processed_data)

        except Exception as e:
            logger.error(f"Medical AI error: {str(e)}")
            return self._professional_medical_response(message, patient_context, processed_data)

    def _extract_medical_insights(self, response):
        """Extract medical conditions and insights from AI response"""
        # Simple pattern matching for common conditions
        conditions = []
        if "iron deficiency" in response.lower() or "anemia" in response.lower():
            conditions.append("Iron Deficiency Anemia")
        if "hypertension" in response.lower() or "high blood pressure" in response.lower():
            conditions.append("Hypertension")
        if "diabetes" in response.lower() or "blood sugar" in response.lower():
            conditions.append("Diabetes")
        if "thyroid" in response.lower():
            conditions.append("Thyroid Disorder")
        
        if conditions:
            st.session_state.medical_conditions = list(set(st.session_state.medical_conditions + conditions))

    def _professional_medical_response(self, message, patient_context, processed_data):
        """Provide professional medical responses when AI is unavailable"""
        message_lower = message.lower()
        
        # Update conversation history
        st.session_state.conversation_history.append(f"Patient: {message}")
        
        # Symptom analysis
        if any(symptom in message_lower for symptom in ['symptom', 'pain', 'hurt', 'feel', 'experience']):
            response = self._analyze_symptoms_professionally(message_lower, patient_context, processed_data)
        
        # Lab results analysis
        elif any(term in message_lower for term in ['lab', 'test', 'result', 'report', 'blood', 'level']):
            response = self._analyze_lab_results_professionally(processed_data)
        
        # Medication questions
        elif any(term in message_lower for term in ['medication', 'medicine', 'pill', 'drug', 'prescription']):
            response = self._provide_medication_guidance(message_lower, patient_context)
        
        # Diagnosis questions
        elif any(term in message_lower for term in ['diagnosis', 'condition', 'disease', 'illness']):
            response = self._discuss_diagnosis(message_lower, patient_context)
        
        # Treatment questions
        elif any(term in message_lower for term in ['treatment', 'cure', 'therapy', 'manage', 'control']):
            response = self._discuss_treatment(message_lower, patient_context)
        
        # General health questions
        elif any(term in message_lower for term in ['health', 'prevent', 'diet', 'exercise', 'lifestyle']):
            response = self._provide_health_advice(message_lower, patient_context)
        
        # Emergency situations
        elif any(term in message_lower for term in ['emergency', 'urgent', '911', 'hospital', 'severe', 'critical']):
            response = self._handle_emergency_situation(message_lower)
        
        # General conversation
        else:
            response = self._general_medical_conversation(message_lower, patient_context)
        
        st.session_state.conversation_history.append(f"Doctor: {response}")
        return response

    def _analyze_symptoms_professionally(self, message, patient_context, processed_data):
        """Professional symptom analysis"""
        analysis = """## 🩺 Symptom Analysis

**Clinical Assessment Approach:**

I understand you're experiencing symptoms that concern you. Let me help you analyze this systematically:

### Differential Diagnosis Considerations:"""

        # Add specific symptom analysis based on context
        if 'fatigue' in message:
            analysis += """
**Fatigue Analysis:**
• **Common Causes**: Iron deficiency, thyroid disorders, sleep issues, nutritional deficiencies
• **Assessment Needed**: Complete blood count, thyroid panel, iron studies, vitamin levels
• **Red Flags**: Chest pain, shortness of breath, significant weight loss, fainting"""
        
        if 'headache' in message:
            analysis += """
**Headache Analysis:**
• **Common Types**: Tension headaches, migraines, sinus-related, cluster headaches
• **Assessment**: Blood pressure, neurological exam, vision check
• **Urgent Signs**: Sudden severe headache, vision changes, confusion, fever with neck stiffness"""
        
        if 'chest' in message and 'pain' in message:
            analysis += """
**Chest Pain Analysis:**
• **Cardiac Considerations**: Angina, myocardial infarction, pericarditis
• **Non-Cardiac Causes**: Acid reflux, costochondritis, anxiety, pulmonary issues
• **Emergency Signs**: Radiating pain, shortness of breath, sweating, nausea"""
        
        analysis += """

### Recommended Next Steps:

1. **Immediate Actions:**
   - Monitor symptom patterns and triggers
   - Keep a symptom diary with timing and severity
   - Note any associated symptoms

2. **Medical Evaluation:**
   - Schedule appointment with primary care physician
   - Discuss symptom progression and impact
   - Review complete medical history

3. **Diagnostic Considerations:**
   - Basic laboratory testing as indicated
   - Specialist referral if needed
   - Imaging studies if clinically warranted

**When to Seek Immediate Care:**
• Chest pain or pressure
• Difficulty breathing
• Severe uncontrolled pain
• Neurological changes
• High fever with other symptoms

Would you like me to analyze specific symptoms in more detail or discuss your test results?"""
        
        return analysis

    def _analyze_lab_results_professionally(self, processed_data):
        """Professional lab results analysis"""
        if not processed_data:
            return """## 🔬 Laboratory Results Analysis

I don't see any laboratory results to analyze yet. 

**To provide comprehensive analysis, please:**
1. Upload your lab reports (PDF/DOCX/TXT formats)
2. Or paste your results in this format:
