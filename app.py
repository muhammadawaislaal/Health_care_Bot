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

class MedicalAIAnalyzer:
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.medical_knowledge_base = self._initialize_medical_knowledge()
        # Updated to current available Groq models
        self.available_models = [
            "llama-3.1-8b-instant",  # Primary model
            "mixtral-8x7b-32768",    # Fallback 1
            "gemma2-9b-it",          # Fallback 2
            "llama3-70b-8192"        # Fallback 3 (if available)
        ]
        self.current_model = "llama-3.1-8b-instant"  # Start with reliable model
        self.rate_limit_delay = 3  # Increased delay between API calls
    
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
                'Transferrin Saturation': {'low': 20, 'high': 50, 'unit': '%', 'critical_low': 15, 'critical_high': 60}
            },
            'symptoms_analysis': {
                'fatigue': ['Iron deficiency', 'Anemia', 'Thyroid issues', 'Sleep disorders'],
                'weakness': ['Electrolyte imbalance', 'Anemia', 'Chronic fatigue', 'Nutritional deficiencies'],
                'pale_skin': ['Anemia', 'Iron deficiency', 'Circulation issues'],
                'shortness_of_breath': ['Anemia', 'Cardiac issues', 'Respiratory conditions'],
                'dizziness': ['Anemia', 'Dehydration', 'Blood pressure issues']
            }
        }

    def call_groq_api(self, messages, model=None, max_tokens=800, temperature=0.7):
        """Make direct API call to Groq with rate limit handling"""
        if not self.api_key:
            return None
        
        # Use provided model or default
        model_to_use = model or self.current_model
        
        # Rate limiting - add delay between API calls
        current_time = time.time()
        time_since_last_call = current_time - st.session_state.last_api_call
        if time_since_last_call < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - time_since_last_call)
        
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Optimize token usage
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
                # Rate limit exceeded
                st.session_state.rate_limit_warning = True
                logger.warning("Rate limit exceeded, using fallback responses")
                # Increase delay for next calls
                self.rate_limit_delay = 5
                return None
            elif response.status_code == 400:
                # Model might be deprecated, try fallback
                error_data = response.json()
                if "decommissioned" in error_data.get("error", {}).get("message", ""):
                    logger.warning(f"Model {model_to_use} is decommissioned, trying fallback")
                    return self._try_fallback_model(messages, max_tokens, temperature)
                return None
            else:
                logger.error(f"Groq API error: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            logger.error(f"Groq API call failed: {str(e)}")
            return None

    def _try_fallback_model(self, messages, max_tokens, temperature):
        """Try alternative models if primary fails"""
        current_index = self.available_models.index(self.current_model) if self.current_model in self.available_models else 0
        fallback_models = self.available_models[current_index + 1:]
        
        for model in fallback_models:
            try:
                # Add delay between fallback attempts
                time.sleep(2)
                
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
                    logger.info(f"Successfully used fallback model: {model}")
                    self.current_model = model  # Switch to this model for future calls
                    return result["choices"][0]["message"]["content"]
                elif response.status_code == 429:
                    st.session_state.rate_limit_warning = True
                    continue
                elif response.status_code == 400:
                    # This model is also deprecated, continue to next
                    continue
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
        """Advanced medical chat using Groq API with intelligent patient context"""
        if not self.api_key or st.session_state.rate_limit_warning:
            return self._intelligent_fallback_response(message, patient_context, processed_data)
        
        try:
            # Build optimized system prompt
            system_prompt = f"""You are Dr. MedAI, a medical assistant. Analyze patient data and provide helpful guidance.

PATIENT CONTEXT:
{patient_context if patient_context else 'No patient info provided'}

MEDICAL DATA:
{processed_data if processed_data else 'No additional medical data'}

INSTRUCTIONS:
- Be specific, practical, and empathetic
- Analyze any lab results or symptoms mentioned
- Provide evidence-based advice when possible
- Suggest appropriate next steps
- Always recommend consulting healthcare providers
- Keep responses clear and under 400 words

Focus on being helpful while acknowledging limitations."""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message}
            ]

            # Get response from Groq API
            response = self.call_groq_api(messages, max_tokens=600, temperature=0.7)
            
            if response:
                return response
            else:
                return self._intelligent_fallback_response(message, patient_context, processed_data)

        except Exception as e:
            logger.error(f"Medical AI error: {str(e)}")
            return self._intelligent_fallback_response(message, patient_context, processed_data)

    def _intelligent_fallback_response(self, message, patient_context, processed_data):
        """Provide intelligent fallback responses based on patient context"""
        message_lower = message.lower()
        
        # Check for specific medical conditions in context
        has_iron_deficiency = "iron" in patient_context.lower() or "iron" in processed_data.lower()
        has_lab_data = any(keyword in processed_data.lower() for keyword in ['hemoglobin', 'wbc', 'rbc', 'ferritin', 'glucose'])
        
        if any(greet in message_lower for greet in ['hi', 'hello', 'hey']):
            if has_iron_deficiency:
                return """👋 Hello! I can see you've mentioned iron deficiency in your information. 

I can help you understand:
• What your lab results might mean
• Dietary recommendations for iron
• Common symptoms and management
• When to seek medical attention

What specific aspect would you like to discuss?"""
            else:
                return """👋 Hello! I'm Dr. MedAI, your medical assistant.

I can help analyze:
• Medical test results and lab values
• Symptoms and health concerns
• Medication questions
• General health guidance

Please share your medical information or upload reports for personalized analysis."""

        elif any(word in message_lower for word in ['iron', 'deficiency', 'anemia', 'ferritin', 'hemoglobin']):
            return self._provide_iron_deficiency_analysis(patient_context, processed_data)
        
        elif any(word in message_lower for word in ['symptom', 'tired', 'fatigue', 'weak', 'dizzy', 'pale']):
            return self._analyze_symptoms(message_lower, patient_context, processed_data)
        
        elif any(word in message_lower for word in ['lab', 'test', 'result', 'report']):
            return self._analyze_lab_results(processed_data)
        
        elif any(word in message_lower for word in ['diet', 'food', 'nutrition', 'eat']):
            return self._provide_nutrition_advice(patient_context, processed_data)
        
        elif any(word in message_lower for word in ['what', 'how', 'why', 'explain']):
            return """I'd be happy to help explain medical concepts or test results!

To provide the most helpful information:
1. Share your specific test results or lab values
2. Describe your symptoms in detail
3. Upload any medical reports you have
4. Tell me about any diagnoses you've received

The more specific information you provide, the better I can assist you."""
        
        else:
            return """I'm here to help with your health questions and concerns.

Please feel free to:
• Share your medical test results
• Describe your symptoms
• Ask about medications or treatments
• Upload medical reports for analysis

I'll do my best to provide helpful information and guidance based on the data you provide."""

    def _provide_iron_deficiency_analysis(self, patient_context, processed_data):
        """Provide detailed iron deficiency analysis"""
        analysis = """🔬 **Iron Deficiency Analysis**

**Understanding Iron Deficiency:**
Iron deficiency occurs when your body doesn't have enough iron to produce adequate hemoglobin. This can lead to anemia, causing various symptoms.

**Common Symptoms:**
• Fatigue and weakness
• Pale skin and conjunctiva
• Shortness of breath
• Dizziness or lightheadedness
• Headaches
• Cold hands and feet

**Key Laboratory Indicators:**
"""
        
        # Extract and analyze lab values
        lab_values = self.extract_lab_values(processed_data)
        
        critical_findings = []
        normal_findings = []
        
        if 'Hemoglobin' in lab_values:
            hb = lab_values['Hemoglobin']
            if hb < 12.0:
                critical_findings.append(f"Hemoglobin: {hb} g/dL (LOW - normal range: 12.0-16.0 g/dL)")
            else:
                normal_findings.append(f"Hemoglobin: {hb} g/dL (Normal)")
        
        if 'Ferritin' in lab_values:
            ferritin = lab_values['Ferritin']
            if ferritin < 15:
                critical_findings.append(f"Ferritin: {ferritin} ng/mL (VERY LOW - indicates iron deficiency)")
            elif ferritin < 30:
                critical_findings.append(f"Ferritin: {ferritin} ng/mL (LOW - suggests iron deficiency)")
            else:
                normal_findings.append(f"Ferritin: {ferritin} ng/mL (Normal)")
        
        if critical_findings:
            analysis += "\n🚨 **Important Findings:**\n" + "\n".join([f"• {finding}" for finding in critical_findings]) + "\n"
        
        if normal_findings:
            analysis += "\n✅ **Normal Values:**\n" + "\n".join([f"• {finding}" for finding in normal_findings]) + "\n"
        
        analysis += """
**Recommended Actions:**

1. **Medical Consultation:** Schedule an appointment with your doctor for proper diagnosis
2. **Laboratory Testing:** Complete iron studies if not already done
3. **Dietary Changes:** Increase iron-rich foods (red meat, spinach, lentils, fortified cereals)
4. **Supplementation:** Consider iron supplements if recommended by your doctor
5. **Follow-up:** Repeat testing in 2-3 months to monitor progress

**When to Seek Urgent Care:**
• Severe fatigue preventing daily activities
• Chest pain or palpitations
• Shortness of breath at rest
• Significant dizziness or fainting

**Important:** Always discuss these findings with your healthcare provider for proper medical advice."""

        return analysis

    def _analyze_symptoms(self, message, patient_context, processed_data):
        """Analyze patient symptoms"""
        return """🤒 **Symptom Analysis**

I understand you're experiencing symptoms that concern you. Let me help you understand what they might indicate and what steps to consider.

**Common Symptom Patterns:**

**Fatigue + Weakness:**
• Iron deficiency anemia (very common)
• Thyroid disorders
• Sleep issues or poor sleep quality
• Vitamin deficiencies (B12, Vitamin D)
• Chronic fatigue syndrome

**Dizziness + Pale Skin:**
• Anemia (reduced oxygen delivery)
• Dehydration or electrolyte imbalance
• Blood pressure fluctuations
• Inner ear problems

**Next Steps to Consider:**
1. **Symptom Tracking:** Keep a diary of when symptoms occur, their severity, and any triggers
2. **Medical Evaluation:** Schedule an appointment with your primary care physician
3. **Basic Testing:** Consider complete blood count (CBC), iron studies, and thyroid panel
4. **Lifestyle Review:** Evaluate sleep patterns, stress levels, diet, and hydration

**Red Flags Requiring Prompt Medical Attention:**
• Chest pain or pressure
• Difficulty breathing or shortness of breath
• Fainting or loss of consciousness
• Severe, persistent headache
• Rapid heart rate or palpitations

Would you like to discuss specific symptoms in more detail or share any test results you have?"""

    def _analyze_lab_results(self, processed_data):
        """Analyze laboratory results"""
        if not processed_data:
            return "I don't see any lab results to analyze yet. Please upload your lab reports or paste your test results in the sidebar for detailed analysis."
        
        lab_values = self.extract_lab_values(processed_data)
        
        if not lab_values:
            return """I found some medical data but couldn't extract specific lab values. 

For the most accurate analysis, please paste your results in this format:
