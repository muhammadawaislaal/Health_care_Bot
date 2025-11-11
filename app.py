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
        # Available Groq models - using smaller models to reduce token usage
        self.available_models = [
            "llama-3.2-1b-preview",  # Smallest model for basic responses
            "llama-3.2-3b-preview",  # Medium model for better analysis
            "llama-3.1-8b-instant",  # Larger model when available
        ]
        self.current_model = "llama-3.2-1b-preview"  # Start with smallest model
        self.rate_limit_delay = 2  # Base delay between API calls
    
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
        
        # Optimize token usage by using smaller models and shorter responses
        data = {
            "messages": messages,
            "model": model_to_use,
            "temperature": temperature,
            "max_tokens": max_tokens,  # Reduced from 1500 to 800
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
                return None
            else:
                logger.error(f"Groq API error: {response.status_code} - {response.text}")
                # Try fallback model if first attempt fails
                if model_to_use != self.available_models[-1]:  # If not the last model in list
                    return self._try_fallback_model(messages, max_tokens, temperature)
                return None
        except Exception as e:
            logger.error(f"Groq API call failed: {str(e)}")
            return None

    def _try_fallback_model(self, messages, max_tokens, temperature):
        """Try alternative models if primary fails"""
        current_index = self.available_models.index(self.current_model)
        fallback_models = self.available_models[current_index + 1:]
        
        for model in fallback_models:
            try:
                # Add delay between fallback attempts
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
                    logger.info(f"Successfully used fallback model: {model}")
                    self.current_model = model  # Switch to this model for future calls
                    return result["choices"][0]["message"]["content"]
                elif response.status_code == 429:
                    st.session_state.rate_limit_warning = True
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
            # Build optimized system prompt (shorter to save tokens)
            system_prompt = f"""You are Dr. MedAI, a medical assistant. Analyze patient data and provide guidance.

PATIENT CONTEXT:
{patient_context if patient_context else 'No patient info'}

MEDICAL DATA:
{processed_data if processed_data else 'No medical data'}

INSTRUCTIONS:
- Be specific and actionable
- Analyze lab results and symptoms
- Provide evidence-based advice
- Be conversational and empathetic
- Suggest next steps
- Always recommend doctor consultation

Keep responses under 500 words. Focus on key insights."""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message}
            ]

            # Get response from Groq API with reduced tokens
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
                return """👋 Hello! I can see you've mentioned iron deficiency. 

I can help you understand:
• Your lab results and what they mean
• Dietary recommendations for iron
• When to seek medical care
• Questions to ask your doctor

What would you like to know about iron deficiency?"""
            else:
                return """👋 Hello! I'm Dr. MedAI, your medical assistant.

I can help analyze:
• Medical test results
• Symptoms and concerns  
• Medication questions
• Health guidance

Please share your medical information or upload reports for personalized analysis."""

        elif any(word in message_lower for word in ['iron', 'deficiency', 'anemia', 'ferritin', 'hemoglobin']):
            return self._provide_iron_deficiency_analysis(patient_context, processed_data)
        
        elif any(word in message_lower for word in ['symptom', 'tired', 'fatigue', 'weak', 'dizzy', 'pale']):
            return self._analyze_symptoms(message_lower, patient_context, processed_data)
        
        elif any(word in message_lower for word in ['lab', 'test', 'result', 'report']):
            return self._analyze_lab_results(processed_data)
        
        elif any(word in message_lower for word in ['diet', 'food', 'nutrition', 'eat']):
            return self._provide_nutrition_advice(patient_context, processed_data)
        
        else:
            return """I'd be happy to help with your health questions! 

Please:
1. Share your specific symptoms or concerns
2. Upload medical reports or test results  
3. Describe any diagnoses you've received

The more information you provide, the better I can assist you."""

    def _provide_iron_deficiency_analysis(self, patient_context, processed_data):
        """Provide detailed iron deficiency analysis"""
        analysis = """🔬 **Iron Deficiency Analysis**

**Common Symptoms:**
• Fatigue and weakness
• Pale skin
• Shortness of breath  
• Dizziness
• Headaches

**Key Lab Findings:**
"""
        
        # Extract and analyze lab values
        lab_values = self.extract_lab_values(processed_data)
        
        critical_findings = []
        if 'Hemoglobin' in lab_values:
            hb = lab_values['Hemoglobin']
            if hb < 12.0:
                critical_findings.append(f"Hemoglobin: {hb} g/dL (LOW - indicates anemia)")
        
        if 'Ferritin' in lab_values:
            ferritin = lab_values['Ferritin']
            if ferritin < 15:
                critical_findings.append(f"Ferritin: {ferritin} ng/mL (VERY LOW - iron deficiency)")
        
        if critical_findings:
            analysis += "\n🚨 **Important Findings:**\n" + "\n".join([f"• {finding}" for finding in critical_findings]) + "\n"
        
        analysis += """
**Recommended Actions:**
1. Consult doctor for proper diagnosis
2. Consider iron supplements if recommended
3. Eat iron-rich foods (red meat, spinach, lentils)
4. Take vitamin C with iron for better absorption
5. Follow-up testing in 2-3 months

**When to Seek Urgent Care:**
• Severe fatigue preventing daily activities
• Chest pain or palpitations
• Shortness of breath at rest"""
        
        return analysis

    def _analyze_symptoms(self, message, patient_context, processed_data):
        """Analyze patient symptoms"""
        return """🤒 **Symptom Analysis**

**Common Causes of Fatigue:**
• Iron deficiency anemia
• Thyroid issues  
• Sleep problems
• Nutritional deficiencies

**Next Steps:**
1. Track symptoms - timing and severity
2. Get blood work (CBC, iron, thyroid)
3. Discuss with your doctor
4. Review sleep and diet

**Seek Urgent Care for:**
• Chest pain
• Difficulty breathing  
• Fainting
• Severe headache"""

    def _analyze_lab_results(self, processed_data):
        """Analyze laboratory results"""
        if not processed_data:
            return "Please upload lab reports or paste test results for analysis."
        
        lab_values = self.extract_lab_values(processed_data)
        
        if not lab_values:
            return "Please paste results in format: Hemoglobin: 12.5 g/dL, Ferritin: 25 ng/mL"
        
        analysis = "🔬 **Lab Results Analysis**\n\n"
        
        abnormal_count = 0
        for test, value in lab_values.items():
            if test in self.medical_knowledge_base['lab_ranges']:
                ranges = self.medical_knowledge_base['lab_ranges'][test]
                if value < ranges['low']:
                    analysis += f"⚠️ **{test}**: {value} {ranges['unit']} (LOW)\n"
                    abnormal_count += 1
                elif value > ranges['high']:
                    analysis += f"⚠️ **{test}**: {value} {ranges['unit']} (HIGH)\n"
                    abnormal_count += 1
                else:
                    analysis += f"✅ **{test}**: {value} {ranges['unit']} (Normal)\n"
        
        if abnormal_count > 0:
            analysis += f"\n**{abnormal_count} abnormal values found** - discuss with your doctor."
        else:
            analysis += "\n**All values normal** - maintain healthy habits."
        
        return analysis

    def _provide_nutrition_advice(self, patient_context, processed_data):
        """Provide nutrition advice based on condition"""
        if "iron" in patient_context.lower() or "iron" in processed_data.lower():
            return """🍎 **Nutrition for Iron Deficiency**

**Iron-Rich Foods:**
• Red meat, poultry, fish
• Spinach, lentils, beans
• Fortified cereals
• Pumpkin seeds

**Enhance Absorption:**
• Vitamin C (citrus, bell peppers)
• Avoid tea/coffee with meals
• Cook in cast iron

**Daily Iron Goals:**
• Men: 8mg • Women: 18mg"""
        
        return """🍎 **General Nutrition**

**Healthy Diet:**
• Vegetables and fruits
• Lean protein  
• Whole grains
• Stay hydrated
• Limit processed foods"""

    def analyze_patient_data(self, patient_info, medical_history, lab_data, image_description, processed_files_text=""):
        """Comprehensive analysis of all patient data"""
        if not self.api_key or st.session_state.rate_limit_warning:
            return self._basic_patient_analysis(patient_info, medical_history, lab_data, image_description, processed_files_text)
        
        try:
            # Combine all data
            combined_data = f"""
PATIENT: {patient_info}
HISTORY: {medical_history}
LABS: {lab_data}
NOTES: {image_description}
FILES: {processed_files_text}
"""

            prompt = f"""
            Analyze this patient case briefly:

            {combined_data}

            Provide concise analysis with:
            - Key findings
            - Risk assessment  
            - Recommendations
            - Urgency level

            Keep under 400 words.
            """

            messages = [
                {
                    "role": "system", 
                    "content": "You are a physician. Provide concise medical analysis with clear recommendations."
                },
                {"role": "user", "content": prompt}
            ]

            response = self.call_groq_api(messages, max_tokens=500, temperature=0.3)
            
            if response:
                return response
            else:
                return self._basic_patient_analysis(patient_info, medical_history, lab_data, image_description, processed_files_text)
                
        except Exception as e:
            logger.error(f"Comprehensive analysis failed: {str(e)}")
            return self._basic_patient_analysis(patient_info, medical_history, lab_data, image_description, processed_files_text)

    def _basic_patient_analysis(self, patient_info, medical_history, lab_data, image_description, processed_files_text):
        """Basic analysis when AI is unavailable"""
        analysis = "## 📊 Patient Data Analysis\n\n"
        
        analysis += f"**Patient**: {patient_info}\n\n"
        
        if medical_history and medical_history.strip():
            analysis += f"**Medical History**: {medical_history}\n\n"
        
        # Analyze lab data
        combined_lab_data = lab_data + "\n" + processed_files_text
        lab_values = self.extract_lab_values(combined_lab_data)
        
        if lab_values:
            analysis += "### 🔬 Laboratory Findings\n"
            abnormal_count = 0
            for test, value in lab_values.items():
                if test in self.medical_knowledge_base['lab_ranges']:
                    ranges = self.medical_knowledge_base['lab_ranges'][test]
                    if value < ranges['low']:
                        analysis += f"⚠️ **{test}**: {value} {ranges['unit']} (LOW)\n"
                        abnormal_count += 1
                    elif value > ranges['high']:
                        analysis += f"⚠️ **{test}**: {value} {ranges['unit']} (HIGH)\n"
                        abnormal_count += 1
                    else:
                        analysis += f"✅ **{test}**: {value} {ranges['unit']} (Normal)\n"
            
            if abnormal_count > 0:
                analysis += f"\n**{abnormal_count} abnormal values found.**\n"
        
        analysis += "\n### 💡 Recommendations\n"
        analysis += "1. Review findings with healthcare provider\n"
        analysis += "2. Discuss any abnormal values\n"
        analysis += "3. Follow up as recommended\n"
        
        return analysis

    def extract_lab_values(self, text):
        """Extract laboratory values from text"""
        patterns = {
            'WBC': r'WBC[\s:\-]*([\d.]+)',
            'RBC': r'RBC[\s:\-]*([\d.]+)',
            'Hemoglobin': r'Hemoglobin[\s:\-]*([\d.]+)|Hgb[\s:\-]*([\d.]+)|Hb[\s:\-]*([\d.]+)',
            'Hematocrit': r'Hematocrit[\s:\-]*([\d.]+)|Hct[\s:\-]*([\d.]+)',
            'Platelets': r'Platelets[\s:\-]*([\d.]+)',
            'Glucose': r'Glucose[\s:\-]*([\d.]+)',
            'Creatinine': r'Creatinine[\s:\-]*([\d.]+)',
            'BUN': r'BUN[\s:\-]*([\d.]+)',
            'ALT': r'ALT[\s:\-]*([\d.]+)',
            'AST': r'AST[\s:\-]*([\d.]+)',
            'Iron': r'Iron[\s:\-]*([\d.]+)',
            'Ferritin': r'Ferritin[\s:\-]*([\d.]+)',
        }
        
        extracted_values = {}
        for test, pattern in patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                for value in match:
                    if value and value.strip():
                        try:
                            extracted_values[test] = float(value)
                            break
                        except ValueError:
                            continue
        return extracted_values

def setup_groq_api():
    """Setup Groq API configuration"""
    st.sidebar.header("🔑 API Configuration")
    
    # Check if API key is in Streamlit secrets
    if 'GROQ_API_KEY' in st.secrets:
        api_key = st.secrets['GROQ_API_KEY']
        
        if not api_key or not api_key.startswith('gsk_'):
            st.sidebar.markdown("<div class='api-status-disconnected'>❌ Invalid API Key Format</div>", unsafe_allow_html=True)
            return None
        
        # Test the API key
        try:
            url = "https://api.groq.com/openai/v1/models"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                st.session_state.api_configured = True
                st.session_state.api_key = api_key
                st.sidebar.markdown("<div class='api-status-connected'>✅ Groq API Connected</div>", unsafe_allow_html=True)
                
                # Check for rate limit warnings
                if st.session_state.rate_limit_warning:
                    st.sidebar.markdown("<div class='rate-limit-warning'>⚠️ Rate limits detected. Using optimized mode.</div>", unsafe_allow_html=True)
                
                return api_key
            else:
                st.sidebar.markdown("<div class='api-status-disconnected'>❌ API Authentication Failed</div>", unsafe_allow_html=True)
                return None
                
        except Exception as e:
            st.sidebar.markdown("<div class='api-status-disconnected'>❌ Connection Error</div>", unsafe_allow_html=True)
    
    else:
        st.sidebar.markdown("<div class='api-status-disconnected'>❌ API Key Not Found</div>", unsafe_allow_html=True)
        st.sidebar.info("""
        Get Groq API key from https://console.groq.com
        Add to Streamlit secrets as GROQ_API_KEY
        """)
    
    return None

def setup_sidebar():
    """Setup the professional medical sidebar"""
    with st.sidebar:
        st.markdown("<div class='medical-card'>", unsafe_allow_html=True)
        st.header("🏥 MediAI Pro")
        st.markdown("**AI-Powered Medical Assistant**")
        if st.session_state.api_configured:
            if st.session_state.rate_limit_warning:
                st.warning("🤖 AI: **OPTIMIZED MODE**")
            else:
                st.success("🤖 AI: **ACTIVE**")
        else:
            st.warning("🤖 AI: **BASIC MODE**")
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.header("👤 Patient Information")
        
        col1, col2 = st.columns(2)
        with col1:
            patient_id = st.text_input("Patient ID", value="PT-001", key="patient_id")
            patient_age = st.number_input("Age", min_value=0, max_value=120, value=45, key="patient_age")
        with col2:
            patient_name = st.text_input("Patient Name", value="John Doe", key="patient_name")
            patient_gender = st.selectbox("Gender", ["Male", "Female", "Other"], key="patient_gender")
        
        st.header("📋 Medical History")
        medical_history = st.text_area(
            "Medical History & Symptoms", 
            height=100,
            placeholder="Example: Iron deficiency, fatigue, pale skin...",
            key="medical_history"
        )
        
        st.header("📁 Medical Data Upload")
        
        tab1, tab2 = st.tabs(["Lab Reports", "Medical Notes"])
        
        with tab1:
            st.subheader("Laboratory Results")
            lab_files = st.file_uploader("Upload Lab Reports", 
                                       type=['pdf', 'docx', 'txt'], 
                                       key="lab_uploader",
                                       accept_multiple_files=True)
            
            if lab_files:
                st.info(f"📎 {len(lab_files)} file(s) uploaded")
            
            lab_text_input = st.text_area("Or Paste Lab Results:", height=100,
                                        placeholder="Hemoglobin: 11.2 g/dL\nFerritin: 12 ng/mL\nWBC: 6.8",
                                        key="lab_text")
        
        with tab2:
            st.subheader("Medical Notes & Symptoms")
            image_description = st.text_area("Clinical Notes & Findings:", height=100,
                                           placeholder="Patient reports fatigue and pale skin.",
                                           key="image_desc")
        
        # Process uploaded files
        processed_files_text = ""
        if lab_files:
            analyzer = MedicalAIAnalyzer(st.session_state.api_key)
            processed_files_text = analyzer.process_uploaded_files(lab_files)
            st.session_state.processed_lab_data = processed_files_text
        
        # Build patient context
        patient_context = f"""
Patient: {patient_name}, {patient_age} years, {patient_gender}
History: {medical_history if medical_history else 'Not specified'}
Labs: {lab_text_input if lab_text_input else 'Not provided'}
Notes: {image_description if image_description else 'Not provided'}
        """.strip()
        
        st.session_state.patient_context = patient_context
        
        # Analysis Controls
        st.markdown("<div class='medical-card'>", unsafe_allow_html=True)
        st.header("🔍 Analysis Controls")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🚀 **Analyze Patient Data**", use_container_width=True, type="primary"):
                st.session_state.analyze_clicked = True
        with col2:
            if st.button("🔄 **Clear All**", use_container_width=True):
                st.session_state.analysis_results = {}
                st.session_state.messages = []
                st.session_state.analyze_clicked = False
                st.session_state.processed_lab_data = ""
                st.session_state.rate_limit_warning = False
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
        
        return {
            'patient_info': f"ID: {patient_id}, Name: {patient_name}, Age: {patient_age}, Gender: {patient_gender}",
            'medical_history': medical_history,
            'lab_data': lab_text_input,
            'image_description': image_description,
            'patient_context': patient_context,
            'processed_files_text': processed_files_text
        }

def display_medical_chat(analyzer):
    """Display advanced medical chat interface"""
    st.header("💬 Dr. MedAI - Medical Conversation Assistant")
    
    # API status indicator
    if st.session_state.api_configured:
        if st.session_state.rate_limit_warning:
            st.warning("🟡 AI Assistant: **OPTIMIZED MODE** - Rate limits detected, using efficient responses")
        else:
            st.success("✅ AI Assistant: **ACTIVE** - Advanced AI conversations enabled")
    else:
        st.warning("🟡 AI Assistant: **BASIC MODE** - Using medical knowledge base")
    
    # Rate limit notice
    if st.session_state.rate_limit_warning:
        st.markdown("<div class='rate-limit-warning'>⚠️ **Note**: API rate limits detected. Responses are optimized for efficiency. Full AI features will resume when limits reset.</div>", unsafe_allow_html=True)
    
    # Patient context summary
    if st.session_state.patient_context:
        with st.expander("📋 Current Patient Context", expanded=False):
            st.text(st.session_state.patient_context)
    
    # Chat container
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    
    # Display chat messages
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"<div class='user-message'><strong>You:</strong> {message['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='assistant-message'><strong>Dr. MedAI:</strong> {message['content']}</div>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Chat input
    user_input = st.text_input(
        "Ask about your medical data, symptoms, or health concerns:",
        placeholder="Example: Can you explain my iron deficiency results?",
        key="chat_input"
    )
    
    col1, col2 = st.columns([3, 1])
    with col1:
        if st.button("Send Message", use_container_width=True) and user_input:
            process_user_message(analyzer, user_input)
    with col2:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    
    # Quick action buttons
    st.subheader("💡 Quick Medical Questions")
    quick_col1, quick_col2 = st.columns(2)
    
    with quick_col1:
        if st.button("🤒 Discuss Symptoms", use_container_width=True):
            process_user_message(analyzer, "I'd like to discuss my symptoms")
        if st.button("🔬 Analyze Test Results", use_container_width=True):
            process_user_message(analyzer, "Can you help me understand my medical test results?")
    
    with quick_col2:
        if st.button("🍎 Nutrition Advice", use_container_width=True):
            process_user_message(analyzer, "What dietary recommendations do you suggest?")
        if st.button("📋 Next Steps", use_container_width=True):
            process_user_message(analyzer, "What are the recommended next steps?")

def process_user_message(analyzer, user_input):
    """Process user message and generate AI response"""
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Generate AI response
    with st.spinner("Dr. MedAI is analyzing..."):
        ai_response = analyzer.chat_with_medical_ai(
            user_input, 
            st.session_state.patient_context,
            st.session_state.processed_lab_data
        )
        
        # Add AI response to chat history
        st.session_state.messages.append({"role": "assistant", "content": ai_response})
    
    st.rerun()

def display_analysis_dashboard(analyzer, patient_data):
    """Display professional medical dashboard"""
    st.header("📊 Medical Analysis Dashboard")
    
    if not st.session_state.analysis_results:
        st.info("""
        🏥 **Welcome to MediAI Pro**
        
        To begin:
        1. Enter patient information
        2. Add medical history or test results  
        3. Upload medical reports
        4. Click **'Analyze Patient Data'**
        5. Use chat for questions
        
        *More information = Better analysis!*
        """)
        return
    
    # Summary Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Analysis", "Completed")
    
    with col2:
        st.metric("AI Mode", "Optimized" if st.session_state.rate_limit_warning else "Active")
    
    with col3:
        has_data = any([
            patient_data['medical_history'], 
            patient_data['lab_data'], 
            patient_data['processed_files_text']
        ])
        st.metric("Data", "Loaded" if has_data else "Minimal")
    
    with col4:
        lab_values = analyzer.extract_lab_values(patient_data['lab_data'] + patient_data['processed_files_text'])
        abnormal_count = sum(1 for test, value in lab_values.items() 
                           if test in analyzer.medical_knowledge_base['lab_ranges'] 
                           and (value < analyzer.medical_knowledge_base['lab_ranges'][test]['low'] 
                                or value > analyzer.medical_knowledge_base['lab_ranges'][test]['high']))
        st.metric("Abnormal Values", abnormal_count if lab_values else "N/A")
    
    # Detailed Analysis
    if 'comprehensive_analysis' in st.session_state.analysis_results:
        with st.expander("📋 **Patient Analysis**", expanded=True):
            st.markdown(st.session_state.analysis_results['comprehensive_analysis'])

def create_advanced_visualizations(patient_data, analyzer):
    """Create professional medical visualizations"""
    st.header("📈 Health Analytics")
    
    # Extract lab values for visualization
    combined_data = patient_data['lab_data'] + patient_data['processed_files_text']
    lab_values = analyzer.extract_lab_values(combined_data)
    
    if lab_values:
        st.subheader("🩸 Lab Values Overview")
        
        # Prepare data for visualization
        viz_data = []
        for test, value in lab_values.items():
            if test in analyzer.medical_knowledge_base['lab_ranges']:
                ranges = analyzer.medical_knowledge_base['lab_ranges'][test]
                status = "Normal"
                color = "green"
                if value < ranges['low']:
                    status = "Low"
                    color = "blue"
                elif value > ranges['high']:
                    status = "High"
                    color = "red"
                
                viz_data.append({
                    'Parameter': test,
                    'Value': value,
                    'Status': status,
                    'Color': color
                })
        
        if viz_data:
            df = pd.DataFrame(viz_data)
            
            fig = go.Figure(data=[
                go.Bar(
                    x=df['Parameter'],
                    y=df['Value'], 
                    marker_color=df['Color'],
                    text=df['Value'],
                    textposition='auto'
                )
            ])
            
            fig.update_layout(
                title='Current Lab Values',
                yaxis_title='Value',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.info("Upload lab reports or enter lab values to see visualizations.")

def main():
    # Professional Header
    st.markdown("<h1 class='main-header'>🏥 MediAI Pro</h1>", unsafe_allow_html=True)
    st.markdown("<h2 class='sub-header'>Intelligent Medical Assistant</h2>", unsafe_allow_html=True)
    
    # Setup Groq API
    api_key = setup_groq_api()
    
    # Initialize analyzer with API key
    analyzer = MedicalAIAnalyzer(api_key)
    
    # Setup sidebar and get patient data
    patient_data = setup_sidebar()
    
    # Perform analysis if requested
    if st.session_state.analyze_clicked:
        with st.spinner("🔄 Analyzing patient data..."):
            analysis_results = {}
            
            analysis_results['comprehensive_analysis'] = analyzer.analyze_patient_data(
                patient_data['patient_info'], 
                patient_data['medical_history'],
                patient_data['lab_data'],
                patient_data['image_description'],
                patient_data['processed_files_text']
            )
            
            st.session_state.analysis_results = analysis_results
            st.session_state.analyze_clicked = False
            
        st.success("✅ Analysis completed!")
        st.rerun()
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["💬 Medical Chat", "📊 Analysis", "📈 Health Insights"])
    
    with tab1:
        display_medical_chat(analyzer)
    
    with tab2:
        display_analysis_dashboard(analyzer, patient_data)
    
    with tab3:
        create_advanced_visualizations(patient_data, analyzer)

if __name__ == "__main__":
    main()
