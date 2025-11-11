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

class MedicalAIAnalyzer:
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.medical_knowledge_base = self._initialize_medical_knowledge()
    
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

    def call_groq_api(self, messages, model="llama3-8b-8192", max_tokens=1500, temperature=0.7):
        """Make direct API call to Groq"""
        if not self.api_key:
            return None
        
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
        
        try:
            response = requests.post(url, headers=headers, json=data, timeout=30)
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                logger.error(f"Groq API error: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            logger.error(f"Groq API call failed: {str(e)}")
            return None

    def chat_with_medical_ai(self, message, patient_context=""):
        """Advanced medical chat using Groq API with intelligent patient context"""
        if not self.api_key:
            return self._intelligent_fallback_response(message, patient_context)
        
        try:
            # Build intelligent system prompt
            system_prompt = f"""You are Dr. MedAI, an advanced AI medical assistant. You have access to the following patient information:

PATIENT CONTEXT:
{patient_context if patient_context else 'No specific patient information provided yet.'}

IMPORTANT INSTRUCTIONS:
1. FIRST analyze the patient context above to understand their medical situation
2. If patient has provided lab results, symptoms, or medical history, focus your response on analyzing that specific information
3. Provide personalized medical guidance based on their actual data
4. For general greetings like "hi", respond naturally and ask about their health concerns or offer to analyze their medical data
5. Always be conversational, empathetic, and professional
6. When analyzing medical data, provide specific insights about what the values mean
7. Suggest appropriate next steps based on the analysis
8. Always recommend consulting healthcare providers for definitive medical advice

RESPONSE GUIDELINES:
- Be natural and conversational
- Show you've understood their medical context
- Provide specific, actionable insights
- Ask clarifying questions when needed
- Never provide definitive diagnoses
- Always emphasize professional medical consultation"""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message}
            ]

            # Get response from Groq API
            response = self.call_groq_api(messages, model="llama3-8b-8192", temperature=0.7)
            
            if response:
                return response
            else:
                return self._intelligent_fallback_response(message, patient_context)

        except Exception as e:
            logger.error(f"Medical AI error: {str(e)}")
            return self._intelligent_fallback_response(message, patient_context)

    def _intelligent_fallback_response(self, message, patient_context):
        """Provide intelligent fallback responses based on patient context"""
        message_lower = message.lower()
        
        # Check if we have patient context
        if patient_context and "iron" in patient_context.lower():
            if any(greet in message_lower for greet in ['hi', 'hello', 'hey']):
                return """👋 Hello! I can see you've mentioned iron deficiency in your medical context. 

I'd be happy to help you understand:
• What iron deficiency means for your health
• Common symptoms and causes
• Dietary recommendations for iron
• When to follow up with your doctor

What specific questions do you have about iron deficiency or your health concerns?"""
            elif any(word in message_lower for word in ['iron', 'deficiency', 'anemia', 'ferritin']):
                return """🔬 **Iron Deficiency Analysis**

Based on your mention of iron deficiency, here's what you should know:

**Understanding Iron Deficiency:**
- Iron is essential for hemoglobin production
- Deficiency can cause fatigue, weakness, and pale skin
- Common in adolescents, women, and vegetarians

**Key Laboratory Values to Monitor:**
- **Hemoglobin**: Below 12 g/dL suggests anemia
- **Ferritin**: Below 15 ng/mL indicates iron deficiency
- **Iron levels**: Below 60 μg/dL
- **TIBC**: Usually elevated in deficiency

**Recommended Actions:**
1. **Consult your doctor** for proper diagnosis
2. **Consider iron supplements** if recommended
3. **Iron-rich foods**: red meat, spinach, lentils, fortified cereals
4. **Vitamin C** helps iron absorption
5. **Follow-up testing** in 2-3 months

**When to Seek Immediate Care:**
- Severe fatigue preventing daily activities
- Shortness of breath at rest
- Chest pain or palpitations

Would you like me to analyze specific lab values or discuss dietary recommendations?"""
        
        # General greeting response
        if any(greet in message_lower for greet in ['hi', 'hello', 'hey']):
            if patient_context:
                return f"""👋 Hello! I can see we have some medical information about you. 

I'm here to help you understand:
• Your medical test results
• Symptoms and health concerns
• Medication questions
• General health guidance

What would you like to discuss about your health today?"""
            else:
                return """👋 Hello! I'm Dr. MedAI, your medical assistant.

I can help you with:
• Analyzing medical test results
• Understanding symptoms
• Medication information
• Health and wellness guidance

Please share your medical concerns or test results, and I'll provide helpful information and analysis.

You can also enter your information in the sidebar to get personalized guidance!"""

        # Symptom analysis
        elif any(symptom in message_lower for symptom in ['tired', 'fatigue', 'weak', 'exhaust']):
            return """😴 **Fatigue Analysis**

Fatigue can have many causes. Let me help you understand:

**Common Causes of Fatigue:**
• **Iron deficiency anemia** - Very common, especially in your age group
• **Sleep issues** - Quality and duration matter
• **Nutritional deficiencies** - Iron, B12, Vitamin D
• **Thyroid problems** - Affects energy metabolism
• **Stress and mental health** - Significant impact on energy

**Questions to Consider:**
1. How long have you been feeling tired?
2. Is it better or worse at certain times?
3. Any other symptoms like dizziness or pale skin?
4. Have you had recent blood tests?

**Next Steps:**
• Discuss these symptoms with your doctor
• Consider complete blood count (CBC) testing
• Review your sleep habits and diet
• Monitor your energy patterns

Would you like to discuss specific test results or other symptoms?"""

        # General medical questions
        else:
            return """🏥 **Medical Guidance**

I'd be happy to help with your health questions. To provide the most helpful information, please:

1. **Share your specific symptoms or concerns**
2. **Tell me about any test results you have**
3. **Describe your medical history if relevant**

I can help you understand:
• What your symptoms might mean
• How to interpret test results
• When to seek medical attention
• General health recommendations

What specific health concern would you like to discuss?"""

    def analyze_patient_data(self, patient_info, medical_history, lab_data, image_description):
        """Comprehensive analysis of all patient data"""
        if not self.api_key:
            return self._basic_patient_analysis(patient_info, medical_history, lab_data, image_description)
        
        try:
            prompt = f"""
            Perform a comprehensive medical analysis of this patient case:

            PATIENT INFORMATION:
            {patient_info}

            MEDICAL HISTORY:
            {medical_history}

            LABORATORY DATA:
            {lab_data}

            IMAGING FINDINGS:
            {image_description}

            Please provide a detailed analysis including:

            **PATIENT OVERVIEW**: Summary of the clinical case
            **KEY FINDINGS**: Important abnormalities or notable results
            **RISK ASSESSMENT**: Potential health risks based on data
            **CLINICAL CORRELATION**: How findings relate to patient presentation
            **RECOMMENDATIONS**: Clear next steps for evaluation and management
            **URGENCY LEVEL**: Assessment of how quickly action is needed

            Format as a professional medical consultation note.
            """

            messages = [
                {
                    "role": "system", 
                    "content": "You are an experienced physician analyzing comprehensive patient data. Provide thorough, evidence-based analysis with clear clinical recommendations. Focus on actionable insights and patient safety."
                },
                {"role": "user", "content": prompt}
            ]

            response = self.call_groq_api(messages, max_tokens=2000, temperature=0.3)
            
            if response:
                return response
            else:
                return self._basic_patient_analysis(patient_info, medical_history, lab_data, image_description)
                
        except Exception as e:
            logger.error(f"Comprehensive analysis failed: {str(e)}")
            return self._basic_patient_analysis(patient_info, medical_history, lab_data, image_description)

    def _basic_patient_analysis(self, patient_info, medical_history, lab_data, image_description):
        """Basic analysis when AI is unavailable"""
        analysis = "## 📊 Patient Data Analysis\n\n"
        
        analysis += f"**Patient**: {patient_info}\n\n"
        
        if medical_history and medical_history.strip():
            analysis += f"**Medical History**: {medical_history}\n\n"
        
        # Analyze lab data
        if lab_data and lab_data.strip():
            lab_values = self.extract_lab_values(lab_data)
            if lab_values:
                analysis += "### 🔬 Laboratory Findings\n"
                for test, value in lab_values.items():
                    if test in self.medical_knowledge_base['lab_ranges']:
                        ranges = self.medical_knowledge_base['lab_ranges'][test]
                        if value < ranges['low']:
                            analysis += f"⚠️ **{test}**: {value} {ranges['unit']} **(LOW)** - Normal range: {ranges['low']}-{ranges['high']}\n"
                        elif value > ranges['high']:
                            analysis += f"⚠️ **{test}**: {value} {ranges['unit']} **(HIGH)** - Normal range: {ranges['low']}-{ranges['high']}\n"
                        else:
                            analysis += f"✅ **{test}**: {value} {ranges['unit']} (Normal)\n"
        
        if image_description and image_description.strip():
            analysis += f"\n### 🖼️ Imaging Notes\n{image_description}\n"
        
        analysis += "\n### 💡 Recommendations\n"
        analysis += "1. **Review findings** with your healthcare provider\n"
        analysis += "2. **Discuss any abnormal values** for proper interpretation\n"
        analysis += "3. **Follow up** as recommended based on clinical context\n"
        analysis += "4. **Maintain records** for future reference\n"
        
        return analysis

    def extract_lab_values(self, text):
        """Extract laboratory values from text"""
        patterns = {
            'WBC': r'WBC\s*[:]?\s*([\d.]+)',
            'RBC': r'RBC\s*[:]?\s*([\d.]+)',
            'Hemoglobin': r'Hemoglobin\s*[:]?\s*([\d.]+)|Hgb\s*[:]?\s*([\d.]+)|Hb\s*[:]?\s*([\d.]+)',
            'Hematocrit': r'Hematocrit\s*[:]?\s*([\d.]+)|Hct\s*[:]?\s*([\d.]+)',
            'Platelets': r'Platelets\s*[:]?\s*([\d.]+)',
            'Glucose': r'Glucose\s*[:]?\s*([\d.]+)',
            'Creatinine': r'Creatinine\s*[:]?\s*([\d.]+)',
            'BUN': r'BUN\s*[:]?\s*([\d.]+)',
            'ALT': r'ALT\s*[:]?\s*([\d.]+)',
            'AST': r'AST\s*[:]?\s*([\d.]+)',
            'Total Cholesterol': r'Total Cholesterol\s*[:]?\s*([\d.]+)|Cholesterol\s*[:]?\s*([\d.]+)',
            'LDL': r'LDL\s*[:]?\s*([\d.]+)',
            'HDL': r'HDL\s*[:]?\s*([\d.]+)',
            'Triglycerides': r'Triglycerides\s*[:]?\s*([\d.]+)',
            'Iron': r'Iron\s*[:]?\s*([\d.]+)',
            'Ferritin': r'Ferritin\s*[:]?\s*([\d.]+)',
            'TIBC': r'TIBC\s*[:]?\s*([\d.]+)',
            'Transferrin Saturation': r'Transferrin Saturation\s*[:]?\s*([\d.]+)|Sat\s*[:]?\s*([\d.]+)'
        }
        
        extracted_values = {}
        for test, pattern in patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                # Find the first non-empty match in the tuple
                for value in match:
                    if value:
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
        
        # Validate the API key format
        if not api_key or not api_key.startswith('gsk_'):
            st.sidebar.markdown("<div class='api-status-disconnected'>❌ Invalid API Key Format</div>", unsafe_allow_html=True)
            st.sidebar.error("API key should start with 'gsk_'. Please check your Streamlit secrets.")
            return None
        
        # Test the API key with a simple request
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
                st.sidebar.success("Advanced AI features are now active!")
                return api_key
            else:
                st.sidebar.markdown("<div class='api-status-disconnected'>❌ API Authentication Failed</div>", unsafe_allow_html=True)
                st.sidebar.error(f"Status: {response.status_code}. Please check your API key.")
                return None
                
        except requests.exceptions.Timeout:
            st.sidebar.markdown("<div class='api-status-disconnected'>❌ Connection Timeout</div>", unsafe_allow_html=True)
            st.sidebar.error("Connection to Groq API timed out. Using basic mode.")
        except Exception as e:
            st.sidebar.markdown("<div class='api-status-disconnected'>❌ Connection Error</div>", unsafe_allow_html=True)
            st.sidebar.error(f"Connection failed: {str(e)}")
    
    else:
        st.sidebar.markdown("<div class='api-status-disconnected'>❌ API Key Not Found</div>", unsafe_allow_html=True)
        st.sidebar.info("""
        To enable advanced AI features:
        1. Get a Groq API key from https://console.groq.com
        2. Add it to your Streamlit secrets as GROQ_API_KEY
        3. Redeploy the app
        """)
    
    return None

def setup_sidebar():
    """Setup the professional medical sidebar"""
    with st.sidebar:
        st.markdown("<div class='medical-card'>", unsafe_allow_html=True)
        st.header("🏥 MediAI Pro")
        st.markdown("**AI-Powered Medical Assistant**")
        if st.session_state.api_configured:
            st.success("🤖 AI: **ACTIVE**")
        else:
            st.warning("🤖 AI: **BASIC MODE**")
            st.info("Configure API for advanced AI features")
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.header("👤 Patient Information")
        
        col1, col2 = st.columns(2)
        with col1:
            patient_id = st.text_input("Patient ID", value="PT-001", key="patient_id")
            patient_age = st.number_input("Age", min_value=0, max_value=120, value=17, key="patient_age")
        with col2:
            patient_name = st.text_input("Patient Name", value="awais", key="patient_name")
            patient_gender = st.selectbox("Gender", ["Male", "Female", "Other"], key="patient_gender")
        
        st.header("📋 Medical History")
        medical_history = st.text_area(
            "Medical History & Symptoms", 
            height=100,
            placeholder="Example: Iron deficiency, fatigue, frequent headaches...",
            key="medical_history"
        )
        
        st.header("📁 Medical Data Upload")
        
        tab1, tab2 = st.tabs(["Lab Reports", "Medical Notes"])
        
        with tab1:
            st.subheader("Laboratory Results")
            lab_files = st.file_uploader("Upload Lab Reports", 
                                       type=['pdf', 'docx', 'txt', 'jpg', 'png'], 
                                       key="lab_uploader",
                                       accept_multiple_files=True)
            lab_text_input = st.text_area("Paste Lab Results:", height=120,
                                        placeholder="""Example:
Hemoglobin: 11.2 g/dL (Low)
Ferritin: 12 ng/mL (Low)
Iron: 45 μg/dL (Low)
WBC: 6.8
Platelets: 245""",
                                        key="lab_text")
        
        with tab2:
            st.subheader("Medical Notes & Symptoms")
            image_description = st.text_area("Medical Notes & Findings:", height=120,
                                           placeholder="""Example:
Patient reports fatigue and pale skin.
Diagnosed with iron deficiency anemia.
Recommended iron supplements and dietary changes.""",
                                           key="image_desc")
        
        # Build patient context
        patient_context = f"""
Patient: {patient_name}, {patient_age} years, {patient_gender}
Medical History: {medical_history if medical_history else 'Not specified'}
Laboratory Data: {lab_text_input if lab_text_input else 'Not provided'}
Medical Notes: {image_description if image_description else 'Not provided'}
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
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
        
        return {
            'patient_info': f"ID: {patient_id}, Name: {patient_name}, Age: {patient_age}, Gender: {patient_gender}",
            'medical_history': medical_history,
            'lab_data': lab_text_input,
            'image_description': image_description,
            'patient_context': patient_context
        }

def display_medical_chat(analyzer):
    """Display advanced medical chat interface"""
    st.header("💬 Dr. MedAI - Medical Conversation Assistant")
    
    # API status indicator
    if st.session_state.api_configured:
        st.success("✅ AI Assistant: **ACTIVE** - Advanced AI conversations enabled")
    else:
        st.warning("🟡 AI Assistant: **BASIC MODE** - Using medical knowledge base")
    
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
        placeholder="Example: Can you explain my iron deficiency results? What should I do about my fatigue?",
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
            process_user_message(analyzer, "I'd like to discuss my symptoms and what they might mean.")
        if st.button("💊 Medication Questions", use_container_width=True):
            process_user_message(analyzer, "I have questions about medications and supplements.")
    
    with quick_col2:
        if st.button("🔬 Analyze Test Results", use_container_width=True):
            process_user_message(analyzer, "Can you help me understand my medical test results?")
        if st.button("🍎 Health Recommendations", use_container_width=True):
            process_user_message(analyzer, "What lifestyle changes or dietary recommendations do you suggest for my condition?")

def process_user_message(analyzer, user_input):
    """Process user message and generate AI response"""
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Generate AI response
    with st.spinner("Dr. MedAI is analyzing your question..."):
        ai_response = analyzer.chat_with_medical_ai(
            user_input, 
            st.session_state.patient_context
        )
        
        # Add AI response to chat history
        st.session_state.messages.append({"role": "assistant", "content": ai_response})
    
    st.rerun()

def display_analysis_dashboard(analyzer):
    """Display professional medical dashboard"""
    st.header("📊 Medical Analysis Dashboard")
    
    if not st.session_state.analysis_results:
        st.info("""
        🏥 **Welcome to MediAI Pro - Medical Assistant**
        
        To begin medical analysis:
        1. 👤 Enter patient information in the sidebar
        2. 📋 Add medical history, symptoms, or test results
        3. 🚀 Click **'Analyze Patient Data'** for comprehensive analysis
        4. 💬 Use the chat for personalized medical conversations
        
        *The more information you provide, the better I can assist you!*
        """)
        return
    
    # Summary Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Analysis Completed", "✓")
    
    with col2:
        st.metric("AI Powered", "Yes" if st.session_state.api_configured else "Basic")
    
    with col3:
        st.metric("Patient Data", "Loaded")
    
    with col4:
        st.metric("Report Status", "Ready")
    
    # Detailed Analysis Sections
    if 'comprehensive_analysis' in st.session_state.analysis_results:
        with st.expander("📋 **Comprehensive Patient Analysis**", expanded=True):
            st.markdown(st.session_state.analysis_results['comprehensive_analysis'])

def create_advanced_visualizations():
    """Create professional medical visualizations"""
    st.header("📈 Health Analytics & Trends")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🩸 Common Blood Values")
        lab_data = {
            'Parameter': ['Hemoglobin', 'WBC', 'Platelets', 'Glucose', 'Iron'],
            'Your Value': [11.2, 6.8, 245, 95, 45],
            'Normal Low': [12.0, 4.0, 150, 70, 60],
            'Normal High': [16.0, 11.0, 450, 100, 170]
        }
        
        df = pd.DataFrame(lab_data)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Parameter'], y=df['Normal High'], 
                               mode='lines', name='Upper Limit', line=dict(dash='dash', color='red')))
        fig.add_trace(go.Scatter(x=df['Parameter'], y=df['Normal Low'], 
                               mode='lines', name='Lower Limit', line=dict(dash='dash', color='red')))
        fig.add_trace(go.Bar(name='Your Values', x=df['Parameter'], y=df['Your Value'], 
                           marker_color=['red', 'green', 'green', 'green', 'red']))
        
        fig.update_layout(
            title='Laboratory Values Overview',
            yaxis_title='Value',
            showlegend=True,
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 Iron Deficiency Impact")
        categories = ['Energy Levels', 'Cognitive Function', 'Physical Performance', 'Immune Function']
        impact_scores = [85, 60, 75, 65]
        
        fig = go.Figure(data=[go.Bar(x=categories, y=impact_scores, 
                                   marker_color=['red', 'orange', 'orange', 'yellow'])])
        fig.update_layout(
            title='Common Effects of Iron Deficiency',
            yaxis_title='Impact Score (%)',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

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
        with st.spinner("🔄 Performing comprehensive medical analysis..."):
            analysis_results = {}
            
            # Perform comprehensive analysis
            analysis_results['comprehensive_analysis'] = analyzer.analyze_patient_data(
                patient_data['patient_info'], 
                patient_data['medical_history'],
                patient_data['lab_data'],
                patient_data['image_description']
            )
            
            st.session_state.analysis_results = analysis_results
            st.session_state.analyze_clicked = False
            
        st.success("✅ Comprehensive medical analysis completed!")
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["💬 Medical Chat", "📊 Analysis", "📈 Health Insights", "📋 Report"])
    
    with tab1:
        display_medical_chat(analyzer)
    
    with tab2:
        display_analysis_dashboard(analyzer)
    
    with tab3:
        create_advanced_visualizations()
    
    with tab4:
        st.header("📋 Medical Summary Report")
        if st.session_state.analysis_results:
            report_content = f"""
# 🏥 MEDICAL ANALYSIS REPORT

## Patient Information
{patient_data['patient_info']}

## Comprehensive Analysis
{st.session_state.analysis_results.get('comprehensive_analysis', 'No analysis available')}

## Key Recommendations

### Immediate Actions
- Review all findings with your healthcare provider
- Address any identified deficiencies or abnormalities
- Implement recommended lifestyle modifications

### Follow-up Plan
- Schedule appropriate medical follow-up
- Monitor symptoms and progress
- Repeat testing as clinically indicated

### Long-term Health Strategy
- Focus on evidence-based preventive care
- Maintain regular health monitoring
- Continue management of chronic conditions

---
*Generated by MediAI Pro on {datetime.now().strftime('%B %d, %Y at %H:%M')}*

**Medical Disclaimer**: This analysis provides educational information and should be reviewed by qualified healthcare professionals. Always seek professional medical advice for personal health concerns.
"""
            st.markdown(report_content)
            
            # Download options
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    label="📥 Download Full Report",
                    data=report_content,
                    file_name=f"Medical_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                    mime="text/markdown",
                    use_container_width=True
                )
            with col2:
                st.download_button(
                    label="📊 Export Analysis",
                    data=json.dumps(st.session_state.analysis_results, indent=2),
                    file_name=f"Analysis_{datetime.now().strftime('%Y%m%d')}.json",
                    mime="application/json",
                    use_container_width=True
                )
        else:
            st.info("Click 'Analyze Patient Data' in the sidebar to generate a comprehensive medical report.")

if __name__ == "__main__":
    main()
