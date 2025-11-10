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
if 'groq_client' not in st.session_state:
    st.session_state.groq_client = None
if 'api_configured' not in st.session_state:
    st.session_state.api_configured = False
if 'analyze_clicked' not in st.session_state:
    st.session_state.analyze_clicked = False

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
                'Triglycerides': {'low': 0, 'high': 150, 'unit': 'mg/dL', 'critical_low': 0, 'critical_high': 500}
            }
        }

    def call_groq_api(self, messages, model="llama3-70b-8192", max_tokens=1500, temperature=0.3):
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
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"Groq API call failed: {str(e)}")
            return None

    def chat_with_medical_ai(self, message, conversation_history=None, patient_context=None):
        """Advanced medical chat using Groq API"""
        if not self.api_key:
            return "🚫 **Medical AI Service Unavailable**\n\nPlease configure Groq API key in Streamlit secrets to enable AI-powered medical conversations."
        
        try:
            # Build system prompt with medical context
            system_prompt = f"""
            You are Dr. MedAI, an advanced AI medical assistant with the following characteristics:

            **ROLE**: Senior Medical Consultant with 20+ years of clinical experience
            **SPECIALTIES**: Internal Medicine, Family Practice, Emergency Medicine, Patient Education

            **PATIENT CONTEXT**:
            {patient_context if patient_context else 'No specific patient context provided'}

            **COMMUNICATION GUIDELINES**:
            - Be empathetic, professional, and clear in all responses
            - Explain medical concepts in simple, understandable language
            - Provide evidence-based medical information
            - Always maintain patient confidentiality and privacy
            - Show genuine concern for patient wellbeing

            **MEDICAL EXPERTISE**:
            - Symptom analysis and triage guidance
            - Medication information and safety
            - Test result interpretation
            - Chronic disease management
            - Preventive health and wellness
            - Emergency recognition

            **SAFETY PROTOCOLS**:
            - NEVER provide definitive diagnoses
            - ALWAYS recommend consulting healthcare providers for personal medical advice
            - Identify red flag symptoms that require immediate medical attention
            - Suggest appropriate specialist referrals when needed
            - Emphasize when symptoms require emergency care

            **RESPONSE STRUCTURE**:
            1. Acknowledge and understand the patient's concern
            2. Provide educational medical information
            3. Suggest appropriate next steps
            4. Highlight when to seek immediate care
            5. Offer supportive guidance and reassurance

            **REMEMBER**: You are an AI assistant supporting healthcare, not replacing professional medical care.
            """

            # Build messages array
            messages = [
                {"role": "system", "content": system_prompt}
            ]

            # Add conversation history (last 10 messages for context)
            if conversation_history:
                for msg in conversation_history[-10:]:
                    messages.append({"role": msg["role"], "content": msg["content"]})

            # Add current user message
            messages.append({"role": "user", "content": message})

            # Get response from Groq API
            response = self.call_groq_api(messages)
            
            if response:
                return response
            else:
                return "⚠️ **I'm experiencing technical difficulties**\n\nPlease try again in a moment or check your API configuration."

        except Exception as e:
            logger.error(f"Medical AI error: {str(e)}")
            return f"⚠️ **I'm experiencing technical difficulties**\n\nError: {str(e)}\n\nPlease try again in a moment."

    def analyze_lab_results(self, lab_data):
        """AI-powered laboratory results analysis"""
        if not self.api_key:
            return self._basic_lab_analysis(lab_data)
        
        try:
            prompt = f"""
            Analyze these laboratory results as a senior medical consultant:

            LABORATORY DATA:
            {lab_data}

            Please provide a comprehensive medical analysis with:

            **CRITICAL FINDINGS**: Identify any values that require immediate medical attention
            **ABNORMALITIES**: Note all values outside normal ranges
            **CLINICAL INTERPRETATION**: Explain what these results mean medically
            **DIFFERENTIAL DIAGNOSIS**: Suggest possible conditions to consider
            **RECOMMENDED ACTIONS**: Provide clear next steps for follow-up
            **URGENCY LEVEL**: Assess overall urgency (Low/Medium/High)

            Format this as a professional medical consultation note.
            """

            messages = [
                {
                    "role": "system", 
                    "content": "You are a senior pathologist and medical consultant. Provide accurate, professional laboratory analysis with clear clinical correlations and actionable recommendations. Always emphasize patient safety."
                },
                {"role": "user", "content": prompt}
            ]

            response = self.call_groq_api(messages, max_tokens=2000, temperature=0.2)
            
            if response:
                return response
            else:
                return self._basic_lab_analysis(lab_data)
                
        except Exception as e:
            logger.error(f"AI lab analysis failed: {str(e)}")
            return self._basic_lab_analysis(lab_data)
    
    def _basic_lab_analysis(self, lab_data):
        """Basic laboratory analysis fallback"""
        extracted_values = self.extract_lab_values(lab_data)
        
        analysis = "## 🔬 Laboratory Analysis\n\n"
        
        if extracted_values:
            analysis += "### 📊 Laboratory Values\n"
            for test, value in extracted_values.items():
                if test in self.medical_knowledge_base['lab_ranges']:
                    ranges = self.medical_knowledge_base['lab_ranges'][test]
                    if value < ranges['low']:
                        analysis += f"⚠️ **{test}**: {value} {ranges['unit']} (Low - Normal: {ranges['low']}-{ranges['high']})\n"
                    elif value > ranges['high']:
                        analysis += f"⚠️ **{test}**: {value} {ranges['unit']} (High - Normal: {ranges['low']}-{ranges['high']})\n"
                    else:
                        analysis += f"✅ **{test}**: {value} {ranges['unit']} (Normal)\n"
        else:
            analysis += "No specific laboratory values detected in the provided data.\n"
        
        analysis += "\n### 🩺 Recommendations\n"
        analysis += "1. Discuss all findings with your healthcare provider\n"
        analysis += "2. Correlate results with clinical symptoms\n"
        analysis += "3. Follow up on any abnormal findings\n"
        
        return analysis
    
    def analyze_medical_image(self, image_description):
        """AI-powered medical image analysis"""
        if not self.api_key:
            return self._basic_image_analysis(image_description)
        
        try:
            prompt = f"""
            Analyze this medical image description as a board-certified radiologist:

            IMAGE FINDINGS DESCRIPTION:
            {image_description}

            Please provide a comprehensive radiology assessment including:

            **TECHNICAL ASSESSMENT**: Image quality and limitations
            **KEY FINDINGS**: Detailed observations and abnormalities
            **DIFFERENTIAL DIAGNOSIS**: Possible conditions to consider
            **CLINICAL CORRELATION**: How findings relate to patient care
            **RECOMMENDATIONS**: Further imaging, follow-up, or specialist consultation

            Format this as a standard radiology report.
            """

            messages = [
                {
                    "role": "system", 
                    "content": "You are a board-certified radiologist with extensive experience. Provide detailed, professional image interpretation following standard radiology reporting protocols. Always recommend formal radiology consultation for definitive interpretation."
                },
                {"role": "user", "content": prompt}
            ]

            response = self.call_groq_api(messages, max_tokens=1800, temperature=0.2)
            
            if response:
                return response
            else:
                return self._basic_image_analysis(image_description)
                
        except Exception as e:
            logger.error(f"AI image analysis failed: {str(e)}")
            return self._basic_image_analysis(image_description)
    
    def _basic_image_analysis(self, image_description):
        """Basic image analysis fallback"""
        analysis = "## 🖼️ Medical Image Analysis\n\n"
        analysis += f"**Description**: {image_description}\n\n"
        analysis += "### 📋 Assessment Framework\n"
        analysis += "1. Evaluate image quality and technical factors\n"
        analysis += "2. Systematic review of anatomical structures\n"
        analysis += "3. Identification of abnormalities\n"
        analysis += "4. Clinical correlation with patient presentation\n"
        analysis += "5. Recommendations for further evaluation\n\n"
        analysis += "### 🎯 Next Steps\n"
        analysis += "- Formal radiology consultation recommended\n"
        analysis += "- Compare with previous studies if available\n"
        analysis += "- Clinical correlation with symptoms\n"
        
        return analysis
    
    def generate_patient_summary(self, patient_info, medical_history, current_findings):
        """AI-powered comprehensive patient summary"""
        if not self.api_key:
            return self._basic_patient_summary(patient_info, medical_history, current_findings)
        
        try:
            prompt = f"""
            Generate a comprehensive patient summary and care plan:

            PATIENT INFORMATION: {patient_info}
            MEDICAL HISTORY: {medical_history}
            CURRENT FINDINGS: {current_findings}

            Please create a structured medical summary including:

            **COMPREHENSIVE ASSESSMENT**:
            - Current clinical status and problem list
            - Risk stratification and prioritization
            - Overall health assessment

            **TREATMENT PLAN**:
            - Immediate interventions if needed
            - Medication recommendations
            - Lifestyle modifications

            **MONITORING & FOLLOW-UP**:
            - Parameters to track
            - Laboratory monitoring schedule
            - Specialist referrals needed
            - Follow-up timeline

            **PATIENT EDUCATION**:
            - Key discussion points
            - Warning signs to watch for
            - Self-management strategies

            Format as a professional clinical note.
            """

            messages = [
                {
                    "role": "system", 
                    "content": "You are an experienced physician creating comprehensive patient care plans. Focus on evidence-based, practical recommendations with clear follow-up actions. Always emphasize the importance of professional medical follow-up."
                },
                {"role": "user", "content": prompt}
            ]

            response = self.call_groq_api(messages, max_tokens=2500, temperature=0.2)
            
            if response:
                return response
            else:
                return self._basic_patient_summary(patient_info, medical_history, current_findings)
                
        except Exception as e:
            logger.error(f"AI patient summary failed: {str(e)}")
            return self._basic_patient_summary(patient_info, medical_history, current_findings)
    
    def _basic_patient_summary(self, patient_info, medical_history, current_findings):
        """Basic patient summary fallback"""
        analysis = "## 👨‍⚕️ Patient Assessment\n\n"
        analysis += f"**Patient**: {patient_info}\n\n"
        analysis += f"**History**: {medical_history}\n\n"
        analysis += f"**Findings**: {current_findings}\n\n"
        analysis += "### 💡 Recommendations\n"
        analysis += "1. Review all findings with healthcare provider\n"
        analysis += "2. Implement recommended monitoring\n"
        analysis += "3. Schedule appropriate follow-up\n"
        
        return analysis
    
    def extract_lab_values(self, text):
        """Extract laboratory values from text"""
        patterns = {
            'WBC': r'WBC\s*[:]?\s*([\d.]+)',
            'Hemoglobin': r'Hemoglobin\s*[:]?\s*([\d.]+)|Hgb\s*[:]?\s*([\d.]+)',
            'Glucose': r'Glucose\s*[:]?\s*([\d.]+)',
            'Creatinine': r'Creatinine\s*[:]?\s*([\d.]+)',
            'ALT': r'ALT\s*[:]?\s*([\d.]+)',
            'AST': r'AST\s*[:]?\s*([\d.]+)',
            'LDL': r'LDL\s*[:]?\s*([\d.]+)',
            'HDL': r'HDL\s*[:]?\s*([\d.]+)',
            'Triglycerides': r'Triglycerides\s*[:]?\s*([\d.]+)'
        }
        
        extracted_values = {}
        for test, pattern in patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                value = match[0] if match[0] else match[1] if len(match) > 1 else match
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
        
        # Test the API key with a simple request
        try:
            url = "https://api.groq.com/openai/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            data = {
                "messages": [{"role": "user", "content": "Say 'Connected'"}],
                "model": "llama3-8b-8192",
                "max_tokens": 5
            }
            
            response = requests.post(url, headers=headers, json=data, timeout=10)
            
            if response.status_code == 200:
                st.session_state.api_configured = True
                st.session_state.api_key = api_key
                st.sidebar.markdown("<div class='api-status-connected'>✅ Groq API Connected (from secrets)</div>", unsafe_allow_html=True)
                return api_key
            else:
                st.sidebar.markdown("<div class='api-status-disconnected'>❌ Groq API Error</div>", unsafe_allow_html=True)
                st.sidebar.error(f"API returned status: {response.status_code}")
                
        except Exception as e:
            st.sidebar.markdown("<div class='api-status-disconnected'>❌ Groq API Error</div>", unsafe_allow_html=True)
            st.sidebar.error(f"Connection failed: {str(e)}")
    
    else:
        st.sidebar.markdown("<div class='api-status-disconnected'>❌ Groq API Not Configured</div>", unsafe_allow_html=True)
        st.sidebar.info("Please add GROQ_API_KEY to your Streamlit secrets.")
    
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
            st.error("🤖 AI: **INACTIVE**")
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
            "Medical History & Medications", 
            height=120,
            value="Hypertension, Type 2 Diabetes, Hyperlipidemia. Medications: Lisinopril 10mg, Metformin 500mg.",
            key="medical_history"
        )
        
        st.header("📁 Medical Data Upload")
        
        tab1, tab2, tab3 = st.tabs(["Lab Reports", "Images", "Documents"])
        
        with tab1:
            st.subheader("Laboratory Results")
            lab_files = st.file_uploader("Upload Lab Reports", 
                                       type=['pdf', 'docx', 'txt', 'jpg', 'png'], 
                                       key="lab_uploader",
                                       accept_multiple_files=True)
            lab_text_input = st.text_area("Paste Lab Results:", height=100,
                                        value="""CBC: WBC 8.2, RBC 4.5, Hgb 14.2, Hct 42%, Platelets 250
Chemistry: Glucose 110, Creatinine 1.1, BUN 18, ALT 25, AST 22
Lipid Panel: Total Cholesterol 185, LDL 110, HDL 45, Triglycerides 150""",
                                        key="lab_text")
        
        with tab2:
            st.subheader("Medical Images")
            image_files = st.file_uploader("Upload Medical Images", 
                                         type=['png', 'jpg', 'jpeg'], 
                                         key="image_uploader",
                                         accept_multiple_files=True)
            image_description = st.text_area("Image Findings Description:", height=100,
                                           value="Chest X-ray: Mild cardiomegaly, clear lung fields, no acute findings.",
                                           key="image_desc")
        
        with tab3:
            st.subheader("Medical Reports")
            report_files = st.file_uploader("Upload Medical Reports", 
                                          type=['pdf', 'docx', 'txt'], 
                                          key="report_uploader",
                                          accept_multiple_files=True)
            report_text_input = st.text_area("Report Content:", height=100,
                                           value="ECG: Normal sinus rhythm, rate 72 bpm. Normal axis. No ST-T changes.",
                                           key="report_text")
        
        # Analysis Controls
        st.markdown("<div class='medical-card'>", unsafe_allow_html=True)
        st.header("🔍 Analysis Controls")
        
        col1, col2 = st.columns(2)
        with col1:
            analyze_disabled = not st.session_state.api_configured
            if st.button("🚀 **AI Analysis**", use_container_width=True, type="primary", disabled=analyze_disabled):
                if st.session_state.api_configured:
                    st.session_state.analyze_clicked = True
                else:
                    st.sidebar.error("Please configure Groq API first")
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
            'report_data': report_text_input,
            'patient_context': f"{patient_name}, {patient_age} years, {patient_gender}. History: {medical_history}"
        }

def display_medical_chat(analyzer, patient_context):
    """Display advanced medical chat interface"""
    st.header("💬 Dr. MedAI - Medical Conversation Assistant")
    
    # API status indicator
    if st.session_state.api_configured:
        st.success("✅ AI Assistant: **ACTIVE** - You can now have natural medical conversations")
    else:
        st.error("❌ AI Assistant: **INACTIVE** - Configure Groq API in sidebar to enable conversations")
    
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
        "Ask about symptoms, medications, test results, or general health concerns:",
        placeholder="Describe your symptoms or ask a medical question...",
        key="chat_input",
        disabled=not st.session_state.api_configured
    )
    
    col1, col2 = st.columns([3, 1])
    with col1:
        if st.button("Send Message", use_container_width=True, disabled=not st.session_state.api_configured) and user_input:
            process_user_message(analyzer, user_input, patient_context)
    with col2:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    
    # Quick action buttons
    if st.session_state.api_configured:
        st.subheader("💡 Quick Medical Questions")
        quick_col1, quick_col2 = st.columns(2)
        
        with quick_col1:
            if st.button("🤒 Discuss Symptoms", use_container_width=True):
                process_user_message(analyzer, "I'd like to discuss some symptoms I'm experiencing. Can you help me understand when to seek medical attention?", patient_context)
            if st.button("💊 Medication Questions", use_container_width=True):
                process_user_message(analyzer, "I have questions about medication safety and management. What should I know?", patient_context)
        
        with quick_col2:
            if st.button("🩺 Test Results Help", use_container_width=True):
                process_user_message(analyzer, "Can you help me understand how to interpret common medical test results?", patient_context)
            if st.button("🍎 Health & Wellness", use_container_width=True):
                process_user_message(analyzer, "What are some evidence-based recommendations for maintaining good health and preventing disease?", patient_context)

def process_user_message(analyzer, user_input, patient_context):
    """Process user message and generate AI response"""
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Generate AI response
    with st.spinner("Dr. MedAI is analyzing your question..."):
        ai_response = analyzer.chat_with_medical_ai(
            user_input, 
            st.session_state.messages,
            patient_context
        )
        
        # Add AI response to chat history
        st.session_state.messages.append({"role": "assistant", "content": ai_response})
    
    st.rerun()

def display_analysis_dashboard(analyzer):
    """Display professional medical dashboard"""
    st.header("📊 Medical Analysis Dashboard")
    
    if not st.session_state.analysis_results:
        st.info("""
        🏥 **Welcome to MediAI Pro - AI-Powered Medical Assistant**
        
        To begin comprehensive medical analysis:
        1. ✅ Configure Groq API in the sidebar (essential for AI features)
        2. 👤 Enter patient information
        3. 📁 Upload or paste medical data
        4. 🚀 Click **'AI Analysis'** for comprehensive analysis
        5. 💬 Use the chat for natural medical conversations
        """)
        return
    
    # Summary Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Analyses Completed", len(st.session_state.analysis_results))
    
    with col2:
        urgency = "Low"
        results_text = " ".join([str(result) for result in st.session_state.analysis_results.values()]).lower()
        if "critical" in results_text or "urgent" in results_text:
            urgency = "High"
        elif "abnormal" in results_text:
            urgency = "Medium"
        st.metric("Clinical Urgency", urgency)
    
    with col3:
        st.metric("AI Powered", "Yes" if st.session_state.api_configured else "No")
    
    with col4:
        st.metric("Report Status", "Ready")
    
    # Detailed Analysis Sections
    if 'lab_analysis' in st.session_state.analysis_results:
        with st.expander("🔬 **AI Laboratory Analysis**", expanded=True):
            st.markdown(st.session_state.analysis_results['lab_analysis'])
    
    if 'image_analysis' in st.session_state.analysis_results:
        with st.expander("🖼️ **AI Image Analysis**", expanded=True):
            st.markdown(st.session_state.analysis_results['image_analysis'])
    
    if 'patient_summary' in st.session_state.analysis_results:
        with st.expander("👨‍⚕️ **AI Patient Assessment**", expanded=True):
            st.markdown(st.session_state.analysis_results['patient_summary'])

def create_advanced_visualizations():
    """Create professional medical visualizations"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🩸 Laboratory Trends")
        lab_data = {
            'Parameter': ['WBC', 'Hgb', 'Glucose', 'Creatinine', 'ALT', 'LDL'],
            'Current': [8.2, 14.2, 110, 1.1, 25, 110],
            'Previous': [7.8, 13.8, 98, 0.9, 22, 125],
            'Normal Low': [4.0, 12.0, 70, 0.6, 7, 0],
            'Normal High': [11.0, 16.0, 100, 1.3, 55, 100]
        }
        
        df = pd.DataFrame(lab_data)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Parameter'], y=df['Normal High'], 
                               mode='lines', name='Upper Limit', line=dict(dash='dash', color='red')))
        fig.add_trace(go.Scatter(x=df['Parameter'], y=df['Normal Low'], 
                               mode='lines', name='Lower Limit', line=dict(dash='dash', color='red')))
        fig.add_trace(go.Bar(name='Current', x=df['Parameter'], y=df['Current'], marker_color='blue'))
        fig.add_trace(go.Bar(name='Previous', x=df['Parameter'], y=df['Previous'], marker_color='lightblue'))
        
        fig.update_layout(
            title='Laboratory Parameters Trend',
            barmode='group',
            showlegend=True,
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 Health Risk Assessment")
        categories = ['Cardiovascular', 'Metabolic', 'Hepatic', 'Renal', 'Hematological']
        risk_scores = [65, 45, 20, 30, 15]
        
        fig = go.Figure(data=[go.Bar(x=categories, y=risk_scores, 
                                   marker_color=['red', 'orange', 'yellow', 'lightgreen', 'green'])])
        fig.update_layout(
            title='System-based Risk Assessment',
            yaxis_title='Risk Score (%)',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

def main():
    # Professional Header
    st.markdown("<h1 class='main-header'>🏥 MediAI Pro</h1>", unsafe_allow_html=True)
    st.markdown("<h2 class='sub-header'>Advanced AI-Powered Medical Assistant</h2>", unsafe_allow_html=True)
    
    # Setup Groq API
    api_key = setup_groq_api()
    
    # Initialize analyzer with API key
    analyzer = MedicalAIAnalyzer(api_key)
    
    # Setup sidebar and get patient data
    patient_data = setup_sidebar()
    
    # Perform analysis if requested
    if st.session_state.analyze_clicked:
        with st.spinner("🔄 Performing AI-powered medical analysis..."):
            analysis_results = {}
            
            # Analyze laboratory data
            if patient_data['lab_data']:
                analysis_results['lab_analysis'] = analyzer.analyze_lab_results(patient_data['lab_data'])
            
            # Analyze medical images
            if patient_data['image_description']:
                analysis_results['image_analysis'] = analyzer.analyze_medical_image(patient_data['image_description'])
            
            # Generate patient summary
            analysis_results['patient_summary'] = analyzer.generate_patient_summary(
                patient_data['patient_info'], 
                patient_data['medical_history'],
                f"Lab: {patient_data['lab_data'][:200]}, Imaging: {patient_data['image_description'][:200]}"
            )
            
            st.session_state.analysis_results = analysis_results
            st.session_state.analyze_clicked = False
            
        st.success("✅ AI-powered medical analysis completed successfully!")
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["💬 AI Chat", "📊 Dashboard", "📈 Analytics", "📋 Report"])
    
    with tab1:
        display_medical_chat(analyzer, patient_data['patient_context'])
    
    with tab2:
        display_analysis_dashboard(analyzer)
    
    with tab3:
        st.header("📈 Medical Analytics")
        create_advanced_visualizations()
    
    with tab4:
        st.header("📋 AI-Generated Medical Report")
        if st.session_state.analysis_results:
            report_content = f"""
# 🏥 AI-POWERED MEDICAL ANALYSIS REPORT

## Patient Information
{patient_data['patient_info']}

## Executive Summary
Comprehensive medical analysis performed using advanced AI (Groq Llama 3 70B) with professional medical knowledge base.

## Detailed Analysis

### Laboratory Findings
{st.session_state.analysis_results.get('lab_analysis', 'No laboratory analysis available')}

### Imaging Analysis  
{st.session_state.analysis_results.get('image_analysis', 'No imaging analysis available')}

### Patient Assessment
{st.session_state.analysis_results.get('patient_summary', 'No patient summary available')}

## AI Clinical Recommendations

### Immediate Actions
- Review all AI-generated findings with qualified healthcare providers
- Address any critical or urgent findings promptly
- Ensure comprehensive understanding of health status

### Follow-up Plan
- Schedule appropriate specialist consultations based on findings
- Implement recommended monitoring and testing
- Adjust treatment plans as clinically indicated

### Long-term Health Strategy
- Continue management of chronic conditions
- Focus on evidence-based preventive care
- Regular health maintenance and screening

---
*Generated by MediAI Pro with Groq AI Assistant on {datetime.now().strftime('%B %d, %Y at %H:%M')}*

**Medical Disclaimer**: This AI-generated report provides general medical information and should be reviewed by qualified healthcare professionals. Always seek professional medical advice for personal health concerns and treatment decisions.
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
                    label="📊 Export Analysis Data",
                    data=json.dumps(st.session_state.analysis_results, indent=2),
                    file_name=f"Analysis_Data_{datetime.now().strftime('%Y%m%d')}.json",
                    mime="application/json",
                    use_container_width=True
                )
        else:
            st.info("Generate AI analysis first to view the comprehensive medical report.")

if __name__ == "__main__":
    main()
