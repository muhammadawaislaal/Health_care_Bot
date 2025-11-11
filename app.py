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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="MediAI Pro - Advanced Medical Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header { font-size: 3rem; color: #1f77b4; text-align: center; margin-bottom: 1rem; font-weight: bold; }
    .sub-header { font-size: 1.5rem; color: #2e86ab; text-align: center; margin-bottom: 2rem; }
    .medical-card { background-color: #f8f9fa; border-radius: 15px; padding: 1.5rem; border-left: 5px solid #1f77b4; margin-bottom: 1rem; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); }
    .chat-container { background-color: #f0f2f6; border-radius: 15px; padding: 1rem; margin-bottom: 1rem; max-height: 500px; overflow-y: auto; }
    .user-message { background-color: #007bff; color: white; padding: 1rem; border-radius: 15px; margin: 0.5rem 0; max-width: 80%; margin-left: auto; word-wrap: break-word; }
    .assistant-message { background-color: #ffffff; color: #333; padding: 1rem; border-radius: 15px; margin: 0.5rem 0; max-width: 80%; margin-right: auto; border: 1px solid #e0e0e0; word-wrap: break-word; }
    .stButton button { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border: none; border-radius: 10px; padding: 0.5rem 1rem; font-weight: bold; }
    .api-status-connected { background-color: #d4edda; color: #155724; padding: 0.5rem; border-radius: 5px; border: 1px solid #c3e6cb; }
    .api-status-disconnected { background-color: #f8d7da; color: #721c24; padding: 0.5rem; border-radius: 5px; border: 1px solid #f5c6cb; }
</style>
""", unsafe_allow_html=True)

if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {}
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
if 'last_api_call' not in st.session_state:
    st.session_state.last_api_call = 0

class MedicalAIAnalyzer:
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.medical_knowledge_base = self._initialize_medical_knowledge()
        self.available_models = ["llama-3.1-8b-instant", "mixtral-8x7b-32768", "gemma2-9b-it"]
        self.current_model = "llama-3.1-8b-instant"
        self.rate_limit_delay = 2

    def _initialize_medical_knowledge(self):
        return {
            'lab_ranges': {
                'WBC': {'low': 4.0, 'high': 11.0, 'unit': '10^3/μL'},
                'RBC': {'low': 4.2, 'high': 5.8, 'unit': '10^6/μL'},
                'Hemoglobin': {'low': 12.0, 'high': 16.0, 'unit': 'g/dL'},
                'Hematocrit': {'low': 36, 'high': 48, 'unit': '%'},
                'Platelets': {'low': 150, 'high': 450, 'unit': '10^3/μL'},
                'Glucose': {'low': 70, 'high': 100, 'unit': 'mg/dL'},
                'Creatinine': {'low': 0.6, 'high': 1.3, 'unit': 'mg/dL'},
                'ALT': {'low': 7, 'high': 55, 'unit': 'U/L'},
                'AST': {'low': 8, 'high': 48, 'unit': 'U/L'},
                'Iron': {'low': 60, 'high': 170, 'unit': 'μg/dL'},
                'Ferritin': {'low': 15, 'high': 150, 'unit': 'ng/mL'},
            }
        }

    def call_groq_api(self, messages, max_tokens=800, temperature=0.7):
        if not self.api_key:
            return None
        
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
            "model": self.current_model,
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
            else:
                return None
        except Exception as e:
            return None

    def extract_text_from_pdf(self, pdf_file):
        try:
            with pdfplumber.open(pdf_file) as pdf:
                text = ""
                for page in pdf.pages:
                    text += page.extract_text() or ""
            return text
        except Exception as e:
            return ""

    def extract_text_from_docx(self, docx_file):
        try:
            doc = docx.Document(docx_file)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return text
        except Exception as e:
            return ""

    def process_uploaded_files(self, uploaded_files):
        all_text = ""
        for uploaded_file in uploaded_files:
            if uploaded_file.type == "application/pdf":
                text = self.extract_text_from_pdf(uploaded_file)
                all_text += f"\n{text}"
            elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                text = self.extract_text_from_docx(uploaded_file)
                all_text += f"\n{text}"
            elif uploaded_file.type == "text/plain":
                text = str(uploaded_file.read(), "utf-8")
                all_text += f"\n{text}"
        return all_text

    def chat_with_medical_ai(self, message, patient_context="", processed_data=""):
        if not self.api_key:
            return self._get_fallback_response(message, patient_context, processed_data)
        
        try:
            system_prompt = f"""You are Dr. MedAI, a medical AI assistant. Provide helpful medical information.

Patient: {patient_context}
Data: {processed_data}

Provide clear, helpful medical information."""
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message}
            ]

            response = self.call_groq_api(messages, max_tokens=600, temperature=0.7)
            
            if response:
                return response
            else:
                return self._get_fallback_response(message, patient_context, processed_data)

        except Exception as e:
            return self._get_fallback_response(message, patient_context, processed_data)

    def _get_fallback_response(self, message, patient_context, processed_data):
        message_lower = message.lower()
        
        if any(word in message_lower for word in ['hi', 'hello', 'hey']):
            return "👋 Hello! I'm Dr. MedAI. I can help analyze medical reports, symptoms, and provide health guidance. Please share your medical information for personalized analysis."
        
        elif any(word in message_lower for word in ['iron', 'deficiency', 'anemia']):
            return self._get_iron_deficiency_analysis(processed_data)
        
        elif any(word in message_lower for word in ['symptom', 'tired', 'fatigue']):
            return "🤒 Common causes of fatigue include iron deficiency, sleep issues, thyroid problems, or nutritional deficiencies. I recommend consulting a doctor and getting blood tests like CBC and iron studies."
        
        elif any(word in message_lower for word in ['lab', 'test', 'result']):
            return self._analyze_lab_results(processed_data)
        
        else:
            return "I'm here to help with medical analysis. Please share your test results, symptoms, or health concerns for detailed guidance."

    def _get_iron_deficiency_analysis(self, processed_data):
        lab_values = self.extract_lab_values(processed_data)
        
        analysis = "🔬 **Iron Deficiency Analysis**\n\n"
        
        if 'Hemoglobin' in lab_values:
            hb = lab_values['Hemoglobin']
            if hb < 12.0:
                analysis += f"Hemoglobin: {hb} g/dL (LOW - indicates anemia)\n"
            else:
                analysis += f"Hemoglobin: {hb} g/dL (Normal)\n"
        
        if 'Ferritin' in lab_values:
            ferritin = lab_values['Ferritin']
            if ferritin < 15:
                analysis += f"Ferritin: {ferritin} ng/mL (VERY LOW - confirms iron deficiency)\n"
            else:
                analysis += f"Ferritin: {ferritin} ng/mL (Normal)\n"
        
        analysis += """
**Recommendations:**
1. Consult doctor for proper diagnosis
2. Consider iron supplements if recommended
3. Eat iron-rich foods (red meat, spinach, lentils)
4. Take vitamin C with iron for better absorption
5. Follow-up testing in 2-3 months"""
        
        return analysis

    def _analyze_lab_results(self, processed_data):
        if not processed_data:
            return "Please provide lab results for analysis."
        
        lab_values = self.extract_lab_values(processed_data)
        
        if not lab_values:
            return "Please format results as: Hemoglobin: 12.5, Ferritin: 25, WBC: 6.8"
        
        analysis = "🔬 **Lab Results Analysis**\n\n"
        
        for test, value in lab_values.items():
            if test in self.medical_knowledge_base['lab_ranges']:
                ranges = self.medical_knowledge_base['lab_ranges'][test]
                if value < ranges['low']:
                    analysis += f"⚠️ {test}: {value} {ranges['unit']} (LOW)\n"
                elif value > ranges['high']:
                    analysis += f"⚠️ {test}: {value} {ranges['unit']} (HIGH)\n"
                else:
                    analysis += f"✅ {test}: {value} {ranges['unit']} (Normal)\n"
        
        analysis += "\nDiscuss these results with your healthcare provider."
        return analysis

    def analyze_patient_data(self, patient_info, medical_history, lab_data, image_description, processed_files_text=""):
        if not self.api_key:
            return self._basic_analysis(patient_info, medical_history, lab_data, processed_files_text)
        
        try:
            combined_data = f"Patient: {patient_info}\nHistory: {medical_history}\nLabs: {lab_data}\nFiles: {processed_files_text}\nNotes: {image_description}"
            
            prompt = f"Analyze this medical case: {combined_data}"
            
            messages = [
                {"role": "system", "content": "You are a medical doctor analyzing patient data."},
                {"role": "user", "content": prompt}
            ]

            response = self.call_groq_api(messages, max_tokens=800, temperature=0.3)
            
            if response:
                return response
            else:
                return self._basic_analysis(patient_info, medical_history, lab_data, processed_files_text)
                
        except Exception as e:
            return self._basic_analysis(patient_info, medical_history, lab_data, processed_files_text)

    def _basic_analysis(self, patient_info, medical_history, lab_data, processed_files_text):
        analysis = "## Medical Analysis\n\n"
        analysis += f"**Patient:** {patient_info}\n\n"
        
        if medical_history:
            analysis += f"**History:** {medical_history}\n\n"
        
        combined_data = lab_data + processed_files_text
        lab_values = self.extract_lab_values(combined_data)
        
        if lab_values:
            analysis += "**Lab Findings:**\n"
            for test, value in lab_values.items():
                if test in self.medical_knowledge_base['lab_ranges']:
                    ranges = self.medical_knowledge_base['lab_ranges'][test]
                    if value < ranges['low']:
                        analysis += f"- {test}: {value} (LOW)\n"
                    elif value > ranges['high']:
                        analysis += f"- {test}: {value} (HIGH)\n"
                    else:
                        analysis += f"- {test}: {value} (Normal)\n"
        
        analysis += "\n**Recommendations:**\n1. Consult healthcare provider\n2. Discuss abnormal results\n3. Follow up as needed"
        return analysis

    def extract_lab_values(self, text):
        patterns = {
            'WBC': r'WBC[\s:\-]*([\d.]+)',
            'RBC': r'RBC[\s:\-]*([\d.]+)',
            'Hemoglobin': r'Hemoglobin[\s:\-]*([\d.]+)|Hgb[\s:\-]*([\d.]+)|Hb[\s:\-]*([\d.]+)',
            'Hematocrit': r'Hematocrit[\s:\-]*([\d.]+)|Hct[\s:\-]*([\d.]+)',
            'Platelets': r'Platelets[\s:\-]*([\d.]+)',
            'Glucose': r'Glucose[\s:\-]*([\d.]+)',
            'Creatinine': r'Creatinine[\s:\-]*([\d.]+)',
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
    st.sidebar.header("🔑 API Configuration")
    
    if 'GROQ_API_KEY' in st.secrets:
        api_key = st.secrets['GROQ_API_KEY']
        
        if api_key and api_key.startswith('gsk_'):
            try:
                url = "https://api.groq.com/openai/v1/models"
                headers = {"Authorization": f"Bearer {api_key}"}
                response = requests.get(url, headers=headers, timeout=10)
                
                if response.status_code == 200:
                    st.session_state.api_configured = True
                    st.session_state.api_key = api_key
                    st.sidebar.markdown("<div class='api-status-connected'>✅ API Connected</div>", unsafe_allow_html=True)
                    return api_key
            except:
                pass
    
    st.sidebar.markdown("<div class='api-status-disconnected'>❌ API Not Configured</div>", unsafe_allow_html=True)
    st.sidebar.info("Add GROQ_API_KEY to Streamlit secrets for AI features")
    return None

def setup_sidebar():
    with st.sidebar:
        st.markdown("<div class='medical-card'>", unsafe_allow_html=True)
        st.header("🏥 MediAI Pro")
        st.markdown("**Medical AI Assistant**")
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.header("👤 Patient Info")
        patient_name = st.text_input("Name", value="John Doe")
        patient_age = st.number_input("Age", value=45, min_value=0, max_value=120)
        patient_gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        
        st.header("📋 Medical History")
        medical_history = st.text_area("History & Symptoms", placeholder="e.g., Fatigue, pale skin...")
        
        st.header("📁 Upload Reports")
        lab_files = st.file_uploader("Lab Reports (PDF/DOCX/TXT)", type=['pdf', 'docx', 'txt'], accept_multiple_files=True)
        
        lab_text = st.text_area("Or Paste Lab Results", placeholder="Hemoglobin: 11.2\nFerritin: 12\nWBC: 6.8")
        
        clinical_notes = st.text_area("Clinical Notes", placeholder="Additional medical notes...")
        
        processed_files_text = ""
        if lab_files:
            analyzer = MedicalAIAnalyzer(st.session_state.api_key)
            processed_files_text = analyzer.process_uploaded_files(lab_files)
            st.session_state.processed_lab_data = processed_files_text
        
        patient_context = f"{patient_name}, {patient_age} years, {patient_gender}. History: {medical_history}. Labs: {lab_text}. Notes: {clinical_notes}"
        st.session_state.patient_context = patient_context
        
        st.markdown("<div class='medical-card'>", unsafe_allow_html=True)
        st.header("🔍 Analysis")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🚀 Analyze Data", use_container_width=True):
                st.session_state.analyze_clicked = True
        with col2:
            if st.button("🔄 Clear", use_container_width=True):
                st.session_state.analysis_results = {}
                st.session_state.messages = []
                st.session_state.analyze_clicked = False
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
        
        return {
            'patient_info': f"{patient_name}, {patient_age}, {patient_gender}",
            'medical_history': medical_history,
            'lab_data': lab_text,
            'image_description': clinical_notes,
            'processed_files_text': processed_files_text
        }

def display_medical_chat(analyzer):
    st.header("💬 Medical Chat")
    
    if st.session_state.api_configured:
        st.success("AI Assistant: Active")
    else:
        st.warning("AI Assistant: Basic Mode")
    
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"<div class='user-message'><strong>You:</strong> {message['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='assistant-message'><strong>Dr. MedAI:</strong> {message['content']}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    user_input = st.text_input("Ask a medical question:", placeholder="e.g., Explain my lab results...")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        if st.button("Send", use_container_width=True) and user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})
            response = analyzer.chat_with_medical_ai(user_input, st.session_state.patient_context, st.session_state.processed_lab_data)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()
    with col2:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

def display_analysis_dashboard(analyzer, patient_data):
    st.header("📊 Analysis Dashboard")
    
    if not st.session_state.analysis_results:
        st.info("Enter patient data and click 'Analyze Data' to get started.")
        return
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Status", "Analyzed")
    with col2:
        st.metric("AI Mode", "Active" if st.session_state.api_configured else "Basic")
    with col3:
        lab_values = analyzer.extract_lab_values(patient_data['lab_data'] + patient_data['processed_files_text'])
        abnormal = sum(1 for test, value in lab_values.items() 
                      if test in analyzer.medical_knowledge_base['lab_ranges'] 
                      and (value < analyzer.medical_knowledge_base['lab_ranges'][test]['low'] 
                           or value > analyzer.medical_knowledge_base['lab_ranges'][test]['high']))
        st.metric("Abnormal Values", abnormal)
    
    if 'comprehensive_analysis' in st.session_state.analysis_results:
        st.markdown(st.session_state.analysis_results['comprehensive_analysis'])

def create_visualizations(patient_data, analyzer):
    st.header("📈 Health Analytics")
    
    combined_data = patient_data['lab_data'] + patient_data['processed_files_text']
    lab_values = analyzer.extract_lab_values(combined_data)
    
    if lab_values:
        viz_data = []
        for test, value in lab_values.items():
            if test in analyzer.medical_knowledge_base['lab_ranges']:
                ranges = analyzer.medical_knowledge_base['lab_ranges'][test]
                status = "Normal"
                if value < ranges['low']:
                    status = "Low"
                elif value > ranges['high']:
                    status = "High"
                viz_data.append({'Test': test, 'Value': value, 'Status': status})
        
        if viz_data:
            df = pd.DataFrame(viz_data)
            fig = px.bar(df, x='Test', y='Value', color='Status', title='Lab Results Overview')
            st.plotly_chart(fig, use_container_width=True)

def main():
    st.markdown("<h1 class='main-header'>🏥 MediAI Pro</h1>", unsafe_allow_html=True)
    st.markdown("<h2 class='sub-header'>Medical AI Assistant</h2>", unsafe_allow_html=True)
    
    api_key = setup_groq_api()
    analyzer = MedicalAIAnalyzer(api_key)
    patient_data = setup_sidebar()
    
    if st.session_state.analyze_clicked:
        with st.spinner("Analyzing medical data..."):
            analysis = analyzer.analyze_patient_data(
                patient_data['patient_info'],
                patient_data['medical_history'],
                patient_data['lab_data'],
                patient_data['image_description'],
                patient_data['processed_files_text']
            )
            st.session_state.analysis_results = {'comprehensive_analysis': analysis}
            st.session_state.analyze_clicked = False
        st.success("Analysis complete!")
        st.rerun()
    
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "📊 Analysis", "📈 Analytics"])
    
    with tab1:
        display_medical_chat(analyzer)
    
    with tab2:
        display_analysis_dashboard(analyzer, patient_data)
    
    with tab3:
        create_visualizations(patient_data, analyzer)

if __name__ == "__main__":
    main()
