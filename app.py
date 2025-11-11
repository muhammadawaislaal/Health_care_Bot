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
    .rate-limit-warning { background-color: #fff3cd; color: #856404; padding: 0.5rem; border-radius: 5px; border: 1px solid #ffeaa7; margin: 0.5rem 0; }
</style>
""", unsafe_allow_html=True)

if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'patient_data' not in st.session_state:
    st.session_state.patient_data = {}
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
if 'rate_limit_warning' not in st.session_state:
    st.session_state.rate_limit_warning = False
if 'last_api_call' not in st.session_state:
    st.session_state.last_api_call = 0

class MedicalAIAnalyzer:
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.medical_knowledge_base = self._initialize_medical_knowledge()
        self.available_models = [
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768",
            "gemma2-9b-it",
            "llama3-70b-8192"
        ]
        self.current_model = "llama-3.1-8b-instant"
        self.rate_limit_delay = 3

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
            },
            'symptoms_analysis': {
                'fatigue': ['Iron deficiency', 'Anemia', 'Thyroid issues', 'Sleep disorders'],
                'weakness': ['Electrolyte imbalance', 'Anemia', 'Chronic fatigue'],
                'pale_skin': ['Anemia', 'Iron deficiency', 'Circulation issues'],
                'shortness_of_breath': ['Anemia', 'Cardiac issues', 'Respiratory conditions'],
                'dizziness': ['Anemia', 'Dehydration', 'Blood pressure issues']
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
            elif response.status_code == 429:
                st.session_state.rate_limit_warning = True
                return None
            elif response.status_code == 400:
                return self._try_fallback_model(messages, max_tokens, temperature)
            else:
                return None
        except Exception as e:
            logger.error(f"API call failed: {str(e)}")
            return None

    def _try_fallback_model(self, messages, max_tokens, temperature):
        current_index = self.available_models.index(self.current_model) if self.current_model in self.available_models else 0
        fallback_models = self.available_models[current_index + 1:]
        
        for model in fallback_models:
            try:
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
                    self.current_model = model
                    return result["choices"][0]["message"]["content"]
            except Exception:
                continue
        return None

    def extract_text_from_pdf(self, pdf_file):
        try:
            with pdfplumber.open(pdf_file) as pdf:
                text = ""
                for page in pdf.pages:
                    text += page.extract_text() or ""
            return text
        except Exception as e:
            return f"Error reading PDF: {str(e)}"

    def extract_text_from_docx(self, docx_file):
        try:
            doc = docx.Document(docx_file)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return text
        except Exception as e:
            return f"Error reading DOCX: {str(e)}"

    def process_uploaded_files(self, uploaded_files):
        all_text = ""
        for uploaded_file in uploaded_files:
            if uploaded_file.type == "application/pdf":
                text = self.extract_text_from_pdf(uploaded_file)
                all_text += f"\n--- PDF Content ---\n{text}\n"
            elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                text = self.extract_text_from_docx(uploaded_file)
                all_text += f"\n--- DOCX Content ---\n{text}\n"
            elif uploaded_file.type == "text/plain":
                text = str(uploaded_file.read(), "utf-8")
                all_text += f"\n--- TXT Content ---\n{text}\n"
        return all_text

    def chat_with_medical_ai(self, message, patient_context="", processed_data=""):
        if not self.api_key or st.session_state.rate_limit_warning:
            return self._intelligent_fallback_response(message, patient_context, processed_data)
        
        try:
            system_prompt = f"""You are Dr. MedAI, an advanced medical AI assistant. Analyze patient data and provide professional medical guidance.

PATIENT CONTEXT:
{patient_context if patient_context else 'No patient information'}

MEDICAL DATA:
{processed_data if processed_data else 'No medical data'}

INSTRUCTIONS:
- Provide specific, evidence-based medical analysis
- Analyze lab results and symptoms thoroughly
- Suggest appropriate medical next steps
- Be empathetic and professional
- Always recommend doctor consultation
- Keep responses clear and actionable"""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message}
            ]

            response = self.call_groq_api(messages, max_tokens=800, temperature=0.7)
            
            if response:
                return response
            else:
                return self._intelligent_fallback_response(message, patient_context, processed_data)

        except Exception as e:
            return self._intelligent_fallback_response(message, patient_context, processed_data)

    def _intelligent_fallback_response(self, message, patient_context, processed_data):
        message_lower = message.lower()
        
        has_iron_deficiency = "iron" in patient_context.lower() or "iron" in processed_data.lower()
        has_lab_data = any(keyword in processed_data.lower() for keyword in ['hemoglobin', 'wbc', 'rbc', 'ferritin'])
        
        if any(greet in message_lower for greet in ['hi', 'hello', 'hey']):
            if has_iron_deficiency:
                return """👋 Hello! I see you have iron deficiency concerns.

I can help you understand:
• Your lab results and their implications
• Dietary recommendations for iron
• Symptom management strategies
• When to seek medical attention

What specific aspect would you like to discuss?"""
            else:
                return """👋 Hello! I'm Dr. MedAI, your medical assistant.

I specialize in:
• Medical test analysis and interpretation
• Symptom assessment and guidance
• Treatment information and education
• Health and wellness recommendations

Please share your medical information for personalized analysis."""

        elif any(word in message_lower for word in ['iron', 'deficiency', 'anemia', 'ferritin', 'hemoglobin']):
            return self._provide_iron_deficiency_analysis(patient_context, processed_data)
        
        elif any(word in message_lower for word in ['symptom', 'tired', 'fatigue', 'weak', 'dizzy']):
            return self._analyze_symptoms()
        
        elif any(word in message_lower for word in ['lab', 'test', 'result', 'report']):
            return self._analyze_lab_results(processed_data)
        
        elif any(word in message_lower for word in ['diet', 'food', 'nutrition']):
            return self._provide_nutrition_advice(patient_context)
        
        else:
            return """I'm here to provide comprehensive medical analysis and guidance.

To help you effectively, please:
1. Share your medical test results or lab values
2. Describe your symptoms in detail
3. Upload any medical reports you have
4. Provide your medical history if relevant

I'll provide professional analysis and recommendations based on your information."""

    def _provide_iron_deficiency_analysis(self, patient_context, processed_data):
        analysis = """🔬 **Comprehensive Iron Deficiency Analysis**

**Clinical Overview:**
Iron deficiency is a common nutritional disorder affecting oxygen transport and energy production. It progresses through stages from iron depletion to iron deficiency anemia.

**Key Laboratory Markers:**"""
        
        lab_values = self.extract_lab_values(processed_data)
        
        critical_findings = []
        if 'Hemoglobin' in lab_values:
            hb = lab_values['Hemoglobin']
            if hb < 12.0:
                critical_findings.append(f"Hemoglobin: {hb} g/dL (LOW - indicates anemia)")
        
        if 'Ferritin' in lab_values:
            ferritin = lab_values['Ferritin']
            if ferritin < 15:
                critical_findings.append(f"Ferritin: {ferritin} ng/mL (VERY LOW - confirms iron deficiency)")
        
        if critical_findings:
            analysis += "\n🚨 **Critical Findings:**\n" + "\n".join([f"• {finding}" for finding in critical_findings]) + "\n"
        
        analysis += """
**Clinical Management Strategy:**

1. **Medical Evaluation:** Comprehensive assessment by healthcare provider
2. **Diagnostic Workup:** Complete iron studies (serum iron, TIBC, transferrin saturation)
3. **Therapeutic Intervention:** 
   - Oral iron supplementation (65-200 mg elemental iron daily)
   - Intravenous iron for severe cases or malabsorption
4. **Dietary Optimization:** Iron-rich foods with absorption enhancers
5. **Monitoring Protocol:** Repeat testing at 4-8 weeks and 3 months

**Referral Indications:**
• Severe anemia (Hb < 8 g/dL)
• Poor response to oral iron therapy
• Suspected malabsorption or blood loss
• Comorbid conditions

**Patient Education:**
• Take iron supplements with vitamin C
• Avoid calcium, tea, coffee with iron
• Expect improvement in 2-4 weeks
• Complete full treatment course (3-6 months)"""

        return analysis

    def _analyze_symptoms(self):
        return """🤒 **Comprehensive Symptom Analysis**

**Differential Diagnosis Framework:**

**Fatigue + Weakness:**
• Iron deficiency anemia (most common)
• Thyroid dysfunction (TSH, T4 required)
• Chronic fatigue syndrome
• Sleep disorders (polysomnography indicated)
• Vitamin deficiencies (B12, folate, Vitamin D)

**Cardiopulmonary Symptoms:**
• Anemia-related dyspnea
• Cardiac dysfunction (echocardiogram)
• Respiratory conditions (pulmonary function tests)
• Deconditioning

**Neurological Symptoms:**
• Orthostatic hypotension
• Vestibular disorders
• Neuropathies (EMG/NCV studies)

**Recommended Diagnostic Pathway:**
1. Complete Blood Count with indices
2. Comprehensive metabolic panel
3. Thyroid function studies
4. Iron studies and B12/folate
5. Inflammatory markers (CRP, ESR)

**Urgent Evaluation Required for:**
• Syncope or near-syncope
• Chest pain or palpitations
• Severe dyspnea at rest
• Neurological deficits"""

    def _analyze_lab_results(self, processed_data):
        if not processed_data:
            return "Please provide lab results for analysis. You can upload reports or paste values directly."
        
        lab_values = self.extract_lab_values(processed_data)
        
        if not lab_values:
            return "Please format lab results as: Hemoglobin: 12.5 g/dL, Ferritin: 25 ng/mL, WBC: 6.8"
        
        analysis = "🔬 **Laboratory Results Analysis**\n\n"
        
        abnormal_count = 0
        for test, value in lab_values.items():
            if test in self.medical_knowledge_base['lab_ranges']:
                ranges = self.medical_knowledge_base['lab_ranges'][test]
                if value < ranges['low']:
                    analysis += f"⚠️ **{test}**: {value} {ranges['unit']} **(LOW)**\n"
                    abnormal_count += 1
                elif value > ranges['high']:
                    analysis += f"⚠️ **{test}**: {value} {ranges['unit']} **(HIGH)**\n"
                    abnormal_count += 1
                else:
                    analysis += f"✅ **{test}**: {value} {ranges['unit']} (Normal)\n"
        
        if abnormal_count > 0:
            analysis += f"\n**Clinical Significance:** {abnormal_count} abnormal values identified\n"
        else:
            analysis += "\n**All values within reference ranges**\n"
        
        analysis += "\n**Next Steps:**\n• Consult with healthcare provider for interpretation\n• Consider repeat testing if indicated\n• Implement lifestyle modifications as appropriate\n• Schedule follow-up as recommended"

        return analysis

    def _provide_nutrition_advice(self, patient_context):
        if "iron" in patient_context.lower():
            return """🍎 **Medical Nutrition Therapy for Iron Deficiency**

**Evidence-Based Dietary Strategy:**

**Heme Iron Sources (High bioavailability):**
• Red meat: Beef, lamb (3oz = 2-3mg iron)
• Organ meats: Liver (exceptionally high in iron)
• Poultry: Dark meat preferred
• Fish and shellfish: Oysters, clams, sardines

**Non-Heme Iron Optimization:**
• Legumes: Lentils, chickpeas, kidney beans
• Leafy greens: Spinach, kale (cooked)
• Fortified foods: Cereals, grains
• Nuts and seeds: Pumpkin seeds, cashews

**Absorption Enhancement:**
• **Vitamin C**: 75-100mg with meals (citrus, bell peppers, broccoli)
• **Meat factor**: Combine plant iron with small amounts of animal protein
• **Avoid inhibitors**: Phytates (whole grains), polyphenols (tea, coffee), calcium

**Sample Medical Nutrition Plan:**
**Breakfast:** Fortified cereal with strawberries + orange juice
**Lunch:** Spinach salad with lean beef + bell peppers
**Dinner:** Lentil soup with tomatoes + small serving of chicken
**Supplements:** Iron + 250mg Vitamin C between meals

**Monitoring Parameters:**
• Dietary iron intake assessment
• Symptom improvement tracking
• Laboratory monitoring at 3 months"""

        return """🍎 **General Medical Nutrition Guidelines**

**Evidence-Based Dietary Principles:**

**Macronutrient Distribution:**
• Complex carbohydrates: 45-65% of calories
• Lean protein: 15-25% of calories
• Healthy fats: 20-35% of calories

**Micronutrient Optimization:**
• **Iron**: Hemoglobin synthesis, oxygen transport
• **B12**: Neurological function, DNA synthesis
• **Vitamin D**: Bone health, immune function
• **Magnesium**: Enzyme cofactor, muscle function
• **Zinc**: Immune function, wound healing

**Therapeutic Eating Patterns:**
• Mediterranean diet: Cardiovascular protection
• DASH diet: Hypertension management
• Anti-inflammatory diet: Chronic disease prevention

**Medical Monitoring:**
• Regular anthropometric measurements
• Biochemical nutritional assessment
• Clinical symptom evaluation"""

    def analyze_patient_data(self, patient_info, medical_history, lab_data, image_description, processed_files_text=""):
        if not self.api_key or st.session_state.rate_limit_warning:
            return self._basic_patient_analysis(patient_info, medical_history, lab_data, image_description, processed_files_text)
        
        try:
            combined_data = f"""
PATIENT INFORMATION:
{patient_info}

MEDICAL HISTORY:
{medical_history}

LABORATORY DATA:
{lab_data}

ADDITIONAL DATA:
{processed_files_text}

CLINICAL NOTES:
{image_description}
"""

            prompt = f"""
            Perform comprehensive medical analysis:

            {combined_data}

            Provide structured analysis with:
            **CLINICAL ASSESSMENT**: Summary and impression
            **DIAGNOSTIC FINDINGS**: Key abnormalities and significance
            **RISK STRATIFICATION**: Health risks and urgency
            **MANAGEMENT PLAN**: Specific recommendations including:
               - Further diagnostic testing
               - Specialist consultations
               - Therapeutic interventions
               - Monitoring parameters
            **PATIENT EDUCATION**: Key information and warnings

            Focus on evidence-based, actionable medical guidance.
            """

            messages = [
                {
                    "role": "system", 
                    "content": "You are an experienced physician providing comprehensive medical analysis. Be specific, evidence-based, and focus on patient safety and clinical outcomes."
                },
                {"role": "user", "content": prompt}
            ]

            response = self.call_groq_api(messages, max_tokens=1200, temperature=0.3)
            
            if response:
                return response
            else:
                return self._basic_patient_analysis(patient_info, medical_history, lab_data, image_description, processed_files_text)
                
        except Exception as e:
            return self._basic_patient_analysis(patient_info, medical_history, lab_data, image_description, processed_files_text)

    def _basic_patient_analysis(self, patient_info, medical_history, lab_data, image_description, processed_files_text):
        analysis = "## 🏥 Comprehensive Medical Analysis\n\n"
        
        analysis += f"**Patient:** {patient_info}\n\n"
        
        if medical_history:
            analysis += f"**Medical History:** {medical_history}\n\n"
        
        combined_lab_data = lab_data + "\n" + processed_files_text
        lab_values = self.extract_lab_values(combined_lab_data)
        
        if lab_values:
            analysis += "### 🔬 Laboratory Assessment\n"
            for test, value in lab_values.items():
                if test in self.medical_knowledge_base['lab_ranges']:
                    ranges = self.medical_knowledge_base['lab_ranges'][test]
                    if value < ranges['low']:
                        analysis += f"⚠️ **{test}**: {value} {ranges['unit']} (LOW)\n"
                    elif value > ranges['high']:
                        analysis += f"⚠️ **{test}**: {value} {ranges['unit']} (HIGH)\n"
                    else:
                        analysis += f"✅ **{test}**: {value} {ranges['unit']} (Normal)\n"
        
        analysis += "\n### 💡 Clinical Recommendations\n"
        analysis += "1. **Comprehensive medical evaluation** with healthcare provider\n"
        analysis += "2. **Diagnostic testing** as clinically indicated\n"
        analysis += "3. **Specialist referral** based on findings\n"
        analysis += "4. **Therapeutic interventions** per medical guidance\n"
        analysis += "5. **Regular monitoring** and follow-up care\n"
        
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
        
        if not api_key or not api_key.startswith('gsk_'):
            st.sidebar.markdown("<div class='api-status-disconnected'>❌ Invalid API Key</div>", unsafe_allow_html=True)
            return None
        
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
                return api_key
            else:
                st.sidebar.markdown("<div class='api-status-disconnected'>❌ API Failed</div>", unsafe_allow_html=True)
                return None
                
        except Exception:
            st.sidebar.markdown("<div class='api-status-disconnected'>❌ Connection Error</div>", unsafe_allow_html=True)
    
    else:
        st.sidebar.markdown("<div class='api-status-disconnected'>❌ API Key Required</div>", unsafe_allow_html=True)
    
    return None

def setup_sidebar():
    with st.sidebar:
        st.markdown("<div class='medical-card'>", unsafe_allow_html=True)
        st.header("🏥 MediAI Pro")
        st.markdown("**Advanced Medical AI Assistant**")
        if st.session_state.api_configured:
            st.success("🤖 AI: **ACTIVE**")
        else:
            st.warning("🤖 AI: **BASIC MODE**")
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.header("👤 Patient Information")
        
        col1, col2 = st.columns(2)
        with col1:
            patient_id = st.text_input("Patient ID", value="PT-001")
            patient_age = st.number_input("Age", min_value=0, max_value=120, value=45)
        with col2:
            patient_name = st.text_input("Patient Name", value="John Doe")
            patient_gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        
        st.header("📋 Medical History")
        medical_history = st.text_area(
            "Medical History & Symptoms", 
            height=100,
            placeholder="Example: Iron deficiency, fatigue, pale skin, shortness of breath..."
        )
        
        st.header("📁 Medical Data Upload")
        
        tab1, tab2 = st.tabs(["Lab Reports", "Clinical Notes"])
        
        with tab1:
            lab_files = st.file_uploader("Upload Lab Reports", 
                                       type=['pdf', 'docx', 'txt'], 
                                       accept_multiple_files=True)
            
            if lab_files:
                st.info(f"📎 {len(lab_files)} file(s) uploaded")
            
            lab_text_input = st.text_area("Paste Lab Results:", height=100,
                                        placeholder="Hemoglobin: 11.2 g/dL\nFerritin: 12 ng/mL\nWBC: 6.8")
        
        with tab2:
            image_description = st.text_area("Clinical Notes:", height=100,
                                           placeholder="Patient reports fatigue, weakness, pale skin...")
        
        processed_files_text = ""
        if lab_files:
            analyzer = MedicalAIAnalyzer(st.session_state.api_key)
            processed_files_text = analyzer.process_uploaded_files(lab_files)
            st.session_state.processed_lab_data = processed_files_text
        
        patient_context = f"""
Patient: {patient_name}, {patient_age} years, {patient_gender}
History: {medical_history if medical_history else 'Not specified'}
Labs: {lab_text_input if lab_text_input else 'Not provided'}
Notes: {image_description if image_description else 'Not provided'}
        """.strip()
        
        st.session_state.patient_context = patient_context
        
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
    st.header("💬 Dr. MedAI - Medical Conversation")
    
    if st.session_state.api_configured:
        st.success("✅ AI Assistant: **ACTIVE**")
    else:
        st.warning("🟡 AI Assistant: **BASIC MODE**")
    
    if st.session_state.patient_context:
        with st.expander("📋 Patient Context", expanded=False):
            st.text(st.session_state.patient_context)
    
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"<div class='user-message'><strong>You:</strong> {message['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='assistant-message'><strong>Dr. MedAI:</strong> {message['content']}</div>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    user_input = st.text_input(
        "Ask medical questions or discuss symptoms:",
        placeholder="Example: Explain my iron deficiency results and recommended treatment..."
    )
    
    col1, col2 = st.columns([3, 1])
    with col1:
        if st.button("Send Message", use_container_width=True) and user_input:
            process_user_message(analyzer, user_input)
    with col2:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    
    st.subheader("💡 Quick Actions")
    quick_col1, quick_col2 = st.columns(2)
    
    with quick_col1:
        if st.button("🤒 Symptoms Analysis", use_container_width=True):
            process_user_message(analyzer, "Analyze my symptoms and suggest possible causes")
        if st.button("🔬 Lab Results", use_container_width=True):
            process_user_message(analyzer, "Interpret my laboratory test results")
    
    with quick_col2:
        if st.button("🍎 Nutrition Advice", use_container_width=True):
            process_user_message(analyzer, "Provide dietary recommendations for my condition")
        if st.button("📋 Treatment Plan", use_container_width=True):
            process_user_message(analyzer, "Suggest a comprehensive treatment plan")

def process_user_message(analyzer, user_input):
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    with st.spinner("Dr. MedAI is analyzing..."):
        ai_response = analyzer.chat_with_medical_ai(
            user_input, 
            st.session_state.patient_context,
            st.session_state.processed_lab_data
        )
        
        st.session_state.messages.append({"role": "assistant", "content": ai_response})
    
    st.rerun()

def display_analysis_dashboard(analyzer, patient_data):
    st.header("📊 Medical Analysis Dashboard")
    
    if not st.session_state.analysis_results:
        st.info("""
        🏥 **MediAI Pro Medical Assistant**
        
        **To Begin Analysis:**
        1. Enter patient information
        2. Provide medical history or symptoms
        3. Upload lab reports or paste results
        4. Click **'Analyze Patient Data'**
        5. Use chat for detailed discussions
        
        *Comprehensive data = Better medical insights!*
        """)
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Analysis Status", "Completed")
    
    with col2:
        st.metric("AI Mode", "Active" if st.session_state.api_configured else "Basic")
    
    with col3:
        has_data = any([patient_data['medical_history'], patient_data['lab_data'], patient_data['processed_files_text']])
        st.metric("Data Quality", "Good" if has_data else "Basic")
    
    with col4:
        lab_values = analyzer.extract_lab_values(patient_data['lab_data'] + patient_data['processed_files_text'])
        abnormal_count = sum(1 for test, value in lab_values.items() 
                           if test in analyzer.medical_knowledge_base['lab_ranges'] 
                           and (value < analyzer.medical_knowledge_base['lab_ranges'][test]['low'] 
                                or value > analyzer.medical_knowledge_base['lab_ranges'][test]['high']))
        st.metric("Abnormal Values", abnormal_count if lab_values else "N/A")
    
    if 'comprehensive_analysis' in st.session_state.analysis_results:
        with st.expander("📋 **Medical Analysis Report**", expanded=True):
            st.markdown(st.session_state.analysis_results['comprehensive_analysis'])

def create_advanced_visualizations(patient_data, analyzer):
    st.header("📈 Medical Analytics")
    
    combined_data = patient_data['lab_data'] + patient_data['processed_files_text']
    lab_values = analyzer.extract_lab_values(combined_data)
    
    if lab_values:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🩸 Laboratory Values")
            
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
                    title='Laboratory Values Overview',
                    yaxis_title='Value',
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📊 Clinical Status")
            
            status_counts = df['Status'].value_counts()
            fig_pie = px.pie(
                values=status_counts.values,
                names=status_counts.index,
                title='Test Result Distribution',
                color=status_counts.index,
                color_discrete_map={'Normal': 'green', 'Low': 'blue', 'High': 'red'}
            )
            st.plotly_chart(fig_pie, use_container_width=True)
    
    else:
        st.info("Provide lab results for medical analytics visualization")

def main():
    st.markdown("<h1 class='main-header'>🏥 MediAI Pro</h1>", unsafe_allow_html=True)
    st.markdown("<h2 class='sub-header'>Advanced Medical AI Assistant</h2>", unsafe_allow_html=True)
    
    api_key = setup_groq_api()
    analyzer = MedicalAIAnalyzer(api_key)
    patient_data = setup_sidebar()
    
    if st.session_state.analyze_clicked:
        with st.spinner("🔄 Performing comprehensive medical analysis..."):
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
            
        st.success("✅ Medical analysis completed!")
        st.rerun()
    
    tab1, tab2, tab3, tab4 = st.tabs(["💬 Medical Chat", "📊 Analysis", "📈 Analytics", "📋 Report"])
    
    with tab1:
        display_medical_chat(analyzer)
    
    with tab2:
        display_analysis_dashboard(analyzer, patient_data)
    
    with tab3:
        create_advanced_visualizations(patient_data, analyzer)
    
    with tab4:
        st.header("📋 Medical Report")
        if st.session_state.analysis_results:
            report_content = f"""# 🏥 MEDICAL ANALYSIS REPORT

## Patient Information
{patient_data['patient_info']}

## Comprehensive Medical Analysis
{st.session_state.analysis_results.get('comprehensive_analysis', 'No analysis available')}

## Clinical Recommendations

### Immediate Actions
- Review findings with healthcare provider
- Address abnormal values medically
- Implement recommended interventions
- Schedule follow-up appointments

### Ongoing Management
- Regular monitoring of key parameters
- Lifestyle modifications as indicated
- Medication adherence if prescribed
- Preventive care strategies

### Long-term Health
- Chronic condition management
- Regular health screenings
- Healthy lifestyle maintenance
- Continuous medical follow-up

---
*Generated by MediAI Pro on {datetime.now().strftime('%B %d, %Y at %H:%M')}*

**Medical Disclaimer**: This analysis provides educational information and should be reviewed by qualified healthcare professionals."""
            st.markdown(report_content)
            
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    label="📥 Download Report",
                    data=report_content,
                    file_name=f"Medical_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                    mime="text/markdown",
                    use_container_width=True
                )
            with col2:
                st.download_button(
                    label="📊 Export Data",
                    data=json.dumps(st.session_state.analysis_results, indent=2),
                    file_name=f"Analysis_{datetime.now().strftime('%Y%m%d')}.json",
                    mime="application/json",
                    use_container_width=True
                )
        else:
            st.info("Click 'Analyze Patient Data' to generate medical report")

if __name__ == "__main__":
    main()
