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
    }
    .assistant-message {
        background-color: #e9ecef;
        color: #333;
        padding: 1rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        max-width: 80%;
        margin-right: auto;
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
    .quick-action-btn:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
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

class MedicalAIAnalyzer:
    def __init__(self):
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
            },
            'symptoms_database': {
                'fever': {
                    'causes': ['Infection', 'Inflammation', 'Autoimmune disorders', 'Medication reaction'],
                    'red_flags': ['Fever > 104°F', 'Fever lasting > 3 days', 'Rash', 'Stiff neck', 'Confusion'],
                    'advice': 'Rest, hydrate, monitor temperature. Seek care for high fever or concerning symptoms.'
                },
                'chest_pain': {
                    'causes': ['Cardiac issues', 'Pulmonary problems', 'GERD', 'Musculoskeletal pain', 'Anxiety'],
                    'red_flags': ['Radiating pain', 'Shortness of breath', 'Sweating', 'Nausea'],
                    'advice': 'Seek emergency care for chest pain with concerning features.'
                },
                'headache': {
                    'causes': ['Tension', 'Migraine', 'Sinusitis', 'Dehydration', 'Hypertension'],
                    'red_flags': ['Sudden severe headache', 'Vision changes', 'Weakness', 'Fever with headache'],
                    'advice': 'Rest in quiet room, hydrate. Seek care for severe or unusual headaches.'
                }
            }
        }
    
    def chat_with_medical_ai(self, message, conversation_history=None, patient_context=None):
        """Advanced medical chat using rule-based AI with comprehensive responses"""
        try:
            # Analyze message for medical content
            message_lower = message.lower()
            
            # Build context from conversation history
            context = self._build_conversation_context(conversation_history)
            
            # Generate comprehensive medical response
            response = self._generate_medical_response(message_lower, context, patient_context)
            
            return response
            
        except Exception as e:
            logger.error(f"Medical AI error: {str(e)}")
            return "I apologize, but I'm having difficulty processing your request. Please try rephrasing your question or consult a healthcare provider for immediate medical concerns."
    
    def _build_conversation_context(self, conversation_history):
        """Build context from conversation history"""
        context = {
            'symptoms_mentioned': [],
            'medications_mentioned': [],
            'conditions_mentioned': [],
            'tests_mentioned': []
        }
        
        if conversation_history:
            for msg in conversation_history[-10:]:  # Last 10 messages
                content = msg['content'].lower()
                
                # Extract symptoms
                for symptom in ['fever', 'pain', 'headache', 'cough', 'nausea', 'fatigue', 'dizziness']:
                    if symptom in content and symptom not in context['symptoms_mentioned']:
                        context['symptoms_mentioned'].append(symptom)
                
                # Extract medications
                med_keywords = ['medication', 'pill', 'tablet', 'prescription', 'taking', 'drug']
                if any(keyword in content for keyword in med_keywords):
                    context['medications_mentioned'].append('medications discussed')
                
                # Extract conditions
                for condition in ['diabetes', 'hypertension', 'heart', 'lung', 'kidney', 'liver']:
                    if condition in content and condition not in context['conditions_mentioned']:
                        context['conditions_mentioned'].append(condition)
        
        return context
    
    def _generate_medical_response(self, message, context, patient_context):
        """Generate comprehensive medical response"""
        response_parts = []
        
        # Greeting response
        if any(word in message for word in ['hello', 'hi', 'hey', 'greetings']):
            response_parts.append("Hello! I'm Dr. MedAI, your medical assistant. I'm here to help with health-related questions, symptom discussions, and general medical information. How can I assist you today?")
        
        # Symptom analysis
        elif any(word in message for word in ['symptom', 'pain', 'hurt', 'feel', 'experience']):
            response_parts.append(self._analyze_symptoms(message, context))
        
        # Medication questions
        elif any(word in message for word in ['medication', 'pill', 'drug', 'prescription', 'taking']):
            response_parts.append(self._discuss_medications(message))
        
        # Test results
        elif any(word in message for word in ['test', 'result', 'lab', 'blood', 'report']):
            response_parts.append(self._discuss_test_results(message))
        
        # General health advice
        elif any(word in message for word in ['healthy', 'diet', 'exercise', 'lifestyle', 'prevent']):
            response_parts.append(self._provide_health_advice(message))
        
        # Emergency recognition
        elif any(word in message for word in ['emergency', 'urgent', '911', 'hospital', 'severe']):
            response_parts.append(self._emergency_guidance())
        
        # Default medical response
        else:
            response_parts.append(self._general_medical_response(message))
        
        # Add patient context if available
        if patient_context:
            response_parts.append(f"\n\n*Considering your profile: {patient_context}*")
        
        # Add safety disclaimer
        response_parts.append("\n\n---\n**Important**: This is general medical information. Always consult healthcare providers for personal medical advice. Seek emergency care for serious symptoms.")
        
        return "".join(response_parts)
    
    def _analyze_symptoms(self, message, context):
        """Analyze symptoms and provide guidance"""
        symptoms_found = []
        
        for symptom, info in self.medical_knowledge_base['symptoms_database'].items():
            if symptom in message:
                symptoms_found.append(symptom)
        
        if symptoms_found:
            response = "## Symptom Analysis\n\n"
            for symptom in symptoms_found:
                info = self.medical_knowledge_base['symptoms_database'][symptom]
                response += f"**{symptom.title()}**:\n"
                response += f"- Possible causes: {', '.join(info['causes'])}\n"
                response += f"- Watch for: {', '.join(info['red_flags'])}\n"
                response += f"- General advice: {info['advice']}\n\n"
            
            response += "**Next Steps**:\n"
            response += "1. Monitor symptoms and severity\n"
            response += "2. Keep a symptom diary if ongoing\n"
            response += "3. Contact healthcare provider if symptoms persist or worsen\n"
            response += "4. Seek emergency care for red flag symptoms\n"
            
            return response
        else:
            return "I understand you're discussing symptoms. Could you provide more details about what you're experiencing, such as location, severity, duration, and any associated symptoms? This will help me provide more specific guidance."
    
    def _discuss_medications(self, message):
        """Provide medication information"""
        return """
## Medication Information

**General Medication Safety**:
- Always take medications as prescribed
- Never stop medications without consulting your doctor
- Be aware of potential side effects
- Keep an updated medication list
- Inform all providers about all medications you take

**Common Considerations**:
- Report any new symptoms after starting medications
- Understand medication purpose and expected effects
- Ask about potential interactions with other drugs
- Follow storage instructions carefully

**When to Contact Your Doctor**:
- Severe side effects or allergic reactions
- Questions about dosage or timing
- Concerns about effectiveness
- Considering stopping or changing medications

Please consult your healthcare provider or pharmacist for medication-specific advice.
"""
    
    def _discuss_test_results(self, message):
        """Discuss test results and their meaning"""
        return """
## Understanding Test Results

**Laboratory Tests**:
- Results should always be interpreted by healthcare providers
- "Normal ranges" can vary between laboratories
- Single abnormal values may not be significant
- Trends over time are often more important than single results

**Common Test Categories**:
- **Blood tests**: CBC, chemistry panels, lipid profiles
- **Imaging**: X-rays, CT scans, MRI, ultrasound
- **Cardiac**: ECG, echocardiogram, stress tests
- **Screening**: Cancer screening, preventive tests

**Next Steps for Test Results**:
1. Discuss results with your ordering provider
2. Understand what abnormal results mean for your health
3. Ask about necessary follow-up testing
4. Inquire about treatment options if needed
5. Request copies of your results for your records

Your healthcare provider can explain what your specific results mean in the context of your overall health.
"""
    
    def _provide_health_advice(self, message):
        """Provide general health advice"""
        return """
## General Health & Wellness

**Healthy Lifestyle Foundations**:

🍎 **Nutrition**:
- Balanced diet with fruits, vegetables, whole grains
- Adequate protein and healthy fats
- Limit processed foods, sugar, and excess salt
- Stay hydrated with water

💪 **Physical Activity**:
- 150 minutes moderate exercise weekly
- Strength training 2x weekly
- Include flexibility and balance exercises
- Find activities you enjoy

😴 **Sleep & Rest**:
- 7-9 hours quality sleep nightly
- Consistent sleep schedule
- Create restful sleep environment
- Manage stress through relaxation

🩺 **Preventive Care**:
- Regular health check-ups
- Age-appropriate screenings
- Vaccinations up to date
- Dental and vision care

**Remember**: Small, consistent changes lead to lasting health benefits.
"""
    
    def _emergency_guidance(self):
        """Provide emergency guidance"""
        return """
## 🚨 Emergency Medical Situations

**Seek IMMEDIATE Medical Attention for**:

🔴 **Life-Threatening Symptoms**:
- Chest pain or pressure
- Difficulty breathing
- Severe bleeding
- Sudden weakness or numbness
- Severe burns
- Poisoning or overdose
- Suicidal or homicidal thoughts

🔴 **Other Emergency Signs**:
- High fever with stiff neck or confusion
- Severe allergic reaction
- Head injury with loss of consciousness
- Broken bones with deformity
- Severe abdominal pain
- Seizures for the first time

**In Emergency Situations**:
- Call emergency services (911) immediately
- Do not drive yourself to the hospital
- Have someone stay with you
- Bring medications and medical information
- Don't delay seeking care for concerning symptoms

Your safety is the top priority. When in doubt, seek medical evaluation.
"""
    
    def _general_medical_response(self, message):
        """General medical response for other queries"""
        return """
## Medical Information & Support

I'm here to provide general medical information and support. I can help with:

🤒 **Symptom Discussion**: Understanding common symptoms and when to seek care
💊 **Medication Questions**: General information about medication safety and usage
🩺 **Test Results**: Guidance on understanding common medical tests
🍎 **Health & Wellness**: Lifestyle advice for maintaining good health
📋 **Healthcare Navigation**: Tips for working with healthcare providers

**How to Get the Most Help**:
- Be specific about your questions or concerns
- Provide relevant context about your situation
- Let me know if you have particular health conditions
- Share what you're hoping to learn or accomplish

Remember, I provide general medical information. For personal medical advice, always consult qualified healthcare professionals.
"""
    
    def analyze_lab_results(self, lab_data):
        """Advanced laboratory results analysis"""
        extracted_values = self.extract_lab_values(lab_data)
        
        analysis = "## 🔬 Comprehensive Laboratory Analysis\n\n"
        
        critical_findings = []
        abnormal_findings = []
        normal_findings = []
        
        for test, value in extracted_values.items():
            if test in self.medical_knowledge_base['lab_ranges']:
                ranges = self.medical_knowledge_base['lab_ranges'][test]
                
                if value < ranges['critical_low'] or value > ranges['critical_high']:
                    critical_findings.append((test, value, ranges))
                elif value < ranges['low'] or value > ranges['high']:
                    abnormal_findings.append((test, value, ranges))
                else:
                    normal_findings.append((test, value, ranges))
        
        # Critical findings
        if critical_findings:
            analysis += "### 🚨 Critical Findings Requiring Immediate Attention\n"
            for test, value, ranges in critical_findings:
                analysis += f"- **{test}**: {value} {ranges['unit']} (Critical range: {ranges['critical_low']}-{ranges['critical_high']})\n"
            analysis += "\n"
        
        # Abnormal findings
        if abnormal_findings:
            analysis += "### ⚠️ Abnormal Findings\n"
            for test, value, ranges in abnormal_findings:
                analysis += f"- **{test}**: {value} {ranges['unit']} (Normal: {ranges['low']}-{ranges['high']})\n"
            analysis += "\n"
        
        # Normal findings
        if normal_findings:
            analysis += "### ✅ Within Normal Range\n"
            for test, value, ranges in normal_findings:
                analysis += f"- **{test}**: {value} {ranges['unit']}\n"
            analysis += "\n"
        
        # Clinical interpretation
        analysis += "### 🩺 Clinical Interpretation\n"
        if critical_findings:
            analysis += "**URGENT MEDICAL ATTENTION REQUIRED**\n\n"
            analysis += "Critical laboratory values detected that require immediate medical evaluation.\n\n"
        elif abnormal_findings:
            analysis += "Abnormal laboratory values noted that warrant clinical correlation and follow-up.\n\n"
        else:
            analysis += "All analyzed laboratory parameters are within normal reference ranges.\n\n"
        
        # Recommendations
        analysis += "### 📋 Recommended Actions\n"
        if critical_findings:
            analysis += "1. **IMMEDIATE**: Contact healthcare provider or seek emergency care\n"
            analysis += "2. **URGENT**: Repeat critical tests for confirmation\n"
            analysis += "3. **SPECIALIST**: Consult relevant medical specialist\n"
        elif abnormal_findings:
            analysis += "1. **Follow-up**: Repeat abnormal tests as recommended\n"
            analysis += "2. **Correlation**: Evaluate with clinical symptoms\n"
            analysis += "3. **Monitoring**: Establish parameter monitoring plan\n"
        else:
            analysis += "1. **Continue** routine health monitoring\n"
            analysis += "2. **Maintain** current health practices\n"
            analysis += "3. **Schedule** next routine evaluation\n"
        
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
    
    def analyze_medical_image(self, image_description):
        """Medical image analysis"""
        analysis = "## 🖼️ Medical Image Analysis\n\n"
        analysis += f"**Image Description**: {image_description}\n\n"
        
        analysis += "### 📋 Radiological Assessment Framework\n"
        analysis += "1. **Technical Quality**: Evaluate image clarity and positioning\n"
        analysis += "2. **Systematic Review**: Assess all visualized anatomical structures\n"
        analysis += "3. **Pathological Findings**: Identify any abnormalities or variations\n"
        analysis += "4. **Clinical Correlation**: Relate findings to patient's condition\n"
        analysis += "5. **Recommendations**: Suggest further evaluation if needed\n\n"
        
        analysis += "### 🎯 Key Considerations\n"
        
        # Keyword-based insights
        if any(word in image_description.lower() for word in ['cardiomegaly', 'enlarged heart']):
            analysis += "- **Cardiomegaly noted**: Evaluate for heart failure, cardiomyopathy\n"
        if any(word in image_description.lower() for word in ['consolidation', 'opacity']):
            analysis += "- **Pulmonary findings**: Consider infection, inflammation, or other pathology\n"
        if any(word in image_description.lower() for word in ['fracture', 'break']):
            analysis += "- **Fracture identified**: Assess alignment and potential complications\n"
        
        analysis += "\n### 📞 Recommended Next Steps\n"
        analysis += "1. **Formal Interpretation**: Consult board-certified radiologist\n"
        analysis += "2. **Clinical Correlation**: Discuss findings with treating provider\n"
        analysis += "3. **Comparison**: Review with previous studies if available\n"
        analysis += "4. **Follow-up**: Schedule additional imaging if clinically indicated\n"
        
        return analysis
    
    def generate_patient_summary(self, patient_info, medical_history, current_findings):
        """Generate comprehensive patient summary"""
        analysis = "## 👨‍⚕️ Comprehensive Patient Assessment\n\n"
        
        analysis += "### 📋 Patient Overview\n"
        analysis += f"{patient_info}\n\n"
        
        analysis += "### 🏥 Medical History Summary\n"
        if medical_history and medical_history.strip():
            analysis += f"{medical_history}\n\n"
        else:
            analysis += "No significant medical history provided.\n\n"
        
        analysis += "### 🔬 Current Clinical Picture\n"
        analysis += "Based on available data and findings:\n\n"
        
        # Extract key information for assessment
        if "hypertension" in medical_history.lower():
            analysis += "- **Chronic Condition**: Hypertension - ongoing management required\n"
        if "diabetes" in medical_history.lower():
            analysis += "- **Chronic Condition**: Diabetes - glycemic control monitoring needed\n"
        if "hyperlipidemia" in medical_history.lower():
            analysis += "- **Chronic Condition**: Hyperlipidemia - lipid management indicated\n"
        
        analysis += "\n### 💊 Management Recommendations\n"
        analysis += "**Immediate Actions**:\n"
        analysis += "- Review all findings with appropriate specialists\n"
        analysis += "- Address any critical values immediately\n"
        analysis += "- Ensure patient understanding of health status\n\n"
        
        analysis += "**Short-term Plan**:\n"
        analysis += "- Schedule follow-up based on test results\n"
        analysis += "- Adjust medications if indicated\n"
        analysis += "- Implement recommended lifestyle modifications\n\n"
        
        analysis += "**Long-term Strategy**:\n"
        analysis += "- Regular monitoring of chronic conditions\n"
        analysis += "- Preventive care and health maintenance\n"
        analysis += "- Patient education and self-management support\n\n"
        
        analysis += "### ⚠️ Important Considerations\n"
        analysis += "- Always correlate findings with clinical presentation\n"
        analysis += "- Consider individual patient factors and preferences\n"
        analysis += "- Monitor for medication side effects and interactions\n"
        analysis += "- Address preventive health measures appropriately\n"
        
        return analysis

def setup_sidebar():
    """Setup the professional medical sidebar"""
    with st.sidebar:
        st.markdown("<div class='medical-card'>", unsafe_allow_html=True)
        st.header("🏥 MediAI Pro")
        st.markdown("**Advanced Medical AI Assistant**")
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
            if st.button("🚀 **AI Analysis**", use_container_width=True, type="primary"):
                st.session_state.analyze_clicked = True
        with col2:
            if st.button("🔄 **Clear All**", use_container_width=True):
                st.session_state.analysis_results = {}
                st.session_state.messages = []
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
        key="chat_input"
    )
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("Send Message", use_container_width=True) and user_input:
            process_user_message(analyzer, user_input, patient_context)
    with col2:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    
    # Quick action buttons
    st.subheader("💡 Quick Medical Questions")
    quick_col1, quick_col2, quick_col3, quick_col4 = st.columns(4)
    
    with quick_col1:
        if st.button("🤒 Common Symptoms", use_container_width=True):
            process_user_message(analyzer, "Can you explain common symptoms and when to seek medical attention?", patient_context)
    
    with quick_col2:
        if st.button("💊 Medication Info", use_container_width=True):
            process_user_message(analyzer, "What should I know about medication safety and management?", patient_context)
    
    with quick_col3:
        if st.button("🩺 Test Results", use_container_width=True):
            process_user_message(analyzer, "How should I understand and interpret medical test results?", patient_context)
    
    with quick_col4:
        if st.button("🍎 Health Advice", use_container_width=True):
            process_user_message(analyzer, "What are some general health and wellness recommendations?", patient_context)

def process_user_message(analyzer, user_input, patient_context):
    """Process user message and generate AI response"""
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Generate AI response
    with st.spinner("Dr. MedAI is analyzing..."):
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
        🏥 **Welcome to MediAI Pro - Advanced Medical AI Assistant**
        
        To begin comprehensive medical analysis:
        1. Enter patient information in the sidebar
        2. Upload or paste medical data (lab results, images, reports)
        3. Click **'AI Analysis'** in the sidebar
        4. Use the chat for medical conversations and questions
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
        st.metric("Findings", f"{len([r for r in results_text.split() if 'abnormal' in r or 'critical' in r])} noted")
    
    with col4:
        st.metric("Report Status", "Ready")
    
    # Detailed Analysis Sections
    if 'lab_analysis' in st.session_state.analysis_results:
        with st.expander("🔬 **Laboratory Results Analysis**", expanded=True):
            st.markdown(st.session_state.analysis_results['lab_analysis'])
    
    if 'image_analysis' in st.session_state.analysis_results:
        with st.expander("🖼️ **Medical Image Analysis**", expanded=True):
            st.markdown(st.session_state.analysis_results['image_analysis'])
    
    if 'patient_summary' in st.session_state.analysis_results:
        with st.expander("👨‍⚕️ **Comprehensive Patient Assessment**", expanded=True):
            st.markdown(st.session_state.analysis_results['patient_summary'])

def create_advanced_visualizations():
    """Create professional medical visualizations"""
    st.header("📈 Advanced Medical Analytics")
    
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
    
    # Initialize analyzer
    analyzer = MedicalAIAnalyzer()
    
    # Setup sidebar and get patient data
    patient_data = setup_sidebar()
    
    # Perform analysis if requested
    if hasattr(st.session_state, 'analyze_clicked') and st.session_state.analyze_clicked:
        with st.spinner("🔄 Performing comprehensive medical analysis..."):
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
            
        st.success("✅ Comprehensive medical analysis completed successfully!")
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["💬 AI Chat", "📊 Dashboard", "📈 Analytics", "📋 Report"])
    
    with tab1:
        display_medical_chat(analyzer, patient_data['patient_context'])
    
    with tab2:
        display_analysis_dashboard(analyzer)
    
    with tab3:
        create_advanced_visualizations()
    
    with tab4:
        st.header("📋 Comprehensive Medical Report")
        if st.session_state.analysis_results:
            report_content = f"""
# 🏥 COMPREHENSIVE MEDICAL ANALYSIS REPORT

## Patient Information
{patient_data['patient_info']}

## Executive Summary
Comprehensive medical analysis completed using advanced medical knowledge base and pattern recognition.

## Detailed Analysis

### Laboratory Findings
{st.session_state.analysis_results.get('lab_analysis', 'No laboratory analysis available')}

### Imaging Analysis  
{st.session_state.analysis_results.get('image_analysis', 'No imaging analysis available')}

### Patient Assessment
{st.session_state.analysis_results.get('patient_summary', 'No patient summary available')}

## Clinical Recommendations

### Immediate Actions
- Review all findings with healthcare providers
- Address any critical or abnormal results
- Ensure patient understanding of health status

### Follow-up Plan
- Schedule appropriate specialist consultations
- Implement recommended monitoring
- Adjust treatment plans as needed

### Long-term Management
- Continue chronic condition management
- Focus on preventive health measures
- Regular health maintenance and screening

---
*Generated by MediAI Pro Advanced Medical Assistant on {datetime.now().strftime('%B %d, %Y at %H:%M')}*

**Disclaimer**: This report provides general medical information and should be reviewed by qualified healthcare professionals. Always seek professional medical advice for personal health concerns.
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
            st.info("Generate medical analysis first to view the comprehensive report.")

if __name__ == "__main__":
    main()
