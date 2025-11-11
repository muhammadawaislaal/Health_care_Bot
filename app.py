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
if 'processed_lab_data' not in st.session_state:
    st.session_state.processed_lab_data = ""

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
        if not self.api_key:
            return self._intelligent_fallback_response(message, patient_context, processed_data)
        
        try:
            # Build intelligent system prompt
            system_prompt = f"""You are Dr. MedAI, an advanced AI medical assistant with deep medical knowledge. Analyze the patient information and provide specific, actionable medical guidance.

PATIENT CONTEXT:
{patient_context if patient_context else 'No specific patient information provided yet.'}

PROCESSED MEDICAL DATA:
{processed_data if processed_data else 'No additional medical data processed.'}

CRITICAL INSTRUCTIONS:
1. BE SPECIFIC AND ACTIONABLE - Provide concrete recommendations based on the data
2. ANALYZE ALL AVAILABLE DATA - Use lab results, symptoms, and medical history
3. PROVIDE EVIDENCE-BASED ADVICE - Reference medical guidelines when possible
4. BE CONVERSATIONAL AND EMPATHETIC - Talk like a caring doctor
5. IDENTIFY URGENT CONCERNS - Flag any critical findings immediately
6. SUGGEST NEXT STEPS - Recommend specific tests, specialists, or treatments
7. EDUCATE THE PATIENT - Explain medical terms and conditions clearly
8. ALWAYS RECOMMEND FOLLOW-UP - Emphasize professional medical consultation

RESPONSE STYLE:
- Be natural, conversational, and warm
- Show genuine concern for patient wellbeing
- Use bullet points for clarity when needed
- Provide specific numbers and ranges when discussing lab results
- Explain what abnormal values mean in practical terms
- Suggest dietary, lifestyle, and medical interventions

IMPORTANT: Never provide definitive diagnoses. Always emphasize consulting healthcare providers."""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message}
            ]

            # Get response from Groq API
            response = self.call_groq_api(messages, model="llama3-8b-8192", temperature=0.7, max_tokens=2000)
            
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
                return """👋 Hello! I can see you've mentioned iron deficiency in your medical information. 

I understand this can be concerning. Let me help you understand what this means and what steps you can take.

Based on iron deficiency, here's what I can help with:
• Understanding your lab results and what they mean
• Dietary recommendations to improve iron levels
• Symptoms management and when to seek urgent care
• Questions to ask your doctor

What specific aspect would you like to discuss first?"""
            else:
                return """👋 Hello! I'm Dr. MedAI, your medical assistant.

I'm here to help you understand:
• Your medical test results and what they mean
• Symptoms and potential causes
• Medication questions and interactions
• Health and wellness guidance

Please share your medical concerns, test results, or upload your medical reports, and I'll provide detailed, personalized analysis."""

        elif any(word in message_lower for word in ['iron', 'deficiency', 'anemia', 'ferritin', 'hemoglobin']):
            return self._provide_iron_deficiency_analysis(patient_context, processed_data)
        
        elif any(word in message_lower for word in ['symptom', 'tired', 'fatigue', 'weak', 'dizzy', 'pale']):
            return self._analyze_symptoms(message_lower, patient_context, processed_data)
        
        elif any(word in message_lower for word in ['lab', 'test', 'result', 'report']):
            return self._analyze_lab_results(processed_data)
        
        elif any(word in message_lower for word in ['diet', 'food', 'nutrition', 'eat']):
            return self._provide_nutrition_advice(patient_context, processed_data)
        
        else:
            return """I'd be happy to help with your health questions! To provide the most accurate and helpful information, please:

1. **Share your specific symptoms or concerns**
2. **Upload your medical reports or test results**
3. **Tell me about any diagnoses you've received**
4. **Describe your current medications or treatments**

The more information you provide, the better I can assist you with personalized guidance and explanations."""

    def _provide_iron_deficiency_analysis(self, patient_context, processed_data):
        """Provide detailed iron deficiency analysis"""
        analysis = """🔬 **Comprehensive Iron Deficiency Analysis**

**Understanding Your Condition:**
Iron deficiency is one of the most common nutritional deficiencies worldwide. It occurs when your body doesn't have enough iron to produce adequate hemoglobin, the protein in red blood cells that carries oxygen.

**Common Symptoms:**
• Fatigue and weakness
• Pale skin and conjunctiva
• Shortness of breath
• Dizziness or lightheadedness
• Headaches
• Cold hands and feet
• Brittle nails
• Unusual cravings for non-nutritive substances (pica)

**Key Laboratory Findings:**
"""
        
        # Extract and analyze lab values
        lab_values = self.extract_lab_values(processed_data)
        
        critical_findings = []
        if 'Hemoglobin' in lab_values:
            hb = lab_values['Hemoglobin']
            if hb < 12.0:
                critical_findings.append(f"Hemoglobin: {hb} g/dL (LOW - indicates anemia)")
            else:
                analysis += f"• Hemoglobin: {hb} g/dL (Normal range: 12.0-16.0 g/dL)\n"
        
        if 'Ferritin' in lab_values:
            ferritin = lab_values['Ferritin']
            if ferritin < 15:
                critical_findings.append(f"Ferritin: {ferritin} ng/mL (VERY LOW - confirms iron deficiency)")
            elif ferritin < 30:
                critical_findings.append(f"Ferritin: {ferritin} ng/mL (LOW - indicates iron deficiency)")
            else:
                analysis += f"• Ferritin: {ferritin} ng/mL (Normal range: 15-150 ng/mL)\n"
        
        if critical_findings:
            analysis += "\n🚨 **CRITICAL FINDINGS:**\n" + "\n".join([f"• {finding}" for finding in critical_findings]) + "\n"
        
        analysis += """
**Recommended Actions:**

1. **Medical Consultation:** Schedule appointment with hematologist or primary care physician
2. **Iron Supplementation:** Typically 65-200 mg elemental iron daily
3. **Dietary Changes:** Increase iron-rich foods (red meat, spinach, lentils)
4. **Vitamin C:** Take with iron supplements to enhance absorption
5. **Follow-up Testing:** Repeat blood tests in 2-3 months
6. **Identify Cause:** Investigate potential blood loss or absorption issues

**When to Seek Immediate Care:**
• Severe fatigue preventing daily activities
• Chest pain or palpitations
• Shortness of breath at rest
• Significant dizziness or fainting

Would you like me to analyze specific lab values or discuss dietary recommendations in more detail?"""
        
        return analysis

    def _analyze_symptoms(self, message, patient_context, processed_data):
        """Analyze patient symptoms"""
        return """🤒 **Symptom Analysis**

I understand you're experiencing symptoms. Let me help you understand what they might indicate:

**Common Symptom Patterns:**

**Fatigue + Weakness:**
• Iron deficiency anemia (most common)
• Thyroid disorders
• Sleep apnea or poor sleep quality
• Chronic fatigue syndrome
• Nutritional deficiencies

**Dizziness + Pale Skin:**
• Anemia (reduced oxygen delivery)
• Dehydration
• Blood pressure issues
• Inner ear problems

**Next Steps:**
1. **Track your symptoms** - note timing, triggers, severity
2. **Get comprehensive blood work** - CBC, iron studies, thyroid panel
3. **Discuss with your doctor** - bring your symptom diary
4. **Consider lifestyle factors** - sleep, stress, diet, exercise

**Red Flags Requiring Urgent Attention:**
• Chest pain or pressure
• Difficulty breathing
• Fainting or near-fainting
• Severe, persistent headache
• Rapid heart rate

Would you like to discuss specific symptoms in more detail or upload your test results for personalized analysis?"""

    def _analyze_lab_results(self, processed_data):
        """Analyze laboratory results"""
        if not processed_data:
            return "I don't see any lab results to analyze. Please upload your lab reports or paste your test results in the sidebar for detailed analysis."
        
        lab_values = self.extract_lab_values(processed_data)
        
        if not lab_values:
            return "I found some medical data but couldn't extract specific lab values. Could you please paste your results in this format:\n\nHemoglobin: 12.5 g/dL\nFerritin: 25 ng/mL\nWBC: 6.8\n\nThis will help me provide more accurate analysis."
        
        analysis = "🔬 **Laboratory Results Analysis**\n\n"
        
        abnormal_count = 0
        for test, value in lab_values.items():
            if test in self.medical_knowledge_base['lab_ranges']:
                ranges = self.medical_knowledge_base['lab_ranges'][test]
                if value < ranges['low']:
                    analysis += f"⚠️ **{test}**: {value} {ranges['unit']} **(LOW)** - Normal: {ranges['low']}-{ranges['high']} {ranges['unit']}\n"
                    abnormal_count += 1
                elif value > ranges['high']:
                    analysis += f"⚠️ **{test}**: {value} {ranges['unit']} **(HIGH)** - Normal: {ranges['low']}-{ranges['high']} {ranges['unit']}\n"
                    abnormal_count += 1
                else:
                    analysis += f"✅ **{test}**: {value} {ranges['unit']} (Normal)\n"
        
        if abnormal_count > 0:
            analysis += f"\n**Found {abnormal_count} abnormal values** that should be discussed with your healthcare provider.\n"
        else:
            analysis += "\n**All analyzed values are within normal ranges.**\n"
        
        analysis += "\n**Recommendations:**\n• Discuss these results with your doctor\n• Consider follow-up testing if symptoms persist\n• Maintain records for future reference"
        
        return analysis

    def _provide_nutrition_advice(self, patient_context, processed_data):
        """Provide nutrition advice based on condition"""
        if "iron" in patient_context.lower() or "iron" in processed_data.lower():
            return """🍎 **Nutrition for Iron Deficiency**

**Iron-Rich Foods to Include:**

**Heme Iron (better absorption):**
• Red meat (beef, lamb) - 3oz provides 2-3mg iron
• Poultry (chicken, turkey)
• Fish and seafood (especially oysters, clams)
• Organ meats (liver) - very high in iron

**Non-Heme Iron:**
• Spinach and leafy greens (cooked)
• Lentils and beans
• Fortified cereals and grains
• Tofu and soy products
• Pumpkin seeds

**Enhance Iron Absorption:**
• **Vitamin C**: Citrus fruits, bell peppers, broccoli, strawberries
• **Avoid with meals**: Tea, coffee, calcium supplements (wait 1-2 hours)
• **Cook in cast iron**: Can increase iron content of food

**Sample Iron-Rich Day:**
• Breakfast: Fortified cereal with strawberries
• Lunch: Spinach salad with lean beef and bell peppers
• Dinner: Lentil soup with orange slices
• Snack: Pumpkin seeds

**Daily Iron Goals:**
• Men: 8mg • Women (19-50): 18mg • Women (51+): 8mg"""
        
        return """🍎 **General Nutrition for Good Health**

**Balanced Diet Principles:**
• Fill half your plate with vegetables and fruits
• Include lean protein with each meal
• Choose whole grains over refined grains
• Stay hydrated with water
• Limit processed foods and added sugars

**Key Nutrients for Energy:**
• Iron: Meat, spinach, lentils
• B12: Animal products, fortified foods
• Vitamin D: Sunlight, fatty fish, fortified dairy
• Magnesium: Nuts, seeds, leafy greens"""

    def analyze_patient_data(self, patient_info, medical_history, lab_data, image_description, processed_files_text=""):
        """Comprehensive analysis of all patient data"""
        if not self.api_key:
            return self._basic_patient_analysis(patient_info, medical_history, lab_data, image_description, processed_files_text)
        
        try:
            # Combine all data
            combined_data = f"""
PATIENT INFORMATION:
{patient_info}

MEDICAL HISTORY:
{medical_history}

LABORATORY DATA:
{lab_data}

ADDITIONAL MEDICAL DATA:
{processed_files_text}

IMAGING/CLINICAL NOTES:
{image_description}
"""

            prompt = f"""
            Perform a comprehensive medical analysis of this patient case. Provide specific, actionable insights.

            PATIENT DATA:
            {combined_data}

            Please provide a detailed analysis with these sections:

            **CLINICAL SUMMARY**: Brief overview of the patient case
            **KEY FINDINGS**: Specific abnormalities and notable results with interpretation
            **RISK ASSESSMENT**: Health risks based on available data
            **CLINICAL CORRELATION**: How findings relate to patient presentation
            **ACTIONABLE RECOMMENDATIONS**: Clear next steps including:
               - Specific tests needed
               - Specialist consultations to consider
               - Lifestyle modifications
               - Medication considerations
            **URGENCY LEVEL**: How quickly action is needed (Low/Medium/High/Critical)
            **PATIENT EDUCATION**: What the patient needs to understand about their condition

            Be specific, practical, and evidence-based. Use numbers and reference ranges when discussing lab results.
            """

            messages = [
                {
                    "role": "system", 
                    "content": "You are an experienced physician analyzing comprehensive patient data. Provide thorough, evidence-based analysis with clear clinical recommendations. Focus on actionable insights and patient safety. Be specific and practical."
                },
                {"role": "user", "content": prompt}
            ]

            response = self.call_groq_api(messages, max_tokens=2500, temperature=0.3)
            
            if response:
                return response
            else:
                return self._basic_patient_analysis(patient_info, medical_history, lab_data, image_description, processed_files_text)
                
        except Exception as e:
            logger.error(f"Comprehensive analysis failed: {str(e)}")
            return self._basic_patient_analysis(patient_info, medical_history, lab_data, image_description, processed_files_text)

    def _basic_patient_analysis(self, patient_info, medical_history, lab_data, image_description, processed_files_text):
        """Basic analysis when AI is unavailable"""
        analysis = "## 📊 Comprehensive Patient Data Analysis\n\n"
        
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
                        analysis += f"⚠️ **{test}**: {value} {ranges['unit']} **(LOW)** - Normal range: {ranges['low']}-{ranges['high']} {ranges['unit']}\n"
                        abnormal_count += 1
                    elif value > ranges['high']:
                        analysis += f"⚠️ **{test}**: {value} {ranges['unit']} **(HIGH)** - Normal range: {ranges['low']}-{ranges['high']} {ranges['unit']}\n"
                        abnormal_count += 1
                    else:
                        analysis += f"✅ **{test}**: {value} {ranges['unit']} (Normal)\n"
            
            if abnormal_count > 0:
                analysis += f"\n**Found {abnormal_count} abnormal values requiring medical attention.**\n"
        
        if processed_files_text:
            analysis += f"\n### 📁 Processed Medical Documents\nMedical documents have been processed and included in analysis.\n"
        
        if image_description and image_description.strip():
            analysis += f"\n### 🖼️ Clinical Notes\n{image_description}\n"
        
        analysis += "\n### 💡 Recommended Next Steps\n"
        analysis += "1. **Review all findings** with your healthcare provider\n"
        analysis += "2. **Discuss abnormal values** for proper clinical interpretation\n"
        analysis += "3. **Consider specialist referral** if indicated by findings\n"
        analysis += "4. **Schedule follow-up testing** as needed\n"
        analysis += "5. **Implement lifestyle changes** based on medical advice\n"
        
        return analysis

    def extract_lab_values(self, text):
        """Extract laboratory values from text with improved pattern matching"""
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
            'Total Cholesterol': r'Total Cholesterol[\s:\-]*([\d.]+)|Cholesterol[\s:\-]*([\d.]+)',
            'LDL': r'LDL[\s:\-]*([\d.]+)',
            'HDL': r'HDL[\s:\-]*([\d.]+)',
            'Triglycerides': r'Triglycerides[\s:\-]*([\d.]+)',
            'Iron': r'Iron[\s:\-]*([\d.]+)',
            'Ferritin': r'Ferritin[\s:\-]*([\d.]+)',
            'TIBC': r'TIBC[\s:\-]*([\d.]+)',
            'Transferrin Saturation': r'Transferrin Saturation[\s:\-]*([\d.]+)|Sat[\s:\-]*([\d.]+)'
        }
        
        extracted_values = {}
        for test, pattern in patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                # Find the first non-empty match in the tuple
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
            patient_age = st.number_input("Age", min_value=0, max_value=120, value=45, key="patient_age")
        with col2:
            patient_name = st.text_input("Patient Name", value="John Doe", key="patient_name")
            patient_gender = st.selectbox("Gender", ["Male", "Female", "Other"], key="patient_gender")
        
        st.header("📋 Medical History")
        medical_history = st.text_area(
            "Medical History & Symptoms", 
            height=100,
            placeholder="Example: Iron deficiency, fatigue, frequent headaches, pale skin...",
            key="medical_history"
        )
        
        st.header("📁 Medical Data Upload")
        
        tab1, tab2 = st.tabs(["Lab Reports", "Medical Notes"])
        
        with tab1:
            st.subheader("Laboratory Results")
            lab_files = st.file_uploader("Upload Lab Reports", 
                                       type=['pdf', 'docx', 'txt'], 
                                       key="lab_uploader",
                                       accept_multiple_files=True,
                                       help="Upload PDF, DOCX, or TXT files containing lab results")
            
            if lab_files:
                st.info(f"📎 {len(lab_files)} file(s) uploaded")
                for file in lab_files:
                    st.write(f"• {file.name} ({file.size//1024} KB)")
            
            lab_text_input = st.text_area("Or Paste Lab Results:", height=120,
                                        placeholder="""Example:
Hemoglobin: 11.2 g/dL (Low)
Ferritin: 12 ng/mL (Low)
Iron: 45 μg/dL (Low)
WBC: 6.8 10^3/μL
Platelets: 245 10^3/μL
Glucose: 95 mg/dL""",
                                        key="lab_text")
        
        with tab2:
            st.subheader("Medical Notes & Symptoms")
            image_description = st.text_area("Clinical Notes & Findings:", height=120,
                                           placeholder="""Example:
Patient reports persistent fatigue and pale skin.
Diagnosed with iron deficiency anemia.
Recommended iron supplements and dietary changes.
Follow-up in 3 months.""",
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
Medical History: {medical_history if medical_history else 'Not specified'}
Laboratory Data: {lab_text_input if lab_text_input else 'Not provided'}
Medical Notes: {image_description if image_description else 'Not provided'}
Uploaded Files: {'Yes' if lab_files else 'No'}
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
        st.success("✅ AI Assistant: **ACTIVE** - Advanced AI conversations enabled")
    else:
        st.warning("🟡 AI Assistant: **BASIC MODE** - Using medical knowledge base")
    
    # Patient context summary
    if st.session_state.patient_context:
        with st.expander("📋 Current Patient Context", expanded=False):
            st.text(st.session_state.patient_context)
    
    # Processed data summary
    if st.session_state.processed_lab_data:
        with st.expander("📊 Processed Medical Data", expanded=False):
            st.text(st.session_state.processed_lab_data[:500] + "..." if len(st.session_state.processed_lab_data) > 500 else st.session_state.processed_lab_data)
    
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
    quick_col1, quick_col2, quick_col3 = st.columns(3)
    
    with quick_col1:
        if st.button("🤒 Discuss Symptoms", use_container_width=True):
            process_user_message(analyzer, "I'd like to discuss my symptoms and what they might mean.")
        if st.button("💊 Medication Questions", use_container_width=True):
            process_user_message(analyzer, "I have questions about medications and supplements.")
    
    with quick_col2:
        if st.button("🔬 Analyze Test Results", use_container_width=True):
            process_user_message(analyzer, "Can you help me understand my medical test results?")
        if st.button("🍎 Nutrition Advice", use_container_width=True):
            process_user_message(analyzer, "What dietary recommendations do you suggest for my condition?")
    
    with quick_col3:
        if st.button("📋 Next Steps", use_container_width=True):
            process_user_message(analyzer, "What are the recommended next steps for my medical condition?")
        if st.button("🩺 General Health", use_container_width=True):
            process_user_message(analyzer, "What are evidence-based recommendations for maintaining good health?")

def process_user_message(analyzer, user_input):
    """Process user message and generate AI response"""
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Generate AI response
    with st.spinner("Dr. MedAI is analyzing your question..."):
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
        🏥 **Welcome to MediAI Pro - Medical Assistant**
        
        To begin medical analysis:
        1. 👤 Enter patient information in the sidebar
        2. 📋 Add medical history, symptoms, or test results
        3. 📁 Upload medical reports (PDF, DOCX, TXT)
        4. 🚀 Click **'Analyze Patient Data'** for comprehensive analysis
        5. 💬 Use the chat for personalized medical conversations
        
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
        has_data = any([
            patient_data['medical_history'], 
            patient_data['lab_data'], 
            patient_data['processed_files_text'],
            patient_data['image_description']
        ])
        st.metric("Patient Data", "Loaded" if has_data else "Minimal")
    
    with col4:
        lab_values = analyzer.extract_lab_values(patient_data['lab_data'] + patient_data['processed_files_text'])
        abnormal_count = sum(1 for test, value in lab_values.items() 
                           if test in analyzer.medical_knowledge_base['lab_ranges'] 
                           and (value < analyzer.medical_knowledge_base['lab_ranges'][test]['low'] 
                                or value > analyzer.medical_knowledge_base['lab_ranges'][test]['high']))
        st.metric("Abnormal Values", abnormal_count if lab_values else "N/A")
    
    # Detailed Analysis Sections
    if 'comprehensive_analysis' in st.session_state.analysis_results:
        with st.expander("📋 **Comprehensive Patient Analysis**", expanded=True):
            st.markdown(st.session_state.analysis_results['comprehensive_analysis'])

def create_advanced_visualizations(patient_data, analyzer):
    """Create professional medical visualizations"""
    st.header("📈 Health Analytics & Trends")
    
    # Extract lab values for visualization
    combined_data = patient_data['lab_data'] + patient_data['processed_files_text']
    lab_values = analyzer.extract_lab_values(combined_data)
    
    if lab_values:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🩸 Current Lab Values")
            
            # Prepare data for visualization
            viz_data = []
            for test, value in lab_values.items():
                if test in analyzer.medical_knowledge_base['lab_ranges']:
                    ranges = analyzer.medical_knowledge_base['lab_ranges'][test]
                    status = "Normal"
                    if value < ranges['low']:
                        status = "Low"
                    elif value > ranges['high']:
                        status = "High"
                    
                    viz_data.append({
                        'Parameter': test,
                        'Value': value,
                        'Lower Limit': ranges['low'],
                        'Upper Limit': ranges['high'],
                        'Status': status
                    })
            
            if viz_data:
                df = pd.DataFrame(viz_data)
                
                fig = go.Figure()
                
                # Add normal range
                fig.add_trace(go.Scatter(
                    x=df['Parameter'], 
                    y=df['Upper Limit'],
                    mode='lines',
                    name='Upper Limit',
                    line=dict(dash='dash', color='red')
                ))
                fig.add_trace(go.Scatter(
                    x=df['Parameter'], 
                    y=df['Lower Limit'],
                    mode='lines',
                    name='Lower Limit',
                    line=dict(dash='dash', color='red'),
                    fill='tonexty',
                    fillcolor='rgba(255,0,0,0.1)'
                ))
                
                # Add actual values with color coding
                colors = []
                for status in df['Status']:
                    if status == "Low":
                        colors.append('blue')
                    elif status == "High":
                        colors.append('red')
                    else:
                        colors.append('green')
                
                fig.add_trace(go.Bar(
                    name='Your Values', 
                    x=df['Parameter'], 
                    y=df['Value'],
                    marker_color=colors,
                    text=df['Value'],
                    textposition='auto'
                ))
                
                fig.update_layout(
                    title='Laboratory Values Overview',
                    yaxis_title='Value',
                    showlegend=True,
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📊 Status Summary")
            
            status_counts = df['Status'].value_counts()
            fig_pie = px.pie(
                values=status_counts.values,
                names=status_counts.index,
                title='Lab Value Status Distribution',
                color=status_counts.index,
                color_discrete_map={'Normal': 'green', 'Low': 'blue', 'High': 'red'}
            )
            st.plotly_chart(fig_pie, use_container_width=True)
    
    else:
        st.info("Upload lab reports or enter lab values in the sidebar to see visualizations.")

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
                patient_data['image_description'],
                patient_data['processed_files_text']
            )
            
            st.session_state.analysis_results = analysis_results
            st.session_state.analyze_clicked = False
            
        st.success("✅ Comprehensive medical analysis completed!")
        st.rerun()
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["💬 Medical Chat", "📊 Analysis", "📈 Health Insights", "📋 Report"])
    
    with tab1:
        display_medical_chat(analyzer)
    
    with tab2:
        display_analysis_dashboard(analyzer, patient_data)
    
    with tab3:
        create_advanced_visualizations(patient_data, analyzer)
    
    with tab4:
        st.header("📋 Medical Summary Report")
        if st.session_state.analysis_results:
            report_content = f"""
# 🏥 MEDICAL ANALYSIS REPORT
## Generated by MediAI Pro

### Patient Information
{patient_data['patient_info']}

### Comprehensive Analysis
{st.session_state.analysis_results.get('comprehensive_analysis', 'No analysis available')}

### Key Recommendations Summary

#### Immediate Actions
- Review all findings with your healthcare provider
- Address any identified deficiencies or abnormalities
- Implement recommended lifestyle modifications
- Schedule necessary follow-up appointments

#### Medical Follow-up
- Specialist consultations as indicated
- Repeat testing per clinical guidelines
- Medication adjustments if needed
- Ongoing symptom monitoring

#### Long-term Health Strategy
- Focus on evidence-based preventive care
- Maintain regular health monitoring
- Continue management of chronic conditions
- Adopt healthy lifestyle habits

---
*Report generated on {datetime.now().strftime('%B %d, %Y at %H:%M')}*

**Medical Disclaimer**: This analysis provides educational information and should be reviewed by qualified healthcare professionals. Always seek professional medical advice for personal health concerns. This report does not constitute medical diagnosis or treatment recommendations.
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
                    file_name=f"Analysis_{datetime.now().strftime('%Y%m%d')}.json",
                    mime="application/json",
                    use_container_width=True
                )
        else:
            st.info("Click 'Analyze Patient Data' in the sidebar to generate a comprehensive medical report.")

if __name__ == "__main__":
    main()
