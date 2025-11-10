import streamlit as st
import openai
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure the page for cloud deployment
st.set_page_config(
    page_title="MediAI - Doctor's Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for cloud deployment
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'patient_data' not in st.session_state:
    st.session_state.patient_data = {}
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {}
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'api_key_configured' not in st.session_state:
    st.session_state.api_key_configured = False
if 'chatbot' not in st.session_state:
    st.session_state.chatbot = None

class MedicalChatbot:
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.client_configured = False
        
        if api_key and api_key.startswith('sk-'):
            try:
                # Configure OpenAI with the API key
                openai.api_key = api_key
                self.client = openai.OpenAI(api_key=api_key)
                self.client_configured = True
                logger.info("OpenAI client configured successfully")
            except Exception as e:
                logger.error(f"Error configuring OpenAI: {str(e)}")
                self.client_configured = False
                st.error(f"❌ Error configuring OpenAI: {str(e)}")
        else:
            self.client_configured = False
    
    def extract_text_from_pdf(self, pdf_file):
        """Extract text from PDF files"""
        try:
            text = ""
            # Create a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(pdf_file.getvalue())
                tmp_file_path = tmp_file.name
            
            with pdfplumber.open(tmp_file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            
            # Clean up temporary file
            os.unlink(tmp_file_path)
            return text if text else "No text could be extracted from PDF"
        except Exception as e:
            logger.error(f"PDF extraction error: {str(e)}")
            return f"Error extracting PDF: {str(e)}"
    
    def extract_text_from_docx(self, docx_file):
        """Extract text from DOCX files"""
        try:
            # Create a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as tmp_file:
                tmp_file.write(docx_file.getvalue())
                tmp_file_path = tmp_file.name
            
            doc = docx.Document(tmp_file_path)
            text = ""
            for paragraph in doc.paragraphs:
                if paragraph.text:
                    text += paragraph.text + "\n"
            
            # Clean up temporary file
            os.unlink(tmp_file_path)
            return text if text else "No text could be extracted from DOCX"
        except Exception as e:
            logger.error(f"DOCX extraction error: {str(e)}")
            return f"Error extracting DOCX: {str(e)}"
    
    def extract_text_from_file(self, file):
        """Extract text from various file types"""
        if file.type == "application/pdf":
            return self.extract_text_from_pdf(file)
        elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            return self.extract_text_from_docx(file)
        elif file.type == "text/plain":
            return str(file.read(), 'utf-8')
        else:
            return f"Unsupported file type: {file.type}"
    
    def analyze_lab_results(self, lab_data):
        """Analyze laboratory test results"""
        if not self.client_configured:
            return "Please configure API key first"
        
        try:
            prompt = f"""
            Analyze these laboratory results and provide a comprehensive medical assessment:
            
            {lab_data}
            
            Please provide a structured analysis with:
            1. **Critical Findings**: Abnormal values that require immediate attention
            2. **Clinical Interpretation**: What these results mean medically
            3. **Differential Diagnosis**: Potential conditions to consider
            4. **Recommended Actions**: Follow-up tests, consultations, or treatments
            5. **Urgency Level**: Low/Medium/High with justification
            6. **Patient Counseling Points**: What to discuss with the patient
            
            Format this professionally for medical practitioners.
            """
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a senior medical consultant with 20 years experience. Provide evidence-based, accurate medical analysis. Always emphasize patient safety and recommend specialist consultation when needed."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=2000
            )
            
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Lab analysis error: {str(e)}")
            return f"Error analyzing lab results: {str(e)}"
    
    def analyze_medical_image(self, image_description=""):
        """Analyze medical images with description"""
        if not self.client_configured:
            return "Please configure API key first"
        
        try:
            prompt = f"""
            Analyze this medical image based on the following description and findings:
            
            Image Description/Findings: {image_description}
            
            Please provide a comprehensive radiological assessment:
            1. **Technical Quality**: Assessment of image quality
            2. **Key Findings**: Detailed observations
            3. **Abnormalities**: Any pathological findings
            4. **Differential Diagnosis**: Possible conditions
            5. **Correlation**: How findings relate to patient's clinical picture
            6. **Recommendations**: Further imaging, follow-up, or specialist referral
            
            Provide this in standard radiology report format.
            """
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a board-certified radiologist. Provide detailed, professional image analysis following standard radiology reporting protocols."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=2000
            )
            
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Image analysis error: {str(e)}")
            return f"Error analyzing image: {str(e)}"
    
    def analyze_medical_report(self, report_text, report_type="Clinical"):
        """Analyze various medical reports"""
        if not self.client_configured:
            return "Please configure API key first"
        
        try:
            prompt = f"""
            Analyze this {report_type} medical report:
            
            {report_text}
            
            Please provide:
            1. **Executive Summary**: Key findings and implications
            2. **Critical Elements**: Components requiring immediate attention
            3. **Clinical Correlation**: How findings relate to patient management
            4. **Action Plan**: Recommended next steps
            5. **Risk Assessment**: Potential complications or risks
            6. **Follow-up Schedule**: Recommended monitoring timeline
            """
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a medical specialist analyzing clinical reports. Provide thorough, actionable insights for clinical decision making."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=2000
            )
            
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Report analysis error: {str(e)}")
            return f"Error analyzing report: {str(e)}"
    
    def generate_patient_summary(self, patient_info, medical_history, current_findings):
        """Generate comprehensive patient summary"""
        if not self.client_configured:
            return "Please configure API key first"
        
        try:
            prompt = f"""
            Generate a comprehensive patient summary and care plan:
            
            PATIENT INFORMATION: {patient_info}
            MEDICAL HISTORY: {medical_history}
            CURRENT FINDINGS: {current_findings}
            
            Please provide:
            
            **COMPREHENSIVE ASSESSMENT**
            - Current clinical status
            - Problem list with prioritization
            - Risk stratification
            
            **TREATMENT PLAN**
            - Immediate interventions
            - Medication recommendations
            - Lifestyle modifications
            
            **MONITORING & FOLLOW-UP**
            - Vital parameters to track
            - Laboratory monitoring schedule
            - Specialist referrals needed
            
            **PATIENT EDUCATION**
            - Key points for patient discussion
            - Warning signs to watch for
            - Self-management strategies
            """
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an experienced physician creating comprehensive patient care plans. Focus on practical, actionable recommendations."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=2500
            )
            
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Patient summary error: {str(e)}")
            return f"Error generating summary: {str(e)}"
    
    def chat_response(self, messages):
        """Generate chat response"""
        if not self.client_configured:
            return "Please configure API key first to use the chat feature."
        
        try:
            system_message = {
                "role": "system", 
                "content": """You are Dr. MedAI, an AI medical assistant for healthcare professionals. 

                YOUR ROLE:
                - Provide evidence-based medical information
                - Assist with clinical decision support
                - Help interpret medical data and reports
                - Suggest differential diagnoses
                - Recommend diagnostic pathways
                - Always emphasize consulting specialists for complex cases

                COMMUNICATION STYLE:
                - Professional and precise
                - Empathetic but clinical
                - Clear and structured responses
                - Cite medical evidence when possible
                - Acknowledge limitations of AI in medicine

                SAFETY PROTOCOLS:
                - Never provide definitive diagnoses
                - Always recommend human physician review
                - Highlight urgent findings that need immediate attention
                - Suggest appropriate specialist consultations
                """
            }
            
            chat_messages = [system_message] + [{"role": m["role"], "content": m["content"]} for m in messages]
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=chat_messages,
                temperature=0.3,
                max_tokens=1500
            )
            
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Chat error: {str(e)}")
            return f"Error in chat: {str(e)}"

def setup_api_configuration():
    """API configuration section"""
    st.sidebar.header("🔑 API Configuration")
    
    # Option 1: Streamlit secrets (recommended for cloud)
    if 'OPENAI_API_KEY' in st.secrets:
        api_key = st.secrets['OPENAI_API_KEY']
        if api_key and api_key.startswith('sk-'):
            st.sidebar.success("✅ API key loaded from Streamlit secrets!")
            
            # Initialize chatbot if not already done
            if st.session_state.chatbot is None or not st.session_state.chatbot.client_configured:
                st.session_state.chatbot = MedicalChatbot(api_key)
            
            return api_key
    
    # Option 2: Manual input as fallback
    st.sidebar.info("Configure your OpenAI API key")
    api_key = st.sidebar.text_input("Enter OpenAI API Key:", type="password", key="api_key_input")
    
    if api_key:
        if st.sidebar.button("Save API Key", key="save_api_key"):
            if api_key.startswith('sk-'):
                st.session_state.chatbot = MedicalChatbot(api_key)
                if st.session_state.chatbot.client_configured:
                    st.session_state.api_key_configured = True
                    st.sidebar.success("✅ API key configured successfully!")
                    st.rerun()
                else:
                    st.sidebar.error("❌ Failed to configure OpenAI client")
            else:
                st.sidebar.error("❌ Invalid API key format. Should start with 'sk-'")
    
    return None

def display_analysis_dashboard():
    """Display analysis results in a dashboard format"""
    st.header("📊 Medical Analysis Dashboard")
    
    if not st.session_state.analysis_results:
        st.info("👆 Upload patient data and click 'Analyze All' to see comprehensive analysis results here.")
        return
    
    # Summary Cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Analyses Completed", len(st.session_state.analysis_results))
    
    with col2:
        urgency = "Medium"
        results_text = " ".join([str(result) for result in st.session_state.analysis_results.values()]).lower()
        if "urgent" in results_text or "critical" in results_text or "emergency" in results_text:
            urgency = "High"
        elif "normal" in results_text and "abnormal" not in results_text:
            urgency = "Low"
        st.metric("Overall Urgency", urgency)
    
    with col3:
        st.metric("Reports Generated", "1")
    
    with col4:
        st.metric("Last Updated", datetime.now().strftime("%H:%M"))
    
    # Detailed Analysis Sections
    if 'lab_analysis' in st.session_state.analysis_results:
        with st.expander("🔬 Laboratory Results Analysis", expanded=True):
            st.markdown(st.session_state.analysis_results['lab_analysis'])
    
    if 'image_analysis' in st.session_state.analysis_results:
        with st.expander("🖼️ Medical Image Analysis", expanded=True):
            st.markdown(st.session_state.analysis_results['image_analysis'])
    
    if 'report_analysis' in st.session_state.analysis_results:
        with st.expander("📄 Medical Report Analysis", expanded=True):
            st.markdown(st.session_state.analysis_results['report_analysis'])
    
    if 'patient_summary' in st.session_state.analysis_results:
        with st.expander("👨‍⚕️ Comprehensive Patient Summary", expanded=True):
            st.markdown(st.session_state.analysis_results['patient_summary'])

def create_medical_visualizations():
    """Create comprehensive medical visualizations"""
    st.header("📈 Medical Data Visualizations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Laboratory Trends
        st.subheader("🩸 Laboratory Value Trends")
        lab_data = {
            'Test': ['WBC', 'Hemoglobin', 'Platelets', 'Glucose', 'Creatinine', 'ALT'],
            'Current': [8.2, 14.2, 250, 110, 1.1, 25],
            'Previous': [7.8, 13.8, 280, 98, 0.9, 22],
            'Unit': ['10^3/μL', 'g/dL', '10^3/μL', 'mg/dL', 'mg/dL', 'U/L']
        }
        
        df = pd.DataFrame(lab_data)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Current', x=df['Test'], y=df['Current'], marker_color='crimson'))
        fig.add_trace(go.Bar(name='Previous', x=df['Test'], y=df['Previous'], marker_color='lightblue'))
        
        fig.update_layout(
            title='Laboratory Values Comparison',
            barmode='group',
            showlegend=True,
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Vital Signs Monitor
        st.subheader("❤️ Vital Signs Trend")
        time_points = ['08:00', '12:00', '16:00', '20:00', '00:00', '04:00']
        heart_rate = [72, 75, 80, 78, 70, 68]
        blood_pressure_sys = [120, 118, 122, 119, 121, 118]
        oxygen_saturation = [98, 97, 96, 98, 97, 98]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=time_points, y=heart_rate, mode='lines+markers', name='Heart Rate', line=dict(color='red')))
        fig.add_trace(go.Scatter(x=time_points, y=blood_pressure_sys, mode='lines+markers', name='BP Systolic', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=time_points, y=oxygen_saturation, mode='lines+markers', name='SpO2', line=dict(color='green')))
        
        fig.update_layout(
            title='24-Hour Vital Signs Monitoring',
            xaxis_title='Time',
            yaxis_title='Value',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

def perform_comprehensive_analysis(chatbot, patient_info, medical_history, lab_data, image_description, report_data):
    """Perform comprehensive medical analysis"""
    analysis_results = {}
    
    # Analyze lab data
    if lab_data and lab_data.strip():
        with st.spinner("🔬 Analyzing laboratory results..."):
            analysis_results['lab_analysis'] = chatbot.analyze_lab_results(lab_data)
    
    # Analyze medical images
    if image_description and image_description.strip():
        with st.spinner("🖼️ Analyzing medical images..."):
            analysis_results['image_analysis'] = chatbot.analyze_medical_image(image_description)
    
    # Analyze medical reports
    if report_data and report_data.strip():
        with st.spinner("📄 Analyzing medical reports..."):
            analysis_results['report_analysis'] = chatbot.analyze_medical_report(report_data, "Clinical")
    
    # Generate comprehensive patient summary
    current_findings = f"Lab: {lab_data[:500]}, Imaging: {image_description[:500]}, Reports: {report_data[:500]}"
    with st.spinner("👨‍⚕️ Generating patient summary..."):
        analysis_results['patient_summary'] = chatbot.generate_patient_summary(
            patient_info, medical_history, current_findings
        )
    
    return analysis_results

def main():
    try:
        st.title("🏥 MediAI - Advanced Medical Assistant")
        st.markdown("### AI-Powered Clinical Decision Support & Patient Analysis")
        
        # API Configuration
        api_key = setup_api_configuration()
        
        if not api_key or st.session_state.chatbot is None or not st.session_state.chatbot.client_configured:
            st.warning("""
            🔑 **API Configuration Required**
            
            To use MediAI, you need to configure your OpenAI API key.
            
            Get your API key from [OpenAI Platform](https://platform.openai.com/api-keys)
            
            Add it to Streamlit secrets or enter it manually in the sidebar.
            """)
            
            # Show demo mode or limited functionality
            st.info("🚀 **Demo Mode**: Some features will be available once API key is configured")
            return
        
        chatbot = st.session_state.chatbot
        
        # Sidebar for patient information and file uploads
        with st.sidebar:
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
                "Enter medical history, medications, allergies, and family history", 
                height=120,
                value="Hypertension (10 years), Type 2 Diabetes (5 years), Hyperlipidemia. Medications: Lisinopril 10mg, Metformin 500mg. Allergies: Penicillin. Family History: Father - CAD, Mother - Diabetes",
                key="medical_history"
            )
            
            st.header("📁 Upload Medical Files")
            
            # File upload sections
            tab_lab, tab_image, tab_report = st.tabs(["Lab Reports", "Images", "Documents"])
            
            with tab_lab:
                st.subheader("Upload Laboratory Results")
                lab_files = st.file_uploader("Upload Lab Reports (PDF/DOCX/TXT)", 
                                           type=['pdf', 'docx', 'txt'], 
                                           key="lab_uploader",
                                           accept_multiple_files=True)
                lab_text_input = st.text_area("Or paste lab results directly:", height=100,
                                            value="""CBC: WBC 8.2 (4.0-11.0), RBC 4.5 (4.2-5.8), Hgb 14.2 (12.0-16.0), Hct 42% (36-48), Platelets 250 (150-450)
Chemistry: Glucose 110 (70-100), Creatinine 1.1 (0.6-1.3), BUN 18 (7-20), ALT 25 (7-55), AST 22 (8-48)
Lipid Panel: Total Cholesterol 185 (<200), LDL 110 (<100), HDL 45 (>40), Triglycerides 150 (<150)""",
                                            key="lab_text")
            
            with tab_image:
                st.subheader("Upload Medical Images")
                image_files = st.file_uploader("Upload Medical Images (PNG/JPG/JPEG)", 
                                             type=['png', 'jpg', 'jpeg'], 
                                             key="image_uploader",
                                             accept_multiple_files=True)
                image_description = st.text_area("Describe image findings or upload details:", height=100,
                                               value="Chest X-ray PA view: Mild cardiomegaly noted. Clear lung fields, no focal consolidation. No pleural effusion. Mediastinal contours are normal.",
                                               key="image_desc")
            
            with tab_report:
                st.subheader("Upload Medical Reports")
                report_files = st.file_uploader("Upload Medical Reports (PDF/DOCX/TXT)", 
                                              type=['pdf', 'docx', 'txt'], 
                                              key="report_uploader",
                                              accept_multiple_files=True)
                report_text_input = st.text_area("Or paste report content directly:", height=100,
                                               value="ECG Report: Normal sinus rhythm, rate 72 bpm. Normal axis. No ST-T wave changes. Echocardiogram: Normal LV function, EF 55%. Mild LVH.",
                                               key="report_text")
            
            # Analysis Controls
            st.header("🔍 Analysis Controls")
            
            analyze_col1, analyze_col2 = st.columns(2)
            
            with analyze_col1:
                if st.button("🚀 Analyze All Data", type="primary", use_container_width=True, key="analyze_all"):
                    with st.spinner("🔄 Comprehensive analysis in progress..."):
                        # Collect all lab data
                        lab_data = lab_text_input
                        if lab_files:
                            for file in lab_files:
                                extracted_text = chatbot.extract_text_from_file(file)
                                lab_data += f"\n\n--- From {file.name} ---\n{extracted_text}"
                        
                        # Collect all report data
                        report_data = report_text_input
                        if report_files:
                            for file in report_files:
                                extracted_text = chatbot.extract_text_from_file(file)
                                report_data += f"\n\n--- From {file.name} ---\n{extracted_text}"
                        
                        patient_info = f"ID: {patient_id}, Name: {patient_name}, Age: {patient_age}, Gender: {patient_gender}"
                        
                        analysis_results = perform_comprehensive_analysis(
                            chatbot, patient_info, medical_history, lab_data, image_description, report_data
                        )
                        
                        st.session_state.analysis_results = analysis_results
                        st.success("✅ Analysis complete! Check the dashboard and report tabs.")
            
            with analyze_col2:
                if st.button("🔄 Clear All Results", use_container_width=True, key="clear_results"):
                    st.session_state.analysis_results = {}
                    st.session_state.messages = []
                    st.session_state.uploaded_files = []
                    st.rerun()

        # MAIN CONTENT AREA
        st.header("💬 Dr. MedAI - Medical Chat Assistant")
        st.markdown("Chat with your AI medical assistant for clinical decision support")
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask about patient analysis, differential diagnosis, or medical queries..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate AI response
            with st.chat_message("assistant"):
                with st.spinner("Dr. MedAI is analyzing..."):
                    ai_response = chatbot.chat_response(st.session_state.messages)
                    st.markdown(ai_response)
                    
                    # Add AI response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": ai_response})

        # Tabs for other features
        tab1, tab2, tab3 = st.tabs(["📊 Analysis Dashboard", "📈 Visual Analytics", "📋 Medical Report"])
        
        with tab1:
            display_analysis_dashboard()
        
        with tab2:
            create_medical_visualizations()
        
        with tab3:
            st.header("📋 Comprehensive Medical Report")
            
            if st.session_state.analysis_results:
                # Generate professional medical report
                report_content = f"""
# COMPREHENSIVE MEDICAL ANALYSIS REPORT

## Patient Information
- **Patient ID**: {patient_id}
- **Name**: {patient_name}
- **Age**: {patient_age}
- **Gender**: {patient_gender}
- **Report Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Executive Summary
{st.session_state.analysis_results.get('patient_summary', 'N/A').split('**COMPREHENSIVE ASSESSMENT**')[0] if 'patient_summary' in st.session_state.analysis_results else 'No summary available'}

## Detailed Analysis

### Laboratory Findings
{st.session_state.analysis_results.get('lab_analysis', 'No laboratory analysis available')}

### Imaging Analysis
{st.session_state.analysis_results.get('image_analysis', 'No imaging analysis available')}

### Report Analysis
{st.session_state.analysis_results.get('report_analysis', 'No report analysis available')}

## Comprehensive Patient Summary
{st.session_state.analysis_results.get('patient_summary', 'No patient summary available')}

## Recommendations & Action Plan
Based on the comprehensive analysis, the following actions are recommended:

1. **Immediate Actions**: Review critical findings with appropriate specialists
2. **Short-term Plan**: Implement recommended monitoring and treatments
3. **Long-term Management**: Establish follow-up schedule and preventive measures

---
*Generated by MediAI Clinical Assistant on {datetime.now().strftime('%Y-%m-%d at %H:%M')}*
*This report should be reviewed by a qualified healthcare professional before clinical decision-making.*
"""
                
                st.markdown(report_content)
                
                # Enhanced download options
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        label="📥 Download Full Report (MD)",
                        data=report_content,
                        file_name=f"medical_report_{patient_id}_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                        mime="text/markdown",
                        use_container_width=True
                    )
                with col2:
                    st.download_button(
                        label="📥 Download Summary (TXT)",
                        data=report_content[:2000] + "...",
                        file_name=f"medical_summary_{patient_id}_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
            else:
                st.info("""
                📝 **No Report Generated Yet**
                
                To generate a comprehensive medical report:
                1. Upload patient data (lab reports, images, documents)
                2. Fill in patient information and medical history
                3. Click 'Analyze All Data' in the sidebar
                4. View your comprehensive report here
                """)
    
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        st.error(f"Application error: {str(e)}")
        st.info("Please check the logs for more details. If this persists, try refreshing the page.")

if __name__ == "__main__":
    main()
