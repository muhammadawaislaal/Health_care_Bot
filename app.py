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
if 'api_key_configured' not in st.session_state:
    st.session_state.api_key_configured = False

class MedicalChatbot:
    def __init__(self, api_key=None):
        if api_key:
            self.client = openai.OpenAI(api_key=api_key)
            self.api_key = api_key
        else:
            self.client = None
            self.api_key = None
    
    def set_api_key(self, api_key):
        """Set API key after initialization"""
        self.client = openai.OpenAI(api_key=api_key)
        self.api_key = api_key
    
    def analyze_lab_results(self, lab_data):
        """Analyze laboratory test results"""
        if not self.client:
            return "Please configure API key first"
        
        try:
            prompt = f"""
            Analyze these laboratory results and provide a medical assessment:
            
            {lab_data}
            
            Please provide:
            1. Abnormal values and their significance
            2. Potential conditions to consider
            3. Recommended follow-up tests
            4. Urgency level (Low/Medium/High)
            
            Format the response in a structured way for medical professionals.
            """
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",  # Using 3.5 for cost efficiency
                messages=[
                    {"role": "system", "content": "You are a medical expert analyzing laboratory results. Provide accurate, evidence-based analysis."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1500
            )
            
            return response.choices[0].message.content
        except Exception as e:
            return f"Error analyzing lab results: {str(e)}"
    
    def analyze_medical_image(self, image_data, image_type):
        """Analyze medical images"""
        if not self.client:
            return "Please configure API key first"
        
        try:
            prompt = f"""
            Analyze this {image_type} medical image description and provide insights:
            
            {image_data}
            
            Please provide:
            1. Key findings
            2. Potential abnormalities
            3. Differential diagnosis
            4. Recommended next steps
            """
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a radiologist analyzing medical images. Provide professional medical insights."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1500
            )
            
            return response.choices[0].message.content
        except Exception as e:
            return f"Error analyzing image: {str(e)}"
    
    def generate_patient_summary(self, patient_info, medical_history, current_findings):
        """Generate comprehensive patient summary"""
        if not self.client:
            return "Please configure API key first"
        
        try:
            prompt = f"""
            Generate a comprehensive patient summary:
            
            Patient Information: {patient_info}
            Medical History: {medical_history}
            Current Findings: {current_findings}
            
            Please provide:
            1. Summary of current condition
            2. Risk factors
            3. Treatment recommendations
            4. Monitoring plan
            5. Patient education points
            """
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an experienced physician creating patient summaries."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=2000
            )
            
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating summary: {str(e)}"
    
    def chat_response(self, messages):
        """Generate chat response"""
        if not self.client:
            return "Please configure API key first to use the chat feature."
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": """You are a medical assistant for doctors. 
                    Provide accurate, evidence-based medical information. 
                    Always recommend consulting with specialists for complex cases.
                    Be precise and professional in your responses."""}
                ] + messages,
                temperature=0.3,
                max_tokens=1000
            )
            
            return response.choices[0].message.content
        except Exception as e:
            return f"Error in chat: {str(e)}"

def setup_api_configuration():
    """API configuration section"""
    st.sidebar.header("🔑 API Configuration")
    
    # Option 1: Streamlit secrets (recommended for cloud)
    if 'OPENAI_API_KEY' in st.secrets:
        api_key = st.secrets['OPENAI_API_KEY']
        st.sidebar.success("API key loaded from Streamlit secrets!")
        return api_key
    
    # Option 2: Manual input
    st.sidebar.info("Configure your OpenAI API key")
    api_key = st.sidebar.text_input("Enter OpenAI API Key:", type="password")
    
    if api_key:
        if st.sidebar.button("Save API Key"):
            st.session_state.api_key_configured = True
            st.sidebar.success("API key configured!")
            return api_key
    
    return None

def display_analysis_dashboard():
    """Display analysis results in a dashboard format"""
    st.header("📊 Analysis Dashboard")
    
    if not st.session_state.analysis_results:
        st.info("No analysis results yet. Upload patient data to see analysis here.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'lab_analysis' in st.session_state.analysis_results:
            with st.expander("Lab Results Analysis", expanded=True):
                st.write(st.session_state.analysis_results['lab_analysis'])
    
    with col2:
        if 'image_analysis' in st.session_state.analysis_results:
            with st.expander("Image Analysis", expanded=True):
                st.write(st.session_state.analysis_results['image_analysis'])
    
    if 'patient_summary' in st.session_state.analysis_results:
        with st.expander("Patient Summary", expanded=True):
            st.write(st.session_state.analysis_results['patient_summary'])

def create_sample_lab_chart():
    """Create sample laboratory results visualization"""
    lab_data = {
        'Test': ['WBC', 'RBC', 'Hemoglobin', 'Platelets', 'Glucose', 'Creatinine'],
        'Value': [8.2, 4.5, 14.2, 250, 110, 1.1],
        'Normal Range Low': [4.0, 4.2, 12.0, 150, 70, 0.6],
        'Normal Range High': [11.0, 5.8, 16.0, 450, 100, 1.3],
        'Unit': ['10^3/μL', '10^6/μL', 'g/dL', '10^3/μL', 'mg/dL', 'mg/dL']
    }
    
    df = pd.DataFrame(lab_data)
    
    fig = go.Figure()
    
    # Add normal range
    for i, row in df.iterrows():
        fig.add_trace(go.Scatter(
            x=[i, i],
            y=[row['Normal Range Low'], row['Normal Range High']],
            mode='lines',
            line=dict(width=15, color='lightblue'),
            name='Normal Range',
            showlegend=(i==0)
        ))
    
    # Add actual values
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Value'],
        mode='markers+text',
        marker=dict(size=15, color='red'),
        text=df['Value'],
        textposition='top center',
        name='Patient Value'
    ))
    
    fig.update_layout(
        title='Laboratory Results Overview',
        xaxis=dict(tickvals=df.index, ticktext=df['Test']),
        yaxis_title='Value',
        showlegend=True
    )
    
    return fig

def main():
    st.title("🏥 MediAI - Doctor's Assistant")
    st.markdown("AI-powered medical analysis and patient report interpretation")
    
    # API Configuration
    api_key = setup_api_configuration()
    
    if not api_key:
        st.warning("""
        🔑 **API Configuration Required**
        
        To use MediAI, you need to configure your OpenAI API key:
        
        1. **Recommended**: Add `OPENAI_API_KEY` to Streamlit secrets (see deployment guide)
        2. **Alternative**: Enter your API key in the sidebar
        
        Get your API key from [OpenAI Platform](https://platform.openai.com/api-keys)
        """)
        return
    
    # Initialize chatbot with API key
    chatbot = MedicalChatbot(api_key)
    
    # Sidebar for patient information
    with st.sidebar:
        st.header("👤 Patient Information")
        
        patient_id = st.text_input("Patient ID", value="PT-001")
        patient_name = st.text_input("Patient Name", value="John Doe")
        patient_age = st.number_input("Age", min_value=0, max_value=120, value=45)
        patient_gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        
        st.header("📋 Medical History")
        medical_history = st.text_area(
            "Enter medical history", 
            value="Hypertension, Type 2 Diabetes, Hyperlipidemia"
        )
        
        st.header("📊 Upload Reports")
        
        # Lab results section
        st.subheader("Laboratory Results")
        lab_results = st.text_area(
            "Paste Lab Results", 
            height=100,
            value="""CBC: WBC 8.2, RBC 4.5, Hgb 14.2, Hct 42%, Platelets 250
Chemistry: Glucose 110, Creatinine 1.1, BUN 18, ALT 25, AST 22
Lipid Panel: Total Cholesterol 185, LDL 110, HDL 45, Triglycerides 150"""
        )
        
        # Medical image section
        st.subheader("Medical Imaging")
        image_description = st.text_area(
            "Image Findings/Description", 
            height=100,
            value="Chest X-ray: Mild cardiomegaly, clear lung fields, no active disease"
        )
        
        analyze_col1, analyze_col2 = st.columns(2)
        
        with analyze_col1:
            if st.button("🚀 Analyze All", type="primary"):
                with st.spinner("Analyzing patient data..."):
                    # Analyze lab results
                    if lab_results:
                        analysis = chatbot.analyze_lab_results(lab_results)
                        st.session_state.analysis_results['lab_analysis'] = analysis
                    
                    # Analyze medical image
                    if image_description:
                        analysis = chatbot.analyze_medical_image(image_description, "X-Ray")
                        st.session_state.analysis_results['image_analysis'] = analysis
                    
                    # Generate patient summary
                    patient_info = f"Name: {patient_name}, Age: {patient_age}, Gender: {patient_gender}"
                    summary = chatbot.generate_patient_summary(
                        patient_info, medical_history, 
                        f"Lab: {lab_results}, Image: {image_description}"
                    )
                    st.session_state.analysis_results['patient_summary'] = summary
                    
                    st.success("Analysis complete! Check the dashboard.")
        
        with analyze_col2:
            if st.button("Clear Results"):
                st.session_state.analysis_results = {}
                st.session_state.messages = []
                st.rerun()
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["💬 Chat Assistant", "📈 Analytics", "📋 Report"])
    
    with tab1:
        st.header("Medical Chat Assistant")
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask about patient analysis or medical queries..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate AI response
            with st.chat_message("assistant"):
                with st.spinner("Analyzing..."):
                    ai_response = chatbot.chat_response(st.session_state.messages)
                    st.markdown(ai_response)
                    
                    # Add AI response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": ai_response})
    
    with tab2:
        display_analysis_dashboard()
        
        # Visualizations
        st.header("📊 Medical Visualizations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Laboratory Trends")
            lab_chart = create_sample_lab_chart()
            st.plotly_chart(lab_chart, use_container_width=True)
        
        with col2:
            st.subheader("Vital Signs Monitor")
            time_points = ['08:00', '12:00', '16:00', '20:00', '00:00']
            heart_rate = [72, 75, 80, 78, 70]
            blood_pressure_sys = [120, 118, 122, 119, 121]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=time_points, y=heart_rate, mode='lines+markers', name='Heart Rate'))
            fig.add_trace(go.Scatter(x=time_points, y=blood_pressure_sys, mode='lines+markers', name='BP Systolic'))
            
            fig.update_layout(title='Vital Signs Monitoring')
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.header("📋 Medical Report")
        
        if st.session_state.analysis_results:
            report_content = f"""
# Medical Analysis Report

**Patient:** {st.session_state.get('patient_name', 'John Doe')}  
**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}

## Laboratory Analysis
{st.session_state.analysis_results.get('lab_analysis', 'N/A')}

## Imaging Analysis  
{st.session_state.analysis_results.get('image_analysis', 'N/A')}

## Patient Summary
{st.session_state.analysis_results.get('patient_summary', 'N/A')}

---
*Generated by MediAI - AI Medical Assistant*
"""
            
            st.markdown(report_content)
            
            # Download button
            st.download_button(
                label="📥 Download Report",
                data=report_content,
                file_name=f"medical_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.md",
                mime="text/markdown"
            )
        else:
            st.info("Generate analysis first to view the report")

if __name__ == "__main__":
    main()
