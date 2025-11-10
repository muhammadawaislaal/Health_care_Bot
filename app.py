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
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure the page
st.set_page_config(
    page_title="MediAI - Doctor's Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'patient_data' not in st.session_state:
    st.session_state.patient_data = {}
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {}

class MedicalChatbot:
    def __init__(self):
        # Initialize OpenAI client (you can replace with other APIs)
        self.api_key = os.getenv('OPENAI_API_KEY')
        if self.api_key:
            self.client = openai.OpenAI(api_key=self.api_key)
        else:
            st.warning("Please set OPENAI_API_KEY in your environment variables")
    
    def analyze_lab_results(self, lab_data):
        """Analyze laboratory test results"""
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
                model="gpt-4",
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
        try:
            # For demonstration, using text description analysis
            # In production, you would use specialized medical imaging APIs
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
                model="gpt-4",
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
                model="gpt-4",
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

def display_analysis_dashboard():
    """Display analysis results in a dashboard format"""
    st.header("📊 Analysis Dashboard")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'lab_analysis' in st.session_state.analysis_results:
            st.subheader("Lab Results Analysis")
            st.write(st.session_state.analysis_results['lab_analysis'])
    
    with col2:
        if 'image_analysis' in st.session_state.analysis_results:
            st.subheader("Image Analysis")
            st.write(st.session_state.analysis_results['image_analysis'])
    
    if 'patient_summary' in st.session_state.analysis_results:
        st.subheader("Patient Summary")
        st.write(st.session_state.analysis_results['patient_summary'])

def create_sample_lab_chart():
    """Create sample laboratory results visualization"""
    # Sample lab data
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
        xaxis=dict(
            tickvals=df.index,
            ticktext=df['Test']
        ),
        yaxis_title='Value',
        showlegend=True
    )
    
    return fig

def main():
    st.title("🏥 MediAI - Doctor's Assistant")
    st.markdown("AI-powered medical analysis and patient report interpretation")
    
    # Initialize chatbot
    chatbot = MedicalChatbot()
    
    # Sidebar for patient information
    with st.sidebar:
        st.header("Patient Information")
        
        patient_id = st.text_input("Patient ID")
        patient_name = st.text_input("Patient Name")
        patient_age = st.number_input("Age", min_value=0, max_value=120, value=45)
        patient_gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        
        st.header("Medical History")
        medical_history = st.text_area("Enter medical history")
        
        st.header("Upload Reports")
        
        # Lab results upload
        lab_results = st.text_area("Paste Lab Results (or upload CSV)")
        lab_file = st.file_uploader("Upload Lab Results CSV", type=['csv'])
        
        # Medical image upload
        image_file = st.file_uploader("Upload Medical Image", type=['png', 'jpg', 'jpeg'])
        image_description = st.text_area("Image Description/Finding")
        
        if st.button("Analyze Patient Data"):
            with st.spinner("Analyzing patient data..."):
                # Analyze lab results
                if lab_results or lab_file:
                    analysis = chatbot.analyze_lab_results(lab_results)
                    st.session_state.analysis_results['lab_analysis'] = analysis
                
                # Analyze medical image
                if image_description:
                    analysis = chatbot.analyze_medical_image(image_description, "X-Ray")
                    st.session_state.analysis_results['image_analysis'] = analysis
                
                # Generate patient summary
                patient_info = f"Name: {patient_name}, Age: {patient_age}, Gender: {patient_gender}"
                summary = chatbot.generate_patient_summary(patient_info, medical_history, 
                                                         f"Lab: {lab_results}, Image: {image_description}")
                st.session_state.analysis_results['patient_summary'] = summary
                
                st.success("Analysis complete!")
    
    # Main chat interface
    st.header("💬 Medical Chat Assistant")
    
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
            with st.spinner("Thinking..."):
                try:
                    # Use the chatbot to generate response
                    response = chatbot.client.chat.completions.create(
                        model="gpt-4",
                        messages=[
                            {"role": "system", "content": """You are a medical assistant for doctors. 
                            Provide accurate, evidence-based medical information. 
                            Always recommend consulting with specialists for complex cases.
                            Be precise and professional in your responses."""}
                        ] + [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
                        temperature=0.3,
                        max_tokens=1000
                    )
                    
                    ai_response = response.choices[0].message.content
                    st.markdown(ai_response)
                    
                    # Add AI response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": ai_response})
                    
                except Exception as e:
                    error_msg = f"Sorry, I encountered an error: {str(e)}"
                    st.markdown(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
    
    # Display analysis dashboard
    if st.session_state.analysis_results:
        display_analysis_dashboard()
    
    # Sample visualizations
    st.header("📈 Medical Visualizations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Laboratory Trends")
        lab_chart = create_sample_lab_chart()
        st.plotly_chart(lab_chart, use_container_width=True)
    
    with col2:
        st.subheader("Vital Signs Monitor")
        # Sample vital signs data
        time_points = ['08:00', '12:00', '16:00', '20:00', '00:00']
        heart_rate = [72, 75, 80, 78, 70]
        blood_pressure_sys = [120, 118, 122, 119, 121]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=time_points, y=heart_rate, mode='lines+markers', name='Heart Rate'))
        fig.add_trace(go.Scatter(x=time_points, y=blood_pressure_sys, mode='lines+markers', name='BP Systolic'))
        
        fig.update_layout(
            title='Vital Signs Monitoring',
            xaxis_title='Time',
            yaxis_title='Value'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Quick actions
    st.header("⚡ Quick Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Generate Report Summary"):
            st.info("Report summary generation triggered")
    
    with col2:
        if st.button("Suggest Follow-up Tests"):
            st.info("Follow-up test suggestions generated")
    
    with col3:
        if st.button("Medication Review"):
            st.info("Medication review initiated")

if __name__ == "__main__":
    main()
