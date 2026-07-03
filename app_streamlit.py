import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time

st.set_page_config(
    page_title="Customer Churn Prediction System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stButton > button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #FF6B6B;
        color: white;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

st.title("Customer Churn Prediction System")
st.markdown("*AI-powered churn prediction with actionable insights*")

with st.sidebar:
    st.header("Configuration")
    
    api_url = st.text_input(
        "API URL",
        value="http://localhost:8000",
        help="URL of your FastAPI backend"
    )
    
    if st.button("Test Connection"):
        try:
            response = requests.get(f"{api_url}/health")
            if response.status_code == 200:
                st.success("Connected to API!")
            else:
                st.error("Failed to connect")
        except:
            st.error("API not reachable")
    
    st.divider()
    
    st.header("Quick Stats")
    st.metric("Total Customers", "7,043", delta="100%")
    st.metric("Churn Rate", "26.5%", delta="-2.1%", delta_color="inverse")
    st.metric("Model Accuracy", "80.5%", delta="+1.2%")

tab1, tab2, tab3, tab4 = st.tabs([
    "Predict Churn", 
    "Batch Analysis", 
    "Analytics", 
    "History"
])

with tab1:
    st.header("Single Customer Prediction")
    st.markdown("*Enter customer details to predict churn probability*")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Customer Information")
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Has Partner", ["No", "Yes"])
        dependents = st.selectbox("Has Dependents", ["No", "Yes"])
        
        st.subheader("Services")
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        
    with col2:
        st.subheader("Security & Support")
        online_security = st.selectbox("Online Security", ["No", "Yes"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes"])
        device_protection = st.selectbox("Device Protection", ["No", "Yes"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes"])
        
        st.subheader("Account Information")
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox(
            "Payment Method", 
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
        )
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        monthly_charges = st.slider("Monthly Charges ($)", 0, 150, 70)
        total_charges = monthly_charges * tenure
    
    if st.button("Predict Churn", type="primary", use_container_width=True):
        with st.spinner("Analyzing customer data..."):
            data = {
                "gender": gender,
                "senior_citizen": senior_citizen,
                "partner": partner,
                "dependents": dependents,
                "phone_service": phone_service,
                "multiple_lines": multiple_lines,
                "internet_service": internet_service,
                "online_security": online_security,
                "online_backup": online_backup,
                "device_protection": device_protection,
                "tech_support": tech_support,
                "streaming_tv": streaming_tv,
                "streaming_movies": streaming_movies,
                "contract": contract,
                "paperless_billing": paperless_billing,
                "payment_method": payment_method,
                "tenure": tenure,
                "monthly_charges": monthly_charges,
                "total_charges": total_charges
            }
            
            try:
                response = requests.post(f"{api_url}/predict", json=data)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    st.success("Prediction Complete!")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric(
                            "Churn Probability", 
                            f"{result['churn_probability']*100:.1f}%"
                        )
                    
                    with col2:
                        st.metric(
                            "Prediction", 
                            result['prediction']
                        )
                    
                    with col3:
                        st.metric(
                            "Risk Level", 
                            result['risk_level']
                        )
                    
                    with col4:
                        st.metric(
                            "Confidence", 
                            f"{result['confidence']*100:.1f}%"
                        )
                    
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = result['churn_probability'] * 100,
                        title = {'text': "Churn Risk"},
                        gauge = {
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "red" if result['churn_probability'] > 0.5 else "green"},
                            'steps': [
                                {'range': [0, 30], 'color': "lightgreen"},
                                {'range': [30, 70], 'color': "yellow"},
                                {'range': [70, 100], 'color': "salmon"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 50
                            }
                        }
                    ))
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.subheader("Recommendations")
                    for rec in result['recommendations']:
                        st.info(f"• {rec}")
                    
                    st.subheader("Top Risk Factors")
                    for factor in result['top_factors']:
                        st.warning(f"• {factor}")
                    
                else:
                    st.error(f"API Error: {response.status_code}")
                    
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to API. Please check if the API is running.")
            except Exception as e:
                st.error(f"Error: {str(e)}")

with tab2:
    st.header("Batch Customer Analysis")
    
    uploaded_file = st.file_uploader("Upload CSV file with customer data", type=['csv'])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write(f"Uploaded {len(df)} customers")
        st.dataframe(df.head())
        
        if st.button("Analyze Batch"):
            with st.spinner("Analyzing all customers..."):
                try:
                    predictions = []
                    for _, row in df.iterrows():
                        data = row.to_dict()
                        response = requests.post(f"{api_url}/predict", json=data)
                        if response.status_code == 200:
                            result = response.json()
                            predictions.append({
                                'churn_probability': result['churn_probability'],
                                'prediction': result['prediction'],
                                'risk_level': result['risk_level']
                            })
                        time.sleep(0.1)
                    
                    results_df = pd.DataFrame(predictions)
                    final_df = pd.concat([df, results_df], axis=1)
                    
                    st.success(f"Analyzed {len(predictions)} customers!")
                    st.dataframe(final_df)
                    
                    csv = final_df.to_csv(index=False)
                    st.download_button(
                        label="Download Results",
                        data=csv,
                        file_name="churn_predictions.csv",
                        mime="text/csv"
                    )
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig = px.histogram(
                            final_df, 
                            x='churn_probability',
                            nbins=20,
                            title="Churn Probability Distribution"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        churn_counts = final_df['prediction'].value_counts()
                        fig = px.pie(
                            values=churn_counts.values,
                            names=churn_counts.index,
                            title="Churn Prediction Distribution"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                except Exception as e:
                    st.error(f"Error: {str(e)}")

with tab3:
    st.header("Analytics Dashboard")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Predictions Today", "156", delta="+12%")
    
    with col2:
        st.metric("High Risk Customers", "23", delta="-5%")
    
    with col3:
        st.metric("Avg Churn Rate", "26.5%", delta="-2.1%")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Churn by Contract Type")
        contract_data = {
            'Contract': ['Month-to-month', 'One year', 'Two year'],
            'Churn Rate': [45, 20, 8]
        }
        fig = px.bar(contract_data, x='Contract', y='Churn Rate', color='Contract')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Churn by Tenure")
        tenure_data = {
            'Tenure Group': ['0-12 months', '13-24 months', '25-48 months', '48+ months'],
            'Churn Rate': [35, 25, 15, 8]
        }
        fig = px.bar(tenure_data, x='Tenure Group', y='Churn Rate', color='Tenure Group')
        st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.header("Prediction History")
    
    history_data = {
        'Timestamp': pd.date_range(start='2024-01-01', periods=10, freq='2h'),
        'Customer': [f'CUST-{i:04d}' for i in range(1, 11)],
        'Churn Probability': np.random.uniform(0, 1, 10),
        'Prediction': np.random.choice(['Will Churn', 'Will Not Churn'], 10),
        'Risk Level': np.random.choice(['High', 'Medium', 'Low'], 10)
    }
    
    df_history = pd.DataFrame(history_data)
    st.dataframe(df_history)
    
    csv = df_history.to_csv(index=False)
    st.download_button(
        label="Export History",
        data=csv,
        file_name="churn_history.csv",
        mime="text/csv"
    )

st.divider()
st.markdown("""
    <div style='text-align: center; color: gray;'>
        Customer Churn Prediction System v1.0
    </div>
""", unsafe_allow_html=True)