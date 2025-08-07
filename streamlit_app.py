import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Employee Attrition Predictor",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .prediction-high {
        background-color: #ffebee;
        border-left-color: #f44336;
    }
    .prediction-medium {
        background-color: #fff3e0;
        border-left-color: #ff9800;
    }
    .prediction-low {
        background-color: #e8f5e8;
        border-left-color: #4caf50;
    }
    .sidebar-info {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_components():
    """Load all model components"""
    try:
        model = joblib.load('models/employee_attrition_model.joblib')
        scaler = joblib.load('models/scaler.joblib')
        label_encoders = joblib.load('models/label_encoders.joblib')
        model_summary = joblib.load('models/model_summary.joblib')
        return model, scaler, label_encoders, model_summary
    except FileNotFoundError as e:
        st.error(f"Model files not found: {e}")
        st.info("Please ensure all model files are in the 'models' directory")
        return None, None, None, None

@st.cache_data
def load_data():
    """Load the HR dataset"""
    try:
        df = pd.read_csv('data/HR_comma_sep.csv')
        return df
    except FileNotFoundError:
        st.error("HR dataset not found. Please ensure 'HR_comma_sep.csv' is in the 'data' directory")
        return None

def predict_attrition(model, scaler, label_encoders, employee_data):
    """Make prediction for employee attrition"""
    # Create DataFrame from input
    df = pd.DataFrame([employee_data])
    
    # Apply label encoding
    for col, encoder in label_encoders.items():
        if col in df.columns:
            try:
                df[col] = encoder.transform(df[col])
            except ValueError:
                # Handle unseen categories
                df[col] = 0
    
    # Make prediction
    try:
        prediction = model.predict(df)
        probability = model.predict_proba(df)[:, 1]
        return prediction[0], probability[0]
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None

def create_gauge_chart(probability):
    """Create a gauge chart for attrition probability"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Attrition Risk (%)"},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': "lightgreen"},
                {'range': [30, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))
    fig.update_layout(height=300)
    return fig

def create_feature_importance_chart(model_summary):
    """Create feature importance chart"""
    if model_summary and 'feature_importance' in model_summary and model_summary['feature_importance']:
        importance_df = pd.DataFrame(model_summary['feature_importance'])
        importance_df = importance_df.sort_values('importance', ascending=True).tail(10)
        
        fig = px.bar(
            importance_df, 
            x='importance', 
            y='feature',
            orientation='h',
            title="Top 10 Feature Importances",
            labels={'importance': 'Importance Score', 'feature': 'Features'}
        )
        fig.update_layout(height=500)
        return fig
    return None

def create_department_analysis(df):
    """Create department-wise attrition analysis"""
    dept_analysis = df.groupby('Department').agg({
        'left': ['count', 'sum']
    }).round(2)
    dept_analysis.columns = ['Total_Employees', 'Employees_Left']
    dept_analysis['Attrition_Rate'] = (dept_analysis['Employees_Left'] / dept_analysis['Total_Employees'] * 100).round(2)
    dept_analysis = dept_analysis.reset_index()
    
    fig = px.bar(
        dept_analysis,
        x='Department',
        y='Attrition_Rate',
        title="Department-wise Attrition Rate",
        labels={'Attrition_Rate': 'Attrition Rate (%)'},
        color='Attrition_Rate',
        color_continuous_scale='Reds'
    )
    fig.update_layout(height=400)
    return fig

def create_satisfaction_distribution(df):
    """Create satisfaction level vs attrition distribution"""
    fig = go.Figure()
    
    # Employees who stayed
    fig.add_trace(go.Histogram(
        x=df[df['left']==0]['satisfaction_level'],
        name='Stayed',
        opacity=0.7,
        nbinsx=20
    ))
    
    # Employees who left
    fig.add_trace(go.Histogram(
        x=df[df['left']==1]['satisfaction_level'],
        name='Left',
        opacity=0.7,
        nbinsx=20
    ))
    
    fig.update_layout(
        title="Satisfaction Level Distribution",
        xaxis_title="Satisfaction Level",
        yaxis_title="Number of Employees",
        barmode='overlay',
        height=400
    )
    return fig

def main():
    # Load model components
    model, scaler, label_encoders, model_summary = load_model_components()
    df = load_data()
    
    if model is None:
        st.stop()
    
    # Main title
    st.markdown('<h1 class="main-header">👥 Employee Attrition Predictor</h1>', unsafe_allow_html=True)
    
    # Sidebar for navigation
    st.sidebar.markdown('<div class="sidebar-info"><h3>🎯 Navigation</h3><p>Use this panel to navigate between different sections of the application.</p></div>', unsafe_allow_html=True)
    
    page = st.sidebar.selectbox(
        "Choose a section:",
        ["🔮 Prediction", "📊 Data Analysis", "📈 Model Insights", "📋 Bulk Prediction"]
    )
    
    if page == "🔮 Prediction":
        st.markdown('<h2 class="sub-header">Individual Employee Prediction</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### Enter Employee Information")
            
            # Create input form
            with st.form("employee_form"):
                col_a, col_b = st.columns(2)
                
                with col_a:
                    satisfaction_level = st.slider(
                        "Satisfaction Level", 
                        min_value=0.0, 
                        max_value=1.0, 
                        value=0.7, 
                        step=0.01,
                        help="Employee satisfaction level (0-1 scale)"
                    )
                    
                    last_evaluation = st.slider(
                        "Last Evaluation Score", 
                        min_value=0.0, 
                        max_value=1.0, 
                        value=0.8, 
                        step=0.01,
                        help="Last performance evaluation score (0-1 scale)"
                    )
                    
                    number_project = st.selectbox(
                        "Number of Projects", 
                        options=[2, 3, 4, 5, 6, 7],
                        index=2,
                        help="Number of projects assigned to employee"
                    )
                    
                    average_monthly_hours = st.number_input(
                        "Average Monthly Hours", 
                        min_value=80, 
                        max_value=350, 
                        value=200,
                        help="Average number of hours worked per month"
                    )
                
                with col_b:
                    time_spend_company = st.selectbox(
                        "Years in Company", 
                        options=list(range(2, 11)),
                        index=2,
                        help="Number of years spent in the company"
                    )
                    
                    work_accident = st.selectbox(
                        "Work Accident", 
                        options=[0, 1],
                        format_func=lambda x: "No" if x == 0 else "Yes",
                        help="Whether employee had a work accident"
                    )
                    
                    promotion_last_5years = st.selectbox(
                        "Promotion in Last 5 Years", 
                        options=[0, 1],
                        format_func=lambda x: "No" if x == 0 else "Yes",
                        help="Whether employee got promoted in last 5 years"
                    )
                    
                    department = st.selectbox(
                        "Department", 
                        options=['sales', 'technical', 'support', 'IT', 'marketing', 
                                'product_mng', 'RandD', 'accounting', 'hr', 'management'],
                        help="Employee's department"
                    )
                    
                    salary = st.selectbox(
                        "Salary Level", 
                        options=['low', 'medium', 'high'],
                        index=1,
                        help="Employee's salary level"
                    )
                
                submitted = st.form_submit_button("🔮 Predict Attrition", use_container_width=True)
        
        with col2:
            st.markdown("### Prediction Result")
            
            if submitted:
                employee_data = {
                    'satisfaction_level': satisfaction_level,
                    'last_evaluation': last_evaluation,
                    'number_project': number_project,
                    'average_montly_hours': average_monthly_hours,
                    'time_spend_company': time_spend_company,
                    'Work_accident': work_accident,
                    'promotion_last_5years': promotion_last_5years,
                    'Department': department,
                    'salary': salary
                }
                
                prediction, probability = predict_attrition(model, scaler, label_encoders, employee_data)
                
                if prediction is not None:
                    # Display gauge chart
                    gauge_fig = create_gauge_chart(probability)
                    st.plotly_chart(gauge_fig, use_container_width=True)
                    
                    # Risk assessment
                    if probability >= 0.7:
                        risk_level = "HIGH RISK"
                        risk_class = "prediction-high"
                        risk_emoji = "🚨"
                    elif probability >= 0.3:
                        risk_level = "MEDIUM RISK"
                        risk_class = "prediction-medium"
                        risk_emoji = "⚠️"
                    else:
                        risk_level = "LOW RISK"
                        risk_class = "prediction-low"
                        risk_emoji = "✅"
                    
                    st.markdown(f"""
                    <div class="metric-card {risk_class}">
                        <h3>{risk_emoji} {risk_level}</h3>
                        <p><strong>Prediction:</strong> {'Will Leave' if prediction == 1 else 'Will Stay'}</p>
                        <p><strong>Probability:</strong> {probability:.1%}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Recommendations
                    st.markdown("### 💡 Recommendations")
                    if probability >= 0.7:
                        st.error("""
                        **Immediate Action Required:**
                        - Schedule urgent one-on-one meeting
                        - Review workload and working hours
                        - Discuss career development opportunities
                        - Consider salary adjustment or benefits improvement
                        """)
                    elif probability >= 0.3:
                        st.warning("""
                        **Monitor Closely:**
                        - Regular check-ins with manager
                        - Provide additional support or training
                        - Acknowledge achievements and contributions
                        """)
                    else:
                        st.success("""
                        **Employee Likely to Stay:**
                        - Continue current engagement strategies
                        - Maintain regular feedback cycles
                        - Consider for leadership opportunities
                        """)
    
    elif page == "📊 Data Analysis":
        if df is not None:
            st.markdown('<h2 class="sub-header">Dataset Analysis</h2>', unsafe_allow_html=True)
            
            # Dataset overview
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Employees", len(df))
            with col2:
                st.metric("Employees Left", df['left'].sum())
            with col3:
                st.metric("Attrition Rate", f"{df['left'].mean():.1%}")
            with col4:
                st.metric("Departments", df['Department'].nunique())
            
            # Charts
            col1, col2 = st.columns(2)
            
            with col1:
                dept_fig = create_department_analysis(df)
                st.plotly_chart(dept_fig, use_container_width=True)
            
            with col2:
                satisfaction_fig = create_satisfaction_distribution(df)
                st.plotly_chart(satisfaction_fig, use_container_width=True)
            
            # Additional insights
            st.markdown("### 📋 Key Insights")
            
            insights_col1, insights_col2 = st.columns(2)
            
            with insights_col1:
                st.markdown("**Satisfaction Analysis:**")
                left_satisfaction = df[df['left']==1]['satisfaction_level'].mean()
                stayed_satisfaction = df[df['left']==0]['satisfaction_level'].mean()
                st.write(f"- Employees who left: {left_satisfaction:.2f}")
                st.write(f"- Employees who stayed: {stayed_satisfaction:.2f}")
                
                st.markdown("**Working Hours Analysis:**")
                left_hours = df[df['left']==1]['average_montly_hours'].mean()
                stayed_hours = df[df['left']==0]['average_montly_hours'].mean()
                st.write(f"- Employees who left: {left_hours:.0f} hours")
                st.write(f"- Employees who stayed: {stayed_hours:.0f} hours")
            
            with insights_col2:
                st.markdown("**Salary Analysis:**")
                salary_analysis = df.groupby('salary')['left'].agg(['count', 'sum'])
                salary_analysis['rate'] = (salary_analysis['sum'] / salary_analysis['count'] * 100).round(1)
                for salary_level in ['low', 'medium', 'high']:
                    if salary_level in salary_analysis.index:
                        rate = salary_analysis.loc[salary_level, 'rate']
                        st.write(f"- {salary_level.title()}: {rate}%")
                
                st.markdown("**Promotion Impact:**")
                promo_analysis = df.groupby('promotion_last_5years')['left'].mean() * 100
                st.write(f"- No promotion: {promo_analysis[0]:.1f}%")
                st.write(f"- Promoted: {promo_analysis[1]:.1f}%")
    
    elif page == "📈 Model Insights":
        st.markdown('<h2 class="sub-header">Model Performance & Insights</h2>', unsafe_allow_html=True)
        
        if model_summary:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("### 🎯 Model Performance")
                st.metric("Model Type", model_summary.get('model_type', 'Unknown'))
                st.metric("AUC Score", f"{model_summary.get('auc_score', 0):.3f}")
                st.metric("Number of Features", len(model_summary.get('features', [])))
            
            with col2:
                # Feature importance chart
                importance_fig = create_feature_importance_chart(model_summary)
                if importance_fig:
                    st.plotly_chart(importance_fig, use_container_width=True)
            
            # Model features
            st.markdown("### 📋 Model Features")
            if 'features' in model_summary:
                features_df = pd.DataFrame({
                    'Feature': model_summary['features'],
                    'Description': [
                        'Employee satisfaction level (0-1)',
                        'Last performance evaluation (0-1)', 
                        'Number of projects assigned',
                        'Average monthly working hours',
                        'Years spent in company',
                        'Work accident history (0/1)',
                        'Promotion in last 5 years (0/1)',
                        'Department (encoded)',
                        'Salary level (encoded)'
                    ]
                })
                st.dataframe(features_df, use_container_width=True)
    
    elif page == "📋 Bulk Prediction":
        st.markdown('<h2 class="sub-header">Bulk Employee Prediction</h2>', unsafe_allow_html=True)
        
        st.markdown("### Upload Employee Data")
        st.info("Upload a CSV file with employee data to predict attrition for multiple employees at once.")
        
        uploaded_file = st.file_uploader(
            "Choose a CSV file", 
            type="csv",
            help="CSV should contain columns: satisfaction_level, last_evaluation, number_project, average_montly_hours, time_spend_company, Work_accident, promotion_last_5years, Department, salary"
        )
        
        if uploaded_file is not None:
            try:
                upload_df = pd.read_csv(uploaded_file)
                st.success(f"File uploaded successfully! {len(upload_df)} employees found.")
                
                # Show preview
                st.markdown("### Data Preview")
                st.dataframe(upload_df.head(), use_container_width=True)
                
                if st.button("🔮 Predict for All Employees", use_container_width=True):
                    # Make predictions for all employees
                    predictions = []
                    probabilities = []
                    
                    progress_bar = st.progress(0)
                    for i, row in upload_df.iterrows():
                        employee_data = row.to_dict()
                        pred, prob = predict_attrition(model, scaler, label_encoders, employee_data)
                        predictions.append(pred)
                        probabilities.append(prob)
                        progress_bar.progress((i + 1) / len(upload_df))
                    
                    # Add predictions to dataframe
                    upload_df['Prediction'] = predictions
                    upload_df['Attrition_Probability'] = probabilities
                    upload_df['Risk_Level'] = upload_df['Attrition_Probability'].apply(
                        lambda x: 'High' if x >= 0.7 else 'Medium' if x >= 0.3 else 'Low'
                    )
                    
                    # Display results
                    st.markdown("### Prediction Results")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        high_risk = len(upload_df[upload_df['Risk_Level'] == 'High'])
                        st.metric("High Risk", high_risk, delta_color="inverse")
                    with col2:
                        medium_risk = len(upload_df[upload_df['Risk_Level'] == 'Medium'])
                        st.metric("Medium Risk", medium_risk)
                    with col3:
                        low_risk = len(upload_df[upload_df['Risk_Level'] == 'Low'])
                        st.metric("Low Risk", low_risk)
                    
                    # Show results table
                    st.dataframe(
                        upload_df[['Prediction', 'Attrition_Probability', 'Risk_Level']].round(3),
                        use_container_width=True
                    )
                    
                    # Download results
                    csv = upload_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results",
                        data=csv,
                        file_name="attrition_predictions.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                    
            except Exception as e:
                st.error(f"Error processing file: {e}")
    
    # Footer
    st.markdown("---")
    st.markdown(
        '<p style="text-align: center; color: #7f8c8d;">Built with ❤️ using Streamlit | Employee Attrition Predictor v1.0</p>',
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()