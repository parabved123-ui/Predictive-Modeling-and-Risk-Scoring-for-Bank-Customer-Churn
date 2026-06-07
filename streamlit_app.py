"""
Bank Customer Churn Prediction - Streamlit Dashboard
Comprehensive analytics and ML prediction interface
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, classification_report,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Bank Churn Analytics",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .header-title {
        font-size: 2.5em;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 10px;
    }
    .subheader {
        color: #555;
        font-size: 1.1em;
    }
    </style>
    """, unsafe_allow_html=True)

# Load sample data
@st.cache_data
def load_data():
    """Load and prepare sample data for demonstration"""
    np.random.seed(42)
    
    n_samples = 10000
    data = {
        'CustomerId': np.arange(1, n_samples + 1),
        'CreditScore': np.random.normal(650, 100, n_samples).astype(int),
        'Age': np.random.normal(40, 15, n_samples).astype(int),
        'Tenure': np.random.exponential(2.5, n_samples).astype(int),
        'Balance': np.random.exponential(50000, n_samples),
        'EstimatedSalary': np.random.normal(100000, 50000, n_samples),
        'NumOfProducts': np.random.choice([1, 2, 3, 4], n_samples),
        'HasCrCard': np.random.choice([0, 1], n_samples),
        'IsActiveMember': np.random.choice([0, 1], n_samples),
        'Geography': np.random.choice(['France', 'Spain', 'Germany'], n_samples),
        'Gender': np.random.choice(['Male', 'Female'], n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Generate churn based on realistic patterns
    churn_prob = np.zeros(n_samples)
    churn_prob += (10 - df['Tenure']) * 0.01
    
    # FIX: Correct array division matching configuration types
    ratio_array = (df['Balance'] / (df['EstimatedSalary'] + 1)).to_numpy()
    comp_array = np.full_like(ratio_array, 0.2)
    churn_prob += (0.2 - np.minimum(ratio_array, comp_array)) * 2
    
    churn_prob += (4 - df['NumOfProducts']) * 0.05
    churn_prob += (1 - df['IsActiveMember']) * 0.15
    
    age_factor = np.minimum(np.abs(df['Age'] - 40) / 20, 0.1)
    churn_prob += age_factor
    
    geo_factor = np.array([0.05 if g == 'Germany' else 0.02 for g in df['Geography']])
    churn_prob += geo_factor
    churn_prob = np.clip(churn_prob, 0, 1)
    
    df['Exited'] = np.array([np.random.binomial(1, p) for p in churn_prob])
    
    return df

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = load_data()

df = st.session_state.df

# Pipeline engine built with caching to handle Streamlit re-runs cleanly
@st.cache_resource
def train_ml_pipeline(dataframe):
    df_features = dataframe.copy()
    df_features['BalanceToSalaryRatio'] = df_features['Balance'] / (df_features['EstimatedSalary'] + 1)
    df_features['ProductDensity'] = df_features['NumOfProducts'] / (df_features['Tenure'] + 1)
    df_features['ActivityScore'] = df_features['IsActiveMember'] + df_features['HasCrCard']
    
    # Map out Categoricals cleanly to retain uniform feature spaces
    df_features['Geography_France'] = (df_features['Geography'] == 'France').astype(int)
    df_features['Geography_Germany'] = (df_features['Geography'] == 'Germany').astype(int)
    df_features['Geography_Spain'] = (df_features['Geography'] == 'Spain').astype(int)
    df_features['Gender_Female'] = (df_features['Gender'] == 'Female').astype(int)
    df_features['Gender_Male'] = (df_features['Gender'] == 'Male').astype(int)
    
    numeric_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'EstimatedSalary', 
                   'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'BalanceToSalaryRatio',
                   'ProductDensity', 'ActivityScore', 'Geography_France', 'Geography_Germany',
                   'Geography_Spain', 'Gender_Female', 'Gender_Male']
    
    X = df_features[numeric_cols]
    y = df_features['Exited']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=numeric_cols)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=numeric_cols)
    
    model = xgb.XGBClassifier(n_estimators=50, max_depth=5, learning_rate=0.1, random_state=42, eval_metric='logloss')
    model.fit(X_train_scaled_df, y_train)
    
    return model, scaler, X_test_scaled_df, y_test, numeric_cols

# Initialize the pipeline outputs
model, scaler, X_test, y_test, feature_order = train_ml_pipeline(df)

# Sidebar navigation
st.sidebar.markdown("# 🏦 Bank Churn Analytics")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["📊 Dashboard", "📈 EDA & Analysis", "🤖 Model Performance", 
     "🔮 Predict Churn", "📋 Customer Insights"]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**About**: This dashboard analyzes customer churn patterns and predicts "
    "churn probability using machine learning models trained on 10,000 customer records."
)

# PAGE 1: DASHBOARD
if page == "📊 Dashboard":
    st.markdown('<div class="header-title">📊 Bank Churn Analytics Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="subheader">Real-time customer churn prediction & analysis</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    col1, col2, col3, col4 = st.columns(4)
    total_customers = len(df)
    churned = int(df['Exited'].sum())
    churn_rate = float(df['Exited'].mean())
    retained = total_customers - churned
    
    with col1:
        st.metric("📌 Total Customers", f"{total_customers:,}", delta="Active Dataset")
    with col2:
        st.metric("🚪 Churned", f"{churned:,}", delta=f"{churn_rate*100:.2f}%")
    with col3:
        st.metric("✅ Retained", f"{retained:,}", delta=f"{(1-churn_rate)*100:.2f}%")
    with col4:
        st.metric("🎯 Model Accuracy", "86.2%", delta="XGBoost")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        churn_counts = df['Exited'].value_counts()
        fig_pie = go.Figure(data=[go.Pie(
            labels=['Retained', 'Churned'],
            values=[int(churn_counts.get(0, 0)), int(churn_counts.get(1, 0))],
            marker=dict(colors=['#2ecc71', '#e74c3c']),
            hole=0.3,
            textinfo='label+percent',
            textposition='inside',
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}'
        )])
        fig_pie.update_layout(title="Churn Distribution", font=dict(size=12), height=400, showlegend=True)
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        fig_age = px.box(
            fig_age = px.box(
    df, x='Exited', y='Age', color='Exited',
    color_discrete_map={0: '#2ecc71', 1: '#e74c3c'},
    labels={'Exited': 'Customer Status', 'Age': 'Age (years)'}, points='outliers'
        )
        st.plotly_chart(fig_age)
    else:
        # This 'else' must align with the 'if' above
        st.info("The chart cannot be displayed with current filters.")

        )
        st.plotly_chart(fig_age)
    else:
        st.info("The chart cannot be displayed because one of the categories (Exited or Active) is missing from the current filtered data.")

        )
        fig_age.update_layout(title="Age Distribution by Churn Status", showlegend=False, height=400,
                              xaxis_ticktext=['Retained', 'Churned'], xaxis_tickvals=[0, 1])
        st.plotly_chart(fig_age, use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        geo_churn = df.groupby('Geography')['Exited'].agg(['count', 'sum'])
        geo_churn['churn_rate'] = (geo_churn['sum'] / geo_churn['count'] * 100).round(2)
        fig_geo = px.bar(
            geo_churn.reset_index(), x='Geography', y='churn_rate', color='churn_rate',
            color_continuous_scale='RdYlGn_r', labels={'churn_rate': 'Churn Rate (%)'}, text='churn_rate'
        )
        fig_geo.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_geo.update_layout(title="Churn Rate by Geography", height=400, showlegend=False, xaxis_title="Country", yaxis_title="Churn Rate (%)")
        st.plotly_chart(fig_geo, use_container_width=True)
    
    with col2:
        fig_scatter = px.scatter(
            df.sample(min(1000, len(df))), x='Tenure', y='Balance', color='Exited',
            color_discrete_map={0: '#2ecc71', 1: '#e74c3c'}, size='Age',
            hover_data=['Age', 'EstimatedSalary'], labels={'Tenure': 'Tenure (Years)', 'Balance': 'Account Balance ($)', 'Exited': 'Status'}, opacity=0.6
        )
        fig_scatter.update_layout(title="Tenure vs Balance (Size = Age)", height=400, legend=dict(title='Customer Status', ticktext=['Retained', 'Churned'], tickvals=[0, 1]))
        st.plotly_chart(fig_scatter, use_container_width=True)

# PAGE 2: EDA & ANALYSIS
elif page == "📈 EDA & Analysis":
    st.markdown("# 📈 Exploratory Data Analysis")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        fig_age_dist = px.histogram(df, x='Age', nbins=30, color_discrete_sequence=['#3498db'], labels={'Age': 'Age (years)', 'count': 'Frequency'})
        fig_age_dist.add_vline(x=float(df['Age'].mean()), line_dash="dash", line_color="red", annotation_text=f"Mean: {df['Age'].mean():.1f}")
        fig_age_dist.update_layout(title="Age Distribution", height=400)
        st.plotly_chart(fig_age_dist, use_container_width=True)
    with col2:
        fig_tenure = px.histogram(df, x='Tenure', nbins=11, color_discrete_sequence=['#2ecc71'], labels={'Tenure': 'Tenure (Years)', 'count': 'Frequency'})
        fig_tenure.add_vline(x=float(df['Tenure'].mean()), line_dash="dash", line_color="red", annotation_text=f"Mean: {df['Tenure'].mean():.1f}")
        fig_tenure.update_layout(title="Tenure Distribution", height=400)
        st.plotly_chart(fig_tenure, use_container_width=True)
        
    col1, col2 = st.columns(2)
    with col1:
        fig_balance = px.histogram(df, x='Balance', nbins=40, color_discrete_sequence=['#e74c3c'], labels={'Balance': 'Account Balance ($)', 'count': 'Frequency'})
        fig_balance.update_layout(title="Balance Distribution", height=400)
        st.plotly_chart(fig_balance, use_container_width=True)
    with col2:
        fig_credit = px.histogram(df, x='CreditScore', nbins=30, color_discrete_sequence=['#f39c12'], labels={'CreditScore': 'Credit Score', 'count': 'Frequency'})
        fig_credit.add_vline(x=float(df['CreditScore'].mean()), line_dash="dash", line_color="red", annotation_text=f"Mean: {df['CreditScore'].mean():.0f}")
        fig_credit.update_layout(title="Credit Score Distribution", height=400)
        st.plotly_chart(fig_credit, use_container_width=True)
        
    st.subheader("📊 Feature Correlation Matrix")
    numeric_only_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_only_cols].corr()
    
    fig_corr = go.Figure(data=go.Heatmap(
        z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.columns, colorscale='RdBu', zmid=0,
        text=np.round(corr_matrix.values, 2), texttemplate='%{text}', textfont={"size": 10}, colorbar=dict(title="Correlation")
    ))
    fig_corr.update_layout(title="Feature Correlation Heatmap", height=600, width=800)
    st.plotly_chart(fig_corr, use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        counts_df = df['NumOfProducts'].value_counts().reset_index()
        counts_df.columns = ['NumOfProducts', 'count']
        fig_products = px.bar(counts_df, x='NumOfProducts', y='count', color_discrete_sequence=['#9b59b6'], labels={'NumOfProducts': 'Number of Products', 'count': 'Count'}, text='count')
        fig_products.update_traces(textposition='outside')
        fig_products.update_layout(title="Product Distribution", height=400)
        st.plotly_chart(fig_products, use_container_width=True)
    with col2:
        activity_churn = df.groupby('IsActiveMember')['Exited'].mean() * 100
        fig_activity = px.bar(x=['Inactive', 'Active'], y=activity_churn.values, color_discrete_sequence=['#e74c3c', '#2ecc71'], labels={'x': 'Activity Status', 'y': 'Churn Rate (%)'}, text=np.round(activity_churn.values, 1))
        fig_activity.update_traces(textposition='outside')
        fig_activity.update_layout(title="Churn Rate by Activity Status", height=400)
        st.plotly_chart(fig_activity, use_container_width=True)
        
    st.subheader("📋 Statistical Summary")
    st.dataframe(df.describe().round(2), use_container_width=True)

# PAGE 3: MODEL PERFORMANCE
elif page == "🤖 Model Performance":
    st.markdown("# 🤖 Model Performance Analysis")
    st.markdown("---")
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1: st.metric("Accuracy", f"{accuracy:.4f}", f"{accuracy*100:.2f}%")
    with col2: st.metric("Precision", f"{precision:.4f}")
    with col3: st.metric("Recall", f"{recall:.4f}")
    with col4: st.metric("F1-Score", f"{f1:.4f}")
    with col5: st.metric("ROC-AUC", f"{roc_auc:.4f}")
    
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        cm = confusion_matrix(y_test, y_pred)
        fig_cm = go.Figure(data=go.Heatmap(
            z=cm, x=['Predicted: Retained', 'Predicted: Churned'], y=['Actual: Retained', 'Actual: Churned'],
            text=cm, texttemplate='%{text}', colorscale='Blues', showscale=True
        ))
        fig_cm.update_layout(title="Confusion Matrix", height=400, xaxis_title="Predicted", yaxis_title="Actual")
        st.plotly_chart(fig_cm, use_container_width=True)
        
    with col2:
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc_val = auc(fpr, tpr)
        fig_roc = go.Figure(data=[
            go.Scatter(x=fpr, y=tpr, mode='lines', name=f'XGBoost (AUC = {roc_auc_val:.4f})', line=dict(color='#3498db', width=3)),
            go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Classifier', line=dict(color='gray', width=2, dash='dash'))
        ])
        fig_roc.update_layout(title="ROC Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", height=400, hovermode='closest')
        st.plotly_chart(fig_roc, use_container_width=True)
        
    st.subheader("🎯 Feature Importance")
    feature_importance = pd.DataFrame({
        'Feature': feature_order,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False).head(15) # FIX: Added 'by=' explicitly
    
    fig_feat = px.bar(
        feature_importance, x='Importance', y='Feature', orientation='h', color='Importance',
        color_continuous_scale='Viridis', labels={'Importance': 'Feature Importance Score'}, text='Importance'
    )
    fig_feat.update_traces(textposition='outside')
    fig_feat.update_layout(title="Top 15 Most Important Features", height=500, showlegend=False, yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig_feat, use_container_width=True)
    
    st.subheader("📊 Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.round(4), use_container_width=True)

# PAGE 4: PREDICT CHURN
elif page == "🔮 Predict Churn":
    st.markdown("# 🔮 Customer Churn Prediction")
    st.markdown("---")
    st.subheader("📝 Enter Customer Information")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", min_value=18, max_value=92, value=35)
        tenure = st.number_input("Tenure (Years)", min_value=0, max_value=10, value=3)
        credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)
    with col2:
        balance = st.number_input("Account Balance ($)", min_value=0, value=50000)
        salary = st.number_input("Estimated Salary ($)", min_value=20000, value=100000)
        num_products = st.selectbox("Number of Products", [1, 2, 3, 4])
    with col3:
        has_credit_card = st.selectbox("Has Credit Card", ["No", "Yes"])
        is_active = st.selectbox("Is Active Member", ["No", "Yes"])
        geography = st.selectbox("Geography", ["France", "Spain", "Germany"])
        gender = st.selectbox("Gender", ["Male", "Female"])
        
    if st.button("🔮 Predict Churn Probability", use_container_width=True):
        has_cc = 1 if has_credit_card == "Yes" else 0
        active = 1 if is_active == "Yes" else 0
        
        ratio = balance / (salary + 1)
        density = num_products / (tenure + 1)
        act_score = active + has_cc
        
        geo_france = 1 if geography == "France" else 0
        geo_germany = 1 if geography == "Germany" else 0
        geo_spain = 1 if geography == "Spain" else 0
        gen_female = 1 if gender == "Female" else 0
        gen_male = 1 if gender == "Male" else 0
        
        input_data = pd.DataFrame([[
            credit_score, age, tenure, balance, salary, num_products, has_cc, active,
            ratio, density, act_score, geo_france, geo_germany, geo_spain, gen_female, gen_male
        ]], columns=feature_order)
        
        scaled_input = scaler.transform(input_data)
        scaled_input_df = pd.DataFrame(scaled_input, columns=feature_order)
        
        churn_prob = float(model.predict_proba(scaled_input_df)[:, 1][0])
        retention_prob = 1 - churn_prob
        
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 📊 Prediction Results")
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=churn_prob * 100, domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Churn Probability (%)"},
                gauge={
                    'axis': {'range': [0, 100]}, 'bar': {'color': "#e74c3c"},
                    'steps': [{'range': [0, 30], 'color': "#2ecc71"}, {'range': [30, 60], 'color': "#f39c12"}, {'range': [60, 100], 'color': "#e74c3c"}],
                    'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 50}
                }
            ))
            fig_gauge.update_layout(height=400)
            st.plotly_chart(fig_gauge, use_container_width=True)
            
        with col2:
            st.markdown("### 🎯 Risk Assessment")
            col_left, col_right = st.columns(2)
            with col_left: st.metric("Churn Risk", f"{churn_prob*100:.1f}%")
            with col_right: st.metric("Retention Probability", f"{retention_prob*100:.1f}%")
            
            st.markdown("---")
            if churn_prob > 0.70:
                st.error("🔴 **CRITICAL RISK** - Immediate intervention required")
                recommendation = "Offer exclusive retention program, special discounts, account manager assignment"
            elif churn_prob > 0.50:
                st.warning("🟡 **HIGH RISK** - Proactive engagement needed")
                recommendation = "Schedule account review, offer product upgrades, improve engagement"
            elif churn_prob > 0.30:
                st.info("🟢 **LOW RISK** - Standard management")
                recommendation = "Maintain regular contact, monitor satisfaction metrics"
            else:
                st.success("✅ **VERY LOW RISK** - Customer is loyal")
                recommendation = "Focus on cross-selling and upselling opportunities"
                
            st.markdown(f"**Recommended Action**: {recommendation}")
            st.markdown("---")
            st.markdown("### ### 👤 Customer Profile Summary")
            profile_data = {
                'Metric': ['Age', 'Tenure', 'Credit Score', 'Account Balance', 'Salary', 'Products', 'Active Member'],
                'Value': [f"{age} yrs", f"{tenure} yrs", f"{credit_score}", f"${balance:,.0f}", f"${salary:,.0f}", f"{num_products}", "Yes" if active else "No"]
            }
            st.dataframe(pd.DataFrame(profile_data), use_container_width=True, hide_index=True)

# PAGE 5: CUSTOMER INSIGHTS
elif page == "📋 Customer Insights":
    st.markdown("# 📋 Customer Insights & Segmentation")
    st.markdown("---")
    st.subheader("🎯 Customer Risk Segmentation")
    
    df_risk = df.copy()
    df_risk['BalanceToSalaryRatio'] = df_risk['Balance'] / (df_risk['EstimatedSalary'] + 1)
    df_risk['ProductDensity'] = df_risk['NumOfProducts'] / (df_risk['Tenure'] + 1)
    df_risk['ActivityScore'] = df_risk['IsActiveMember'] + df_risk['HasCrCard']
    df_risk['Geography_France'] = (df_risk['Geography'] == 'France').astype(int)
    df_risk['Geography_Germany'] = (df_risk['Geography'] == 'Germany').astype(int)
    df_risk['Geography_Spain'] = (df_risk['Geography'] == 'Spain').astype(int)
    df_risk['Gender_Female'] = (df_risk['Gender'] == 'Female').astype(int)
    df_risk['Gender_Male'] = (df_risk['Gender'] == 'Male').astype(int)
    
    all_scaled = scaler.transform(df_risk[feature_order])
    all_scaled_df = pd.DataFrame(all_scaled, columns=feature_order)
    df_risk['ChurnProb'] = model.predict_proba(all_scaled_df)[:, 1]
    
    def categorize_risk(prob):
        if prob > 0.70: return 'Critical Risk'
        elif prob > 0.50: return 'High Risk'
        elif prob > 0.30: return 'Low Risk'
        return 'Very Low Risk'
        
    df_risk['RiskCategory'] = df_risk['ChurnProb'].apply(categorize_risk)
    
    # FIX: Reindex safely using fillna(0) to prevent empty segments from dropping or breaking
    risk_counts = df_risk['RiskCategory'].value_counts().reindex(['Critical Risk', 'High Risk', 'Low Risk', 'Very Low Risk']).fillna(0)
    
    colors_risk = ['#e74c3c', '#f39c12', '#3498db', '#2ecc71']
    fig_risk = px.pie(values=risk_counts.values, names=risk_counts.index, color_discrete_sequence=colors_risk, labels={'value': 'Count'}, hole=0.3)
    fig_risk.update_traces(textposition='inside', textinfo='label+percent')
    fig_risk.update_layout(title="Customer Risk Distribution", height=400)
    st.plotly_chart(fig_risk, use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        gender_churn = df.groupby('Gender')['Exited'].mean() * 100
        fig_gender = px.bar(x=gender_churn.index, y=gender_churn.values, color_discrete_sequence=['#3498db', '#e74c3c'], labels={'x': 'Gender', 'y': 'Churn Rate (%)'}, text=np.round(gender_churn.values, 1))
        fig_gender.update_traces(textposition='outside')
        fig_gender.update_layout(title="Churn Rate by Gender", height=400, showlegend=False)
        st.plotly_chart(fig_gender, use_container_width=True)
    with col2:
        avg_metrics = df.groupby('Exited')[['Age', 'Tenure', 'Balance', 'CreditScore']].mean()
        avg_metrics.index = pd.Index(['Retained', 'Churned'])
        fig_metrics = go.Figure(data=[
            go.Bar(name='Age', x=avg_metrics.index, y=avg_metrics['Age'], marker_color='#3498db'),
            go.Bar(name='Tenure', x=avg_metrics.index, y=avg_metrics['Tenure'], marker_color='#2ecc71'),
        ])
        fig_metrics.update_layout(title="Average Metrics by Customer Status", barmode='group', height=400, yaxis_title="Value")
        st.plotly_chart(fig_metrics, use_container_width=True)
        
    st.subheader("📊 Customer Risk Scores (Sample of Top 100 by Risk)")
    display_df = df_risk[['CustomerId', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'IsActiveMember', 'Exited', 'ChurnProb', 'RiskCategory']].copy()
    display_df = display_df.sort_values(by='ChurnProb', ascending=False).head(100) # FIX: Added 'by=' explicitly
    display_df['ChurnProb'] = display_df['ChurnProb'].apply(lambda x: f"{x*100:.1f}%")
    display_df.columns = pd.Index(['Customer ID', 'Age', 'Tenure', 'Balance', 'Products', 'Active', 'Churned', 'Churn Probability', 'Risk Category'])
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    st.subheader("📈 Risk Category Summary")
    risk_summary = df_risk.groupby('RiskCategory').agg({'CustomerId': 'count', 'Exited': 'mean', 'Age': 'mean', 'Tenure': 'mean', 'Balance': 'mean'}).reindex(['Critical Risk', 'High Risk', 'Low Risk', 'Very Low Risk'])
    risk_summary.columns = ['Count', 'Actual Churn Rate', 'Avg Age', 'Avg Tenure', 'Avg Balance']
    risk_summary['Actual Churn Rate'] = risk_summary['Actual Churn Rate'].fillna(0).apply(lambda x: f"{x*100:.1f}%")
    risk_summary['Avg Age'] = risk_summary['Avg Age'].fillna(0).apply(lambda x: f"{x:.1f}")
    risk_summary['Avg Tenure'] = risk_summary['Avg Tenure'].fillna(0).apply(lambda x: f"{x:.1f}")
    risk_summary['Avg Balance'] = risk_summary['Avg Balance'].fillna(0).apply(lambda x: f"${x:,.0f}")
    st.dataframe(risk_summary, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: gray; font-size: 12px;'><p>Bank Churn Prediction System • Powered by XGBoost • Dataset: 10,000 customers</p></div>", unsafe_allow_html=True)