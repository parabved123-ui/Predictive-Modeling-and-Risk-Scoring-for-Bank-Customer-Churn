# ⚙️ Configuration & Customization Guide

Learn how to customize the Bank Churn Dashboard for your needs.

---

## 🎨 UI Customization

### Change App Title & Icon

Edit the first lines in `streamlit_app.py`:

```python
st.set_page_config(
    page_title="My Bank Analytics",  # Change this
    page_icon="💼",                   # Change this emoji
    layout="wide",                    # or "centered"
    initial_sidebar_state="expanded"  # or "collapsed"
)
```

### Change Color Scheme

Modify CSS in the styling section:

```python
st.markdown("""
    <style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        # Change these hex colors
    }
    </style>
    """, unsafe_allow_html=True)
```

**Popular color combinations:**
- Professional: `#1f77b4` (Blue), `#ff7f0e` (Orange)
- Healthcare: `#2ecc71` (Green), `#e74c3c` (Red)
- Finance: `#3498db` (Blue), `#f39c12` (Gold)
- Tech: `#9b59b6` (Purple), `#1abc9c` (Teal)

---

## 📊 Data Customization

### Load Your Own Data

Replace the `load_data()` function:

```python
@st.cache_data
def load_data():
    # Option 1: Load from CSV
    df = pd.read_csv('your_customer_data.csv')
    
    # Option 2: Load from Excel
    df = pd.read_excel('your_customer_data.xlsx')
    
    # Option 3: Load from SQL Database
    import sqlalchemy
    engine = sqlalchemy.create_engine('your_database_url')
    df = pd.read_sql('SELECT * FROM customers', engine)
    
    return df
```

### Map Your Columns

If your data has different column names:

```python
# At the top of the script, add:
COLUMN_MAPPING = {
    'age': 'Age',
    'tenure_years': 'Tenure',
    'balance_usd': 'Balance',
    'salary_usd': 'EstimatedSalary',
    'churned': 'Exited',  # Your target variable
    'country': 'Geography',
    'is_active': 'IsActiveMember'
}

# Then in load_data():
df = df.rename(columns=COLUMN_MAPPING)
```

---

## 🤖 Model Customization

### Train on Your Data

Replace the model training section:

```python
# In the "🤖 Model Performance" section, modify:
from sklearn.model_selection import train_test_split

# Split your data
X = df_scaled.drop(['CustomerId', 'Exited'], axis=1)
y = df['Exited']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,      # 80-20 split
    random_state=42,
    stratify=y           # Maintain class balance
)

# Train XGBoost
model = xgb.XGBClassifier(
    n_estimators=200,        # More iterations = better but slower
    max_depth=7,             # Deeper trees = more complex patterns
    learning_rate=0.05,      # Lower = more careful learning
    min_child_weight=5,      # Higher = avoid overfitting
    subsample=0.8,           # 80% of data per iteration
    colsample_bytree=0.8,    # 80% of features per iteration
    reg_lambda=1.0,          # L2 regularization
    random_state=42
)

model.fit(X_train, y_train)
```

### Use Different Models

```python
# Try other models:

# 1. Random Forest
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)

# 2. Gradient Boosting
from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5
)

# 3. Logistic Regression
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000)

# Train any model the same way:
model.fit(X_train, y_train)
```

---

## 📈 Feature Engineering

### Add Custom Features

Modify the `load_data()` function:

```python
def load_data():
    df = pd.read_csv('your_data.csv')
    
    # Add derived features
    df['BalanceToSalaryRatio'] = df['Balance'] / (df['Salary'] + 1)
    df['ProductsPerYear'] = df['NumOfProducts'] / (df['Tenure'] + 1)
    df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 30, 40, 50, 100])
    df['LongTenure'] = (df['Tenure'] > 5).astype(int)
    df['HighBalance'] = (df['Balance'] > df['Balance'].median()).astype(int)
    
    return df
```

---

## 🎯 Prediction Customization

### Change Churn Probability Calculation

In the "🔮 Predict Churn" section:

```python
# Current calculation:
churn_prob = 0.3  # Base rate

# Customize weights:
churn_prob += (abs(age - 40) / 100) * 0.20  # Age weight: 20%
churn_prob -= (tenure / 10) * 0.25            # Tenure weight: 25%
churn_prob -= min((balance / salary) * 0.15, 0.15)  # Balance weight: 15%
# ... adjust weights as needed

# Normalize to 0-1
churn_prob = max(0, min(1, churn_prob))
```

### Customize Risk Categories

```python
def categorize_risk(prob):
    if prob > 0.80:              # Change thresholds
        return '🔴 Critical'
    elif prob > 0.60:
        return '🟠 High'
    elif prob > 0.40:
        return '🟡 Medium'
    else:
        return '🟢 Low'
```

---

## 📊 Visualization Customization

### Change Chart Colors

```python
# For Plotly charts:
color_discrete_map = {
    0: '#2ecc71',  # Green for retained
    1: '#e74c3c'   # Red for churned
}

# Apply to any chart:
fig = px.bar(
    df.groupby('Geography')['Exited'].mean(),
    color_discrete_sequence=['#3498db', '#9b59b6', '#f39c12']
)
```

### Add New Visualizations

```python
# Add to any tab:
st.subheader("My Custom Chart")

# Create with Plotly
fig = px.scatter(
    df,
    x='Tenure',
    y='Balance',
    color='Exited',
    size='Age',
    hover_data=['CustomerId', 'Geography']
)
fig.update_layout(title="My Chart", height=500)
st.plotly_chart(fig, use_container_width=True)

# Or with regular Matplotlib
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(df['Tenure'], df['Balance'], alpha=0.5)
ax.set_xlabel('Tenure')
ax.set_ylabel('Balance')
st.pyplot(fig)
```

---

## 🔐 Performance Optimization

### Enable Caching for Large Datasets

```python
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_data():
    return pd.read_csv('large_file.csv')

@st.cache_resource  # Cache resource once
def load_model():
    return joblib.load('trained_model.pkl')
```

### Reduce Data Size

```python
# In load_data():
if len(df) > 50000:
    # Sample for faster processing
    df = df.sample(50000, random_state=42)
    st.warning(f"Dataset sampled to 50,000 rows for performance")
```

---

## 🌐 Deployment Configuration

### Environment Variables

Create `.env` file:
```
DATABASE_URL=postgresql://user:pass@localhost/dbname
API_KEY=your_api_key_here
DEBUG=False
```

Use in code:
```python
import os
from dotenv import load_dotenv

load_dotenv()
database_url = os.getenv('DATABASE_URL')
```

### Streamlit Config

Create `.streamlit/config.toml`:
```toml
[theme]
primaryColor = "#FF4B4B"
backgroundColor = "#0E1117"
secondaryBackgroundColor = "#262730"
textColor = "#FAFAFA"
font = "sans serif"

[client]
showErrorDetails = false
toolbarMode = "minimal"

[server]
port = 8501
headless = true
runOnSave = true
```

---

## 📧 Advanced: Add Email Alerts

```python
import smtplib
from email.mime.text import MIMEText

def send_alert(customer_id, churn_prob):
    """Send email for high-risk customers"""
    if churn_prob > 0.70:
        message = f"Customer {customer_id} has {churn_prob*100:.1f}% churn risk"
        
        # Configure your email
        sender = "your_email@gmail.com"
        password = "your_app_password"
        recipient = "manager@bank.com"
        
        msg = MIMEText(message)
        msg['Subject'] = f"HIGH RISK: Customer {customer_id}"
        
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(sender, password)
            server.sendmail(sender, recipient, msg.as_string())
```

---

## 📱 Mobile Optimization

The app is responsive by default, but optimize further:

```python
# Add to page config:
st.set_page_config(
    layout="wide",
    initial_sidebar_state="collapsed"  # Hide sidebar on mobile
)

# Use mobile-friendly layouts:
if st.session_state.get('is_mobile', False):
    col1, col2 = st.columns(1)  # Stack columns on mobile
else:
    col1, col2 = st.columns(2)  # Side by side on desktop
```

---

## 🔧 Debugging Tips

### Enable Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Use in your code:
logger.debug(f"Data shape: {df.shape}")
logger.warning(f"Missing values: {df.isnull().sum()}")
```

### Add Progress Indicators

```python
with st.spinner('Training model...'):
    model.fit(X_train, y_train)

# Or progress bar
progress_bar = st.progress(0)
for i in range(100):
    progress_bar.progress(i + 1)
```

### Inspect Data

```python
with st.expander("📋 Debug: Data Sample"):
    st.write(f"Shape: {df.shape}")
    st.dataframe(df.head())
    st.write(df.describe())
```

---

## 🚀 Next Steps

1. **Backup Original**: Save original `streamlit_app.py`
2. **Make Small Changes**: Test one modification at a time
3. **Use Git**: Track changes with version control
4. **Document Changes**: Add comments to modified sections
5. **Test Thoroughly**: Verify calculations with sample data

---

**Need More Help?** Check README.md or Streamlit docs: https://docs.streamlit.io
