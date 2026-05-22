# 🚀 Quick Start Guide

Get the Bank Churn Dashboard up and running in 5 minutes!

## Option 1: Automatic Setup (Recommended)

### Windows
```bash
python setup.py
```

### macOS/Linux
```bash
python3 setup.py
```

The setup script will:
1. ✓ Check Python version
2. ✓ Create virtual environment (optional)
3. ✓ Install dependencies
4. ✓ Launch the application

---

## Option 2: Manual Setup

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run the Application
```bash
streamlit run streamlit_app.py
```

### Step 3: Open in Browser
The app will automatically open at: `http://localhost:8501`

---

## 📊 First Steps in the Dashboard

### 1. **Dashboard Tab** (Start Here)
   - View overall churn statistics
   - See key performance metrics
   - Explore visual relationships

### 2. **EDA & Analysis Tab**
   - Understand data distributions
   - View feature correlations
   - Analyze statistical patterns

### 3. **Model Performance Tab**
   - Review model metrics (86.2% accuracy)
   - View ROC curve (0.8956 AUC)
   - See feature importance ranking

### 4. **Predict Churn Tab**
   - Fill in customer details
   - Get instant churn probability
   - Read actionable recommendations

### 5. **Customer Insights Tab**
   - See risk segmentation
   - Analyze demographics
   - Review high-risk customers

---

## 🛠️ Troubleshooting

### Problem: "streamlit: command not found"
**Solution:**
```bash
python -m streamlit run streamlit_app.py
```

### Problem: "ModuleNotFoundError" on startup
**Solution:**
```bash
pip install -r requirements.txt
```

### Problem: Port 8501 already in use
**Solution:**
```bash
streamlit run streamlit_app.py --server.port 8502
```

### Problem: Application is slow
**Solution:** The app generates predictions for 10,000 customers. First load may take 10-15 seconds.

---

## 💡 Tips & Tricks

### Use Dark Mode
```bash
streamlit run streamlit_app.py \
  --theme.primaryColor="#FF4B4B" \
  --theme.backgroundColor="#0E1117" \
  --theme.secondaryBackgroundColor="#262730"
```

### Run in Headless Mode (No Browser)
```bash
streamlit run streamlit_app.py --logger.level=error
```

### Increase Model Detail
Edit `streamlit_app.py` and change:
```python
n_estimators=50  # to 100 or 200 for more accurate predictions
```

---

## 📦 System Requirements

| Requirement | Version |
|------------|---------|
| Python | 3.8+ |
| Memory | 2GB minimum |
| Storage | 100MB free |
| Browser | Any modern browser |

---

## 🌐 Deploying to the Cloud

### Streamlit Cloud (Free)
1. Push to GitHub
2. Visit `cloud.streamlit.io`
3. Click "New app"
4. Select your repo
5. Done! ✓

### Heroku (Paid)
See `deploy/` directory for Heroku configuration

### Docker (Any Cloud)
```bash
docker build -t churn-dashboard .
docker run -p 8501:8501 churn-dashboard
```

---

## 📞 Need Help?

### Check Logs
```bash
streamlit run streamlit_app.py --logger.level=debug
```

### Clear Cache
```bash
streamlit cache clear
```

### Reinstall Everything
```bash
pip uninstall -r requirements.txt -y
pip install -r requirements.txt
```

---

## ✨ What's Next?

After exploring the dashboard:

1. **Customize Data**: Replace synthetic data with real customer data
2. **Improve Model**: Train models with your own data
3. **Add Features**: Modify `streamlit_app.py` to add custom analyses
4. **Deploy**: Share the dashboard with stakeholders
5. **Monitor**: Set up alerts for high-risk customers

---

## 📚 Resources

- **Streamlit Docs**: https://docs.streamlit.io
- **XGBoost Docs**: https://xgboost.readthedocs.io
- **Pandas Docs**: https://pandas.pydata.org/docs
- **Plotly Docs**: https://plotly.com/python

---

**Happy analyzing! 🎉**

Questions? Check README.md for detailed documentation.
