# AI Enforcement Intelligence System

> **Advanced Data Science & ML Application for Multi-Platform Enforcement Analytics**

A production-ready Streamlit application showcasing modern data science practices, machine learning integration, and interactive data visualization for enforcement monitoring and predictive analytics.

---

## ✨ Features

### Core Analytics
- **Multi-Platform Monitoring** - Comparative analysis across different enforcement platforms
- **Time-Series Analysis** - Track trends with monthly aggregation and trend detection
- **Anomaly Detection** - ML-powered isolation forest for outlier identification
- **Organization Ranking** - Comprehensive performance metrics and comparisons

### Predictive Intelligence
- **6-Month Forecasting** - Linear regression-based trend projection
- **Trend Analysis** - Statistical trend calculation with growth rate metrics
- **Risk Intelligence** - Volatility assessment and anomaly flagging

### User Experience
- **Interactive Dashboards** - Real-time filters and responsive visualizations
- **Professional Dark Theme** - Modern, eye-friendly design with gradient backgrounds
- **Multi-page Application** - Modular architecture with Streamlit multi-page support
- **PDF Export** - Generate detailed reports programmatically

### AI Assistant
- **Offline NLP Engine** - Intent-based Q&A without external APIs
- **Analytics Queries** - Natural language interface to data insights
- **No API Dependencies** - Fully self-contained ML inference

---

## 🛠️ Tech Stack

### Core Framework
- **Streamlit** - Interactive web application framework
- **Python 3.10+** - Modern Python with type hints

### Data Science & ML
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Scikit-learn** - Machine learning (Isolation Forest, Linear Regression)
- **Plotly** - Interactive data visualization

### Reporting & Export
- **ReportLab** - PDF document generation
- **Python-dotenv** - Secure environment configuration

### Deployment
- **Render.yaml** - Cloud deployment configuration
- **Docker-ready** - Container-compatible architecture

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10 or higher
- pip or conda package manager
- Git

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd social_media_Main
```

2. **Create virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure environment**
```bash
cp .env.example .env
# Edit .env with your configuration if needed
```

5. **Run the application**
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## 📁 Project Structure

```
social_media_Main/
├── app.py                              # Main dashboard application
├── config.py                           # Configuration management
├── requirements.txt                    # Python dependencies
├── .env.example                        # Environment template
├── .gitignore                          # Git ignore rules
├── .streamlit/                         # Streamlit config
├── pages/
│   └── forecast.py                    # Predictive intelligence module
├── preprocessed_enforcement_data.csv  # Data file (excluded from git)
├── Procfile                            # Deployment configuration
├── render.yaml                         # Render deployment config
└── README.md                           # This file
```

---

## 🔐 Security Best Practices

This project implements enterprise-grade security practices:

### Secrets Management
- **No Hardcoded Credentials** - All sensitive data in `.env`
- **Environment Variables** - Secure configuration through config.py
- **Git Protection** - `.env` excluded from version control
- **Template File** - `.env.example` shows required variables

### Data Protection
- **Input Validation** - Required columns checked on data load
- **Error Handling** - Graceful failure modes with user-friendly messages
- **Type Checking** - Python type hints for better IDE support

### Code Quality
- **Docstrings** - All functions documented with Args/Returns
- **Modular Architecture** - Separation of concerns (config, main app, pages)
- **Caching** - Streamlit cache decorators for performance

---

## 📊 Data Format

The application expects a CSV file with these columns:

| Column | Type | Description |
|--------|------|-------------|
| `date` | datetime | Date of the enforcement action |
| `organization` | string | Platform/organization name |
| `action_as_per_source` | string | Type of action (Content Removed, Account Banned, etc.) |
| `standard_value` | numeric | Enforcement volume/count |

**Supported Actions:**
- Content Actioned
- Content Removed
- Removed
- Total Accounts Banned
- Total Accounts Suspended

---

## 💡 Machine Learning Models

### Anomaly Detection
**Algorithm:** Isolation Forest
- **Contamination:** 8% (configurable in config.py)
- **Purpose:** Identify unusual enforcement patterns
- **Output:** Flagged anomalies with timestamps

### Trend Forecasting
**Algorithm:** Linear Regression
- **Method:** Least squares line fitting to time-indexed data
- **Forecast Horizon:** 6 months ahead
- **Metrics:** Slope (trend direction) and R² validation

### NLP Intent Classification
**Method:** Keyword-based classification
- **Offline:** No API calls required
- **Categories:** Trend, highest, lowest, average, comparison, volatility, forecast
- **Performance:** Real-time response

---

## 🎨 UI/UX Features

### Professional Dark Theme
- **Gradient Backgrounds** - Modern, eye-friendly blue gradients
- **Consistent Styling** - CSS variables applied across pages
- **Responsive Design** - Mobile and desktop compatible
- **Accessibility** - High contrast colors and readable typography

### Interactive Components
- **Dynamic Filters** - Date range and multi-select organization filters
- **Real-time Charts** - Plotly interactive visualizations
- **Hover Information** - Detailed tooltip data on charts
- **Export Functionality** - PDF reports with one click

---

## 📈 Analytics Features

### Key Metrics Dashboard
- Latest Month Volume
- Average Monthly Value
- Overall Trend (📈 up, 📉 down, ➖ stable)
- Growth Percentage

### Visualizations
1. **Trend Chart** - Line plot by organization over time
2. **Organization Rankings** - Sortable performance table
3. **Anomaly Detection** - Highlighted unusual patterns
4. **Forecast Chart** - 6-month prediction with regression line

### AI Assistant Queries
- "Show me the trend" → Trend direction and percentage
- "Who is the top performer?" → Highest organization
- "What's the average?" → Monthly average metrics
- "Forecast the next 6 months" → Future trend projection
- "Which org is most volatile?" → Volatility analysis

---

## 🔧 Configuration

### Environment Variables
All configuration through `.env` file:

```env
ENVIRONMENT=development
DATA_PATH=preprocessed_enforcement_data.csv
API_KEY=your_api_key_here  # If using external APIs
```

### Streamlit Settings
Located in `.streamlit/config.toml`:
- Server configuration
- Logger settings
- Client preferences

---

## 📦 Deployment

### Render Deployment
Pre-configured in `render.yaml`:
```bash
render deploy
```

**Features:**
- Automatic Python 3.10 environment
- All dependencies installed
- Streamlit server hardened for production

### Docker Deployment
Container-ready with standard Dockerfile setup

---

## 🧪 Testing & Validation

### Data Validation
- CSV format verification
- Required column checks
- Date format validation
- Type conversion error handling

### Model Validation
- Minimum data points for anomaly detection (n > 6)
- Linear regression slope confidence
- Forecast confidence intervals

---

## 📝 Data Science Methodology

### Exploratory Data Analysis (EDA)
1. Time-series decomposition
2. Organization distribution analysis
3. Action type frequency distribution
4. Temporal trends identification

### Feature Engineering
- Time-based indexing for regression
- Date aggregation (daily → monthly)
- Organization-wise segregation
- Anomaly scoring via Isolation Forest

### Model Validation
- Regression R² scores
- Anomaly detection precision/recall
- Forecast error metrics
- Cross-validation ready

---

## 🎓 Key Skills Demonstrated

This project showcases:
- ✅ Full-stack data science application development
- ✅ Production-ready Python code practices
- ✅ ML model integration and deployment
- ✅ Interactive UI/UX with Streamlit
- ✅ Secure credential management
- ✅ Data visualization best practices
- ✅ API-less AI implementation
- ✅ Cloud deployment readiness
- ✅ Time-series forecasting
- ✅ Anomaly detection algorithms

---

## 🐛 Troubleshooting

### Issue: "Data file not found"
**Solution:** Ensure `preprocessed_enforcement_data.csv` is in the root directory

### Issue: Import errors
**Solution:** Run `pip install -r requirements.txt` again

### Issue: Slow performance on large datasets
**Solution:** Filter by date range or organization in sidebar

### Issue: Environment variables not loading
**Solution:** Ensure `.env` file is in root directory, not `.env.txt`

---

## 📧 Support & Contribution

For issues, questions, or improvements:
1. Check existing documentation
2. Review code comments and docstrings
3. Test with sample data first
4. Submit detailed error reports with stack traces

---

## 📄 License

MIT License - See LICENSE file for details

---

## 👤 Author

**Data Science Professional**
- Focus: Data Analytics, ML Engineering, Cloud Deployment
- Stack: Python, SQL, Streamlit, Scikit-learn, Pandas
- Expertise: Time-series forecasting, Anomaly detection, Dashboard development

---

## 🙏 Acknowledgments

- **Streamlit Community** - For the amazing web framework
- **Plotly** - For interactive visualizations
- **Scikit-learn** - For ML algorithms
- **Render** - For cloud hosting

---

**Last Updated:** April 2026  
**Version:** 1.0.0  
**Status:** Production Ready ✨

The app will be available at `http://localhost:8501`

## 📤 Deployment to Render

### Step 1: Push to GitHub
```bash
git add .
git commit -m "fix: resolve git conflicts and deployment configuration"
git push origin main
```

### Step 2: Deploy on Render
1. Go to [render.com](https://render.com)
2. Click "New" → "Web Service"
3. Connect your GitHub repository
4. Select the repository: `social_media_Main`
5. **Runtime**: Python 3.10
6. **Build Command**: `pip install -r requirements.txt`
7. **Start Command**: `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`
8. Click "Create Web Service"

### Step 3: Set Environment Variables (in Render Dashboard)
Add these environment variables:
- `STREAMLIT_SERVER_HEADLESS=true`
- `STREAMLIT_SERVER_ENABLECORS=false`
- `STREAMLIT_LOGGER_LEVEL=info`

## 📁 Project Structure

```
social_media_Main/
├── app.py                          # Main dashboard
├── pages/
│   └── forecast.py                 # Predictive analytics page
├── preprocessed_enforcement_data.csv # Dataset
├── requirements.txt                # Python dependencies
├── setup.sh                        # Render setup script
├── Procfile                        # Procfile for Render
├── render.yaml                     # Render configuration
├── .streamlit/
│   └── config.toml                # Streamlit theme configuration
├── .gitignore                      # Git ignore rules
└── .python-version                # Python version specification
```

## 🔧 Configuration

The `.streamlit/config.toml` file contains the professional theme settings:
- **Primary Color**: #0066ff (Modern Blue)
- **Background**: #0f1419 (Dark)
- **Secondary Background**: #1a1f28 (Darker)
- **Text Color**: #e8eef2 (Light Gray)
- **Font**: Sans serif

## 📊 Dashboard Features

### Main Dashboard (app.py)
- Monthly enforcement trends across organizations
- Organization ranking and benchmarking
- Anomaly detection with risk flagging
- 6-month enforcement forecasts
- KPI metrics (Latest, Average, Trend, Growth %)
- AI-powered natural language query assistant
- PDF report export functionality

### Forecast Page (pages/forecast.py)
- Top 4 organization selection
- Individual forecasting with confidence bands
- Growth projection comparison
- Reliability scoring system
- Historical + predicted visualization

## 🤖 AI Assistant Capabilities

The offline AI engine handles natural language queries:
- **Trend Analysis**: Ask about increasing/decreasing patterns
- **Highest/Lowest**: Identify top and bottom performers
- **Comparison**: Compare organizations side by side
- **Forecasting**: Get 6-month projections
- **Volatility**: Analyze organization instability

## 📈 Data Processing

- **Input**: CSV with enforcement data (date, organization, action, value)
- **Processing**: Grouping, aggregation, time-series analysis
- **Output**: Interactive visualizations, metrics, predictions

## ⚙️ Environment Variables

No API keys or sensitive data required! This is a fully offline, client-side analytics application.

## 🐛 Troubleshooting

### Issue: "ModuleNotFoundError"
```bash
pip install -r requirements.txt --force-reinstall
```

### Issue: "Port already in use"
```bash
streamlit run app.py --server.port 8502
```

### Issue: "Connection timeout on Render"
- Check that all environment variables are set correctly
- Verify the CSV file path is accessible
- Check Render logs for detailed error messages

## 📝 Notes

- All computations are done client-side (no server communication)
- Data remains private - nothing is sent to external services
- Perfect for portfolio demonstrations of data science capabilities
- Fully reproducible from the CSV dataset

## 🎯 Portfolio Value

This project demonstrates:
- ✅ Full-stack data science skills (ML, visualization, deployment)
- ✅ Professional UI/UX design principles
- ✅ Production-ready code with proper configuration management
- ✅ Cloud deployment expertise (Render)
- ✅ Git workflow and version control
- ✅ Advanced analytics (forecasting, anomaly detection, NLP)

## 📧 Support

For issues or improvements, please open an issue on GitHub.

---

**Ready to deploy!** Push to GitHub and follow the Render deployment steps above. 🚀
