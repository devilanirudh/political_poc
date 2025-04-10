# Political Insights Hub

A sophisticated dashboard application built with Streamlit, designed to showcase data-driven political consultancy capabilities through advanced analytics, AI-powered sentiment analysis, and predictive modeling.

## 🚀 Features

- **Interactive Profile Showcase** - Professional presentation of skills and experience
- **Political Sentiment Analysis** - Real-time analysis of political statements and public sentiment
- **Voter Behavior Prediction** - Machine learning models to predict voter turnout and preferences
- **Policy Impact Simulator** - Analyze and forecast outcomes of various policy decisions
- **Campaign Effectiveness** - Track and optimize campaign strategies (coming soon)
- **Decision Support System** - Comprehensive data-driven decision framework (coming soon)

## 📋 Prerequisites

- Python 3.8+
- pip (Python package manager)

## 🔧 Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/political-insights-hub.git
cd political-insights-hub
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
```

3. Activate the virtual environment:
   - On Windows:
   ```bash
   venv\Scripts\activate
   ```
   - On macOS/Linux:
   ```bash
   source venv/bin/activate
   ```

4. Install required dependencies:
```bash
pip install -r requirements.txt
```

5. Run the application:
```bash
streamlit run main2.py
```

## ⚙️ Dependencies

Create a requirements.txt file with the following contents:

```
streamlit==1.24.0
pandas==1.5.3
numpy==1.23.5
plotly==5.14.1
matplotlib==3.7.1
seaborn==0.12.2
nltk==3.8.1
scikit-learn==1.2.2
streamlit-option-menu==0.3.2
wordcloud==1.8.2.2
altair==4.2.2
pillow==9.5.0
joblib==1.2.0
```

## 📊 Application Modules

### 1. Political Sentiment Analysis

This module demonstrates the power of natural language processing to analyze public sentiment toward political entities, policies, and events. Features include:

- **Statement Analysis** - Evaluate the sentiment of political statements
- **Batch Processing** - Analyze multiple statements simultaneously
- **Trend Visualization** - Track sentiment changes over time
- **Key Term Extraction** - Identify important terms in political discourse

### 2. Voter Behavior Prediction

Leverage machine learning to understand and predict voter behavior:

- **Turnout Prediction Model** - Identify likely voters based on multiple factors
- **Geographic Analysis** - Visualize turnout patterns by geographic region
- **Voter Profile Simulator** - Interactive tool to understand voting likelihood by demographic
- **Resource Allocation** - Optimize campaign resources based on data-driven insights

### 3. Policy Impact Simulator

Forecast the potential impact of policy decisions across multiple dimensions:

- **Multi-dimensional Analysis** - Evaluate policies across key performance indicators
- **Comparative Scenario Planning** - Compare different policy approaches
- **Electoral Impact Modeling** - Understand how policy choices affect electoral outcomes
- **Demographic Response Prediction** - See how different demographic groups respond to policies

## 💡 Use Cases for Political Consultancy

- **Data-Driven Strategy Development** - Create campaign and policy strategies based on solid data analysis
- **Audience Targeting** - Identify key voter groups and optimize messaging
- **Message Testing** - Analyze sentiment response to different political messages
- **Policy Development** - Build policies with broader appeal and better outcomes
- **Resource Optimization** - Allocate campaign resources for maximum impact
- **Risk Mitigation** - Identify potential challenges in advance

## 🛠️ Customization

The application is designed to be highly customizable. You can modify:

- Data sources for real political data
- Machine learning models for improved accuracy
- UI elements for client branding
- Analysis parameters and dimensions

## 👨‍💻 About the Developer

This application was developed by [Anirudh Dev](https://github.com/devilanirudh), a Backend AI Engineer with expertise in machine learning, data analytics, and political data science.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details. 