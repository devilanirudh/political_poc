import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import requests
from io import BytesIO
import time
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
import json
from wordcloud import WordCloud
import altair as alt
from streamlit_option_menu import option_menu
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import plotly.figure_factory as ff

# Download nltk resources
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

# App configuration
st.set_page_config(
    page_title="Data-Driven Decision Hub | Anirudh Dev",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
def load_css():
    st.markdown("""
    <style>
        .main {
            background-color: #f5f7f9;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 24px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: #f0f2f6;
            border-radius: 4px 4px 0px 0px;
            gap: 1px;
            padding-top: 10px;
            padding-bottom: 10px;
        }
        .stTabs [aria-selected="true"] {
            background-color: #4c78a8;
            color: white;
        }
        h1, h2, h3 {
            color: #1e3a8a;
        }
        .stSidebar .sidebar-content {
            background-color: #f8fafc;
        }
        div.stButton > button:first-child {
            background-color: #4c78a8;
            color: white;
            border-radius: 5px;
        }
        div.stButton > button:hover {
            background-color: #2c5282;
            color: white;
        }
        .metric-card {
            background-color: #ffffff;
            border-radius: 10px;
            padding: 15px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 15px;
        }
        .important-metric {
            font-size: 24px;
            font-weight: bold;
            color: #1e3a8a;
        }
        .card-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 10px;
        }
    </style>
    """, unsafe_allow_html=True)

# Load the custom CSS
load_css()

# Navigation
def create_navigation():
    with st.sidebar:
        selected = option_menu(
            menu_title="Navigation",
            options=[
                "Profile", 
                "Sentiment Analysis", 
                "Behavior Prediction", 
                "Campaign Effectiveness", 
                "Impact Simulator",
                "Decision Support System"
            ],
            icons=[
                "person-circle", 
                "graph-up", 
                "people-fill", 
                "megaphone-fill", 
                "clipboard-data", 
                "gear-fill"
            ],
            menu_icon="cast",
            default_index=0,
        )
    return selected

# Profile page
def show_profile():
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Use local image file instead of GitHub avatar
        try:
            # Try to open the image file
            image = Image.open("DSC_0244.jpg")
            st.image(image, width=300)
        except Exception as e:
            st.error(f"Error loading image: {e}")
            # Fallback to GitHub avatar if local image fails
            st.image("https://avatars.githubusercontent.com/u/96188673", width=250)
            
        st.markdown("### Contact")
        st.markdown("📞 +91 9410020324")
        st.markdown("✉️ [anirudhdevs91@gmail.com](mailto:anirudhdevs91@gmail.com)")
        st.markdown("🔗 [LinkedIn](https://www.linkedin.com/in/anirudh06dev/)")
        st.markdown("💻 [GitHub](https://github.com/devilanirudh)")
        
    with col2:
        st.title("Anirudh Dev")
        st.subheader("Backend AI Engineer | Data Scientist")
        st.markdown("""
        I specialize in leveraging advanced AI and data science techniques to drive data-informed decision-making across industries. My expertise spans from developing robust backend systems to creating innovative data analysis solutions that provide actionable insights.
        """)
        
        st.markdown("### Experience")
        st.markdown("""
        **Prodloop | Backend AI Engineer**  
        *December 2024 - Present*
        - Engineered backend solutions to support conversational AI systems for real-time analytics
        - Designed scalable production-grade systems enhancing model performance for data processing
        - Developed data pipelines for processing sentiment and behavioral data
        
        **Outlier.ai | RHLF AI Trainer**  
        *December 2024 - February 2025*
        - Reduced user misunderstanding rates from 42% to below 10% through optimized conversation flows
        - Enhanced AI understanding across 1,000+ unique interactions related to complex topics
        """)
    
    st.markdown("### Technical Skills")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **Data Processing & Analysis**
        - Python & Data Science Libraries
        - Large-scale Data Processing
        - Real-time Analytics Pipelines
        """)
    
    with col2:
        st.markdown("""
        **AI & Machine Learning**
        - Sentiment Analysis
        - Predictive Modeling
        - Natural Language Processing
        """)
    
    with col3:
        st.markdown("""
        **Infrastructure & Deployment**
        - Scalable Cloud Solutions (GCP)
        - Database Design & Optimization
        - RESTful APIs & GraphQL
        """)
    
    st.markdown("### Key Projects")
    
    with st.expander("Agentic RAG System"):
        st.markdown("""
        **Technologies**: Python, LangChain, FastAPI, GCP
        
        **Applications**:
        - Developed an AI system capable of analyzing vast amounts of documents with 85% better relevance scores
        - Implemented secure, scalable infrastructure handling 500+ queries/minute with 99.9% uptime
        - Created context-aware response systems for complex discourse analysis
        """)
        st.markdown("[View GitHub Repository](https://github.com/devilanirudh/agno_agentic_rag)")
    
    with st.expander("TechConnect Hub Realtime"):
        st.markdown("""
        **Technologies**: Django, PostgreSQL, JWT, GCP
        
        **Applications**:
        - Built real-time collaboration platform adaptable for teams and analysts
        - Implemented secure authentication and file storage systems critical for sensitive data
        - Developed comprehensive project management features for coordinating campaigns and initiatives
        """)
        st.markdown("[View GitHub Repository](https://github.com/devilanirudh/Tech_connect_hub_realtime)")

# Main function to generate sample political data
def generate_sample_data():
    # Sample data for demonstration purposes
    np.random.seed(42)
    dates = pd.date_range(start='1/1/2024', periods=100)
    
    # Party approval ratings
    party_a_approval = 40 + np.cumsum(np.random.normal(0, 1, 100)) % 15
    party_b_approval = 45 + np.cumsum(np.random.normal(0, 1, 100)) % 10
    
    approval_df = pd.DataFrame({
        'Date': dates,
        'Party A': party_a_approval,
        'Party B': party_b_approval
    })
    
    # Voter demographics
    age_groups = ['18-24', '25-34', '35-44', '45-54', '55-64', '65+']
    party_a_demo = [15, 25, 35, 45, 55, 60]
    party_b_demo = [55, 45, 40, 35, 25, 20]
    
    demo_df = pd.DataFrame({
        'Age Group': age_groups,
        'Party A': party_a_demo,
        'Party B': party_b_demo
    })
    
    # Sample tweets for sentiment analysis
    sample_tweets = [
        "The new healthcare policy will greatly benefit our citizens.",
        "This tax plan is terrible and will hurt the middle class.",
        "I'm impressed by the education reform proposed yesterday.",
        "The infrastructure bill is a complete disaster.",
        "Our community strongly supports the new environmental regulations.",
        "The foreign policy decisions have been catastrophic.",
        "Economic growth is strong under the current administration.",
        "Rising unemployment is a major concern for voters.",
        "The latest polling suggests a tight race in the upcoming election.",
        "Campaign promises are rarely kept after the election."
    ]
    
    # Policy impact data
    policies = ['Healthcare Reform', 'Tax Plan', 'Education Initiative', 
                'Infrastructure Bill', 'Environmental Regulations']
    
    metrics = ['Public Approval', 'Economic Impact', 'Implementation Feasibility', 
               'Long-term Benefit', 'Political Cost']
    
    policy_data = {}
    for policy in policies:
        policy_data[policy] = {metric: np.random.randint(30, 90) for metric in metrics}
    
    return approval_df, demo_df, sample_tweets, policy_data

# Political Sentiment Analysis page
def show_sentiment_analysis():
    st.title("Sentiment Analysis")
    st.markdown("""
    This module demonstrates how AI can analyze public sentiment toward brands, products, and services.
    Using natural language processing and sentiment analysis techniques, we can gauge customer opinion from various sources.
    """)
    
    # Demo tabs
    tab1, tab2 = st.tabs(["Message Analysis", "Trend Visualization"])
    
    with tab1:
        st.subheader("Social Media Sentiment Analyzer")
        
        # Sample data
        _, _, sample_tweets, _ = generate_sample_data()
        
        # Update sample messages to be technology-focused
        sample_messages = [
            "The new product features will greatly benefit our customers.",
            "This pricing plan is terrible and will hurt small businesses.",
            "I'm impressed by the software update released yesterday.",
            "The website redesign is a complete disaster.",
            "Our community strongly supports the new sustainability initiatives.",
            "The customer service experience has been catastrophic.",
            "Performance is strong under the current infrastructure.",
            "Rising costs are a major concern for users.",
            "The latest survey suggests high satisfaction with the platform.",
            "Company promises are rarely kept after the product launch."
        ]
        
        # Input area
        user_input = st.text_area("Enter a statement to analyze:", 
                                  sample_messages[0], height=100)
        
        # Analyze button
        if st.button("Analyze Sentiment"):
            with st.spinner("Analyzing sentiment..."):
                # Simulate processing time
                time.sleep(1)
                
                # Sentiment analysis
                sid = SentimentIntensityAnalyzer()
                sentiment_scores = sid.polarity_scores(user_input)
                
                # Display results
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Positive", f"{sentiment_scores['pos']*100:.1f}%")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Neutral", f"{sentiment_scores['neu']*100:.1f}%")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col3:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Negative", f"{sentiment_scores['neg']*100:.1f}%")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Overall sentiment
                overall = "Positive" if sentiment_scores['compound'] > 0.05 else "Negative" if sentiment_scores['compound'] < -0.05 else "Neutral"
                st.markdown(f"### Overall Sentiment: **{overall}**")
                
                # Visualization
                fig = go.Figure(go.Bar(
                    x=['Positive', 'Neutral', 'Negative'],
                    y=[sentiment_scores['pos'], sentiment_scores['neu'], sentiment_scores['neg']],
                    marker_color=['#4CAF50', '#FFC107', '#F44336']
                ))
                fig.update_layout(
                    title="Sentiment Distribution",
                    xaxis_title="Sentiment Category",
                    yaxis_title="Score",
                    template="plotly_white"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Suggested actions based on sentiment
                st.subheader("Strategic Insights")
                if overall == "Positive":
                    st.success("This message resonates positively. Consider amplifying this narrative in marketing communications.")
                elif overall == "Negative":
                    st.error("This message evokes negative sentiment. Consider addressing concerns or reframing the narrative.")
                else:
                    st.info("This message has a neutral reception. Consider adding more compelling elements to increase engagement.")
        
        # Sample analysis
        st.subheader("Batch Sentiment Analysis")
        if st.button("Analyze Sample Statements"):
            with st.spinner("Processing multiple statements..."):
                # Simulate processing
                time.sleep(2)
                
                # Analyze sample messages
                results = []
                for message in sample_messages:
                    scores = sid.polarity_scores(message)
                    sentiment = "Positive" if scores['compound'] > 0.05 else "Negative" if scores['compound'] < -0.05 else "Neutral"
                    results.append({
                        'Statement': message,
                        'Sentiment': sentiment,
                        'Compound Score': scores['compound']
                    })
                
                # Create DataFrame and display
                results_df = pd.DataFrame(results)
                st.dataframe(results_df)
                
                # Count sentiments
                sentiment_counts = results_df['Sentiment'].value_counts()
                
                # Pie chart
                fig = px.pie(
                    names=sentiment_counts.index,
                    values=sentiment_counts.values,
                    title="Overall Sentiment Distribution",
                    color=sentiment_counts.index,
                    color_discrete_map={'Positive': '#4CAF50', 'Neutral': '#FFC107', 'Negative': '#F44336'}
                )
                st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Sentiment Trends")
        
        # Generate time-series sentiment data
        np.random.seed(42)
        dates = pd.date_range(start='1/1/2024', periods=60)
        
        product_a_sentiment = 0.2 + np.cumsum(np.random.normal(0, 0.05, 60)) % 0.5
        product_b_sentiment = 0.1 + np.cumsum(np.random.normal(0, 0.05, 60)) % 0.6
        
        trend_df = pd.DataFrame({
            'Date': dates,
            'Product A': product_a_sentiment,
            'Product B': product_b_sentiment
        })
        
        # Plot trends
        fig = px.line(
            trend_df, x='Date', 
            y=['Product A', 'Product B'],
            title="Sentiment Trends for Key Products",
            labels={'value': 'Sentiment Score', 'variable': 'Product'},
            color_discrete_sequence=['#4c78a8', '#f58518']
        )
        
        # Add events
        events = {
            '2024-01-15': 'Product Launch',
            '2024-02-01': 'Media Coverage',
            '2024-02-15': 'Competitor Response',
            '2024-03-01': 'Marketing Campaign'
        }
        
        for date, event in events.items():
            fig.add_vline(x=date, line_dash="dash", line_color="gray")
            fig.add_annotation(x=date, y=1.05, text=event, showarrow=False, yref="paper")
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Word cloud from key terms
        st.subheader("Key Terms in Customer Feedback")
        
        business_terms = {
            "innovation": 85, "quality": 78, "service": 65, "value": 90,
            "technology": 72, "sustainability": 68, "efficiency": 80,
            "reliability": 75, "performance": 88, "cost": 70, "design": 85,
            "features": 65, "support": 60, "interface": 75, "experience": 55
        }
        
        # Create and display wordcloud
        wordcloud = WordCloud(
            width=800, height=400,
            background_color='white',
            colormap='viridis',
            max_words=100
        ).generate_from_frequencies(business_terms)
        
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)
        
        # Call to action
        st.info("This analysis demonstrates how sentiment tracking can inform product development, marketing strategy, and customer service improvements. The system can be customized to monitor specific brands, products, or features.")

# Generate synthetic user data for prediction model
def generate_user_data(n_samples=1000):
    np.random.seed(42)
    
    # Define demographic features
    age = np.random.normal(45, 15, n_samples).clip(18, 90).astype(int)
    income_levels = ['Low', 'Medium', 'High']
    income = np.random.choice(income_levels, n_samples, p=[0.3, 0.5, 0.2])
    
    education_levels = ['High School', 'Bachelor', 'Master', 'PhD']
    education = np.random.choice(education_levels, n_samples, p=[0.4, 0.4, 0.15, 0.05])
    
    regions = ['Urban', 'Suburban', 'Rural']
    region = np.random.choice(regions, n_samples, p=[0.4, 0.35, 0.25])
    
    # Previous engagement history (0: didn't engage, 1: engaged)
    prev_engagement = np.random.binomial(1, 0.7, n_samples)
    
    # Feature importance ratings (0-10 scale)
    feature_a_importance = np.random.randint(0, 11, n_samples)
    feature_b_importance = np.random.randint(0, 11, n_samples)
    feature_c_importance = np.random.randint(0, 11, n_samples)
    
    # Synthetic relationships for engagement likelihood
    # Age effect
    age_effect = -((age - 35) ** 2) / 400
    
    # Education effect
    edu_map = {'High School': 0, 'Bachelor': 1, 'Master': 2, 'PhD': 3}
    education_num = np.array([edu_map[e] for e in education])
    education_effect = education_num * 0.4
    
    # Region effect
    region_map = {'Urban': 0.2, 'Suburban': 0.3, 'Rural': 0.1}
    region_effect = np.array([region_map[r] for r in region])
    
    # History effect
    history_effect = prev_engagement * 1.5
    
    # Feature preference effects
    feature_effect = (feature_a_importance + feature_b_importance + feature_c_importance) / 30
    
    # Combine effects with some randomness
    engagement_prob = 0.3 + age_effect + education_effect + region_effect + history_effect + feature_effect
    engagement_prob = np.clip(engagement_prob, 0.01, 0.99)
    
    # Generate target (will engage or not)
    will_engage = np.random.binomial(1, engagement_prob)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Age': age,
        'Income': income,
        'Education': education,
        'Region': region,
        'PreviousEngagement': prev_engagement,
        'FeatureA_Importance': feature_a_importance,
        'FeatureB_Importance': feature_b_importance,
        'FeatureC_Importance': feature_c_importance,
        'WillEngage': will_engage
    })
    
    return df

# Function to train engagement prediction model
def train_engagement_model(df):
    # Prepare features and target
    X = df.drop('WillEngage', axis=1)
    y = df['WillEngage']
    
    # Encode categorical features
    categorical_cols = ['Income', 'Education', 'Region']
    X_processed = X.copy()
    
    for col in categorical_cols:
        le = LabelEncoder()
        X_processed[col] = le.fit_transform(X_processed[col])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.3, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': X_processed.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    return model, accuracy, feature_importance, cm, X_test, y_test, y_pred

# Behavior Prediction page
def show_behavior_prediction():
    st.title("User Behavior Prediction")
    st.markdown("""
    This module demonstrates the use of machine learning to predict user engagement and behavior.
    We use demographic data, past patterns, and feature preferences to identify likely engagement patterns
    and optimize marketing and product development efforts.
    """)
    
    # Tabs for different prediction features
    tab1, tab2, tab3 = st.tabs(["Engagement Model", "Geographic Analysis", "User Simulator"])
    
    with tab1:
        st.subheader("Predictive Model: User Engagement")
        
        # Generate data and train model
        if st.button("Generate & Train Behavior Prediction Model"):
            with st.spinner("Generating user data and training model..."):
                # Simulate processing time
                time.sleep(2)
                
                # Generate data
                user_df = generate_user_data(1500)
                
                # Train model
                model, accuracy, feature_imp, conf_matrix, X_test, y_test, y_pred = train_engagement_model(user_df)
                
                # Display results
                st.success(f"Model trained successfully with {accuracy*100:.2f}% accuracy!")
                
                # Show sample data
                st.subheader("Sample User Data")
                st.dataframe(user_df.head())
                
                # Feature importance
                st.subheader("Feature Importance")
                fig = px.bar(
                    feature_imp, x='Importance', y='Feature',
                    orientation='h',
                    title="Factors Influencing User Engagement",
                    color='Importance',
                    color_continuous_scale='blues'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Confusion Matrix
                st.subheader("Prediction Accuracy")
                labels = ['Will Not Engage', 'Will Engage']
                cm_fig = ff.create_annotated_heatmap(
                    z=conf_matrix, 
                    x=labels, 
                    y=labels, 
                    colorscale='Blues',
                    showscale=True
                )
                cm_fig.update_layout(
                    title="Confusion Matrix",
                    xaxis_title="Predicted",
                    yaxis_title="Actual"
                )
                st.plotly_chart(cm_fig, use_container_width=True)
                
                # Save important variables in session state
                st.session_state['engagement_model'] = model
                st.session_state['user_df'] = user_df
                
                # Strategic insights
                st.subheader("Strategic Insights")
                st.markdown("""
                Based on the prediction model, here are key strategic recommendations:
                
                1. **Priority Segments**: Focus outreach on demographics with high potential but lower current engagement
                2. **Resource Allocation**: Optimize marketing and product resources based on predicted engagement rates by region
                3. **Feature Development**: Emphasize product features that resonate with key user segments
                """)
                
                # User conversion impact
                conversion_impact = pd.DataFrame({
                    'User Group': ['Low Engagement', 'Occasional Users', 'Power Users'],
                    'Conversion Rate': [0.15, 0.35, 0.05],
                    'Business Impact': [0.40, 0.40, 0.20]
                })
                
                impact_fig = px.scatter(
                    conversion_impact,
                    x='Conversion Rate',
                    y='Business Impact',
                    size=[40, 60, 30],
                    color='User Group',
                    text='User Group',
                    title="Resource Allocation Strategy Matrix"
                )
                impact_fig.update_traces(textposition='top center')
                impact_fig.update_layout(
                    xaxis_title="Conversion Rate (% of users persuaded)",
                    yaxis_title="Business Impact (contribution to growth)"
                )
                st.plotly_chart(impact_fig, use_container_width=True)
    
    with tab2:
        st.subheader("Geographic Engagement Analysis")
        
        # Create synthetic regional data
        np.random.seed(42)
        regions = [f"Region {i}" for i in range(1, 11)]
        urban_rural = [np.random.choice(['Urban', 'Suburban', 'Rural'], p=[0.4, 0.3, 0.3]) for _ in range(10)]
        population = np.random.randint(50000, 200000, 10)
        past_engagement = np.random.uniform(0.4, 0.7, 10)
        pred_engagement = past_engagement + np.random.normal(0.05, 0.03, 10)
        pred_engagement = np.clip(pred_engagement, 0.4, 0.8)
        
        region_df = pd.DataFrame({
            'Region': regions,
            'Type': urban_rural,
            'Population': population,
            'Past Engagement': past_engagement,
            'Predicted Engagement': pred_engagement,
            'Growth Potential': np.random.uniform(0.05, 0.25, 10)
        })
        
        # Display region data
        st.dataframe(region_df)
        
        # Engagement map visualization
        st.subheader("Regional Engagement Projections")
        region_fig = px.bar(
            region_df,
            x='Region',
            y=['Past Engagement', 'Predicted Engagement'],
            barmode='group',
            title="User Engagement by Region",
            color_discrete_sequence=['#b3cde3', '#4c78a8']
        )
        st.plotly_chart(region_fig, use_container_width=True)
        
        # Engagement opportunity analysis
        st.subheader("Growth Opportunity Analysis")
        
        # Calculate additional metrics
        region_df['Engagement Improvement'] = region_df['Predicted Engagement'] - region_df['Past Engagement']
        region_df['Potential Users'] = region_df['Population'] * region_df['Growth Potential']
        
        # Show opportunity scatter plot
        opportunity_fig = px.scatter(
            region_df,
            x='Growth Potential',
            y='Engagement Improvement',
            size='Population',
            color='Type',
            hover_name='Region',
            text='Region',
            size_max=60,
            title="Opportunity Matrix by Region"
        )
        opportunity_fig.update_traces(textposition='top center')
        opportunity_fig.update_layout(
            xaxis_title="Growth Potential (% of potential new users)",
            yaxis_title="Projected Engagement Improvement (%)"
        )
        st.plotly_chart(opportunity_fig, use_container_width=True)
        
        # Strategic recommendations
        st.markdown("### Resource Allocation Recommendations")
        
        # Calculate priority score
        region_df['Priority Score'] = (region_df['Growth Potential'] * 
                                      region_df['Population'] * 
                                      (1 + region_df['Engagement Improvement']))
        
        # Sort by priority score
        priority_regions = region_df.sort_values('Priority Score', ascending=False)[['Region', 'Type', 'Priority Score']]
        
        # Display top priority regions
        st.markdown("#### Priority Regions for Marketing and Development")
        st.dataframe(priority_regions)
        
        # Resource allocation pie chart
        total_score = priority_regions['Priority Score'].sum()
        priority_regions['Resource Allocation'] = (priority_regions['Priority Score'] / total_score * 100).round(1)
        
        allocation_fig = px.pie(
            priority_regions,
            values='Resource Allocation',
            names='Region',
            title="Recommended Resource Allocation (%)",
            color_discrete_sequence=px.colors.sequential.Blues_r
        )
        st.plotly_chart(allocation_fig, use_container_width=True)
        
    with tab3:
        st.subheader("Interactive User Profile Simulator")
        st.markdown("""
        This tool allows you to simulate different user profiles and see the predicted likelihood of engagement.
        Use this to understand how different factors influence user behavior and to identify target demographics.
        """)
        
        # Check if model exists in session state
        if 'engagement_model' not in st.session_state:
            st.warning("Please train the model first by clicking the 'Generate & Train Behavior Prediction Model' button in the Engagement Model tab.")
        else:
            # Create interactive user profile
            col1, col2 = st.columns(2)
            
            with col1:
                age = st.slider("Age", 18, 90, 35)
                income = st.selectbox("Income Level", ['Low', 'Medium', 'High'])
                education = st.selectbox("Education Level", ['High School', 'Bachelor', 'Master', 'PhD'])
                region = st.selectbox("Region", ['Urban', 'Suburban', 'Rural'])
            
            with col2:
                prev_engagement = st.radio("Previously Engaged with Product", ['No', 'Yes'])
                prev_engagement = 1 if prev_engagement == 'Yes' else 0
                
                feature_a = st.slider("Feature A Importance", 0, 10, 5)
                feature_b = st.slider("Feature B Importance", 0, 10, 5)
                feature_c = st.slider("Feature C Importance", 0, 10, 5)
            
            # Create test profile
            if st.button("Predict Engagement Likelihood"):
                with st.spinner("Analyzing user profile..."):
                    # Simulate processing
                    time.sleep(1)
                    
                    # Create profile DataFrame
                    profile = pd.DataFrame({
                        'Age': [age],
                        'Income': [income],
                        'Education': [education],
                        'Region': [region],
                        'PreviousEngagement': [prev_engagement],
                        'FeatureA_Importance': [feature_a],
                        'FeatureB_Importance': [feature_b],
                        'FeatureC_Importance': [feature_c]
                    })
                    
                    # Prepare data for prediction (encode categorical variables)
                    categorical_cols = ['Income', 'Education', 'Region']
                    profile_processed = profile.copy()
                    
                    # Get the original dataframe from session state
                    train_df = st.session_state['user_df']
                    
                    # Encode each categorical column using the training data
                    for col in categorical_cols:
                        le = LabelEncoder()
                        le.fit(train_df[col])
                        profile_processed[col] = le.transform(profile_processed[col])
                    
                    # Make prediction
                    model = st.session_state['engagement_model']
                    engagement_prob = model.predict_proba(profile_processed)[0][1]
                    
                    # Display prediction
                    st.markdown("### Prediction Results")
                    
                    # Gauge chart for engagement probability
                    gauge_fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=engagement_prob * 100,
                        title={'text': "Likelihood to Engage (%)"},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "#4c78a8"},
                            'steps': [
                                {'range': [0, 30], 'color': "#f8d7da"},
                                {'range': [30, 70], 'color': "#fff3cd"},
                                {'range': [70, 100], 'color': "#d1e7dd"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 50
                            }
                        }
                    ))
                    
                    gauge_fig.update_layout(
                        height=300,
                        margin=dict(l=10, r=10, t=50, b=10),
                    )
                    
                    st.plotly_chart(gauge_fig, use_container_width=True)
                    
                    # Strategic insights based on profile
                    st.subheader("Engagement Strategy")
                    
                    if engagement_prob > 0.7:
                        st.success("""
                        **High Propensity User**
                        
                        This user profile has a high likelihood of engagement. Recommended strategy:
                        - Focus on loyalty programs and premium features
                        - Encourage them to become product advocates
                        - Provide early access to new features and updates
                        """)
                    elif engagement_prob > 0.4:
                        st.warning("""
                        **Moderate Propensity User**
                        
                        This user profile has a moderate likelihood of engagement. Recommended strategy:
                        - Target with more intensive onboarding and educational content
                        - Focus messaging on features they care most about
                        - Address potential barriers to engagement
                        - Provide special offers and incentives
                        """)
                    else:
                        st.error("""
                        **Low Propensity User**
                        
                        This user profile has a low likelihood of engagement. Recommended strategy:
                        - Determine if resources should be allocated to this demographic
                        - If targeting, use very personalized outreach focused on immediate value
                        - Address skepticism about product benefits
                        - Provide comprehensive assistance and simplified entry points
                        """)
                    
                    # Similar profiles analysis
                    st.subheader("Demographic Segment Analysis")
                    
                    # Get comparable users from the dataset
                    user_df = st.session_state['user_df']
                    
                    # Filter for similar age group
                    age_group_min = max(18, age - 10)
                    age_group_max = min(90, age + 10)
                    similar_age = user_df[(user_df['Age'] >= age_group_min) & (user_df['Age'] <= age_group_max)]
                    
                    # Filter for same region
                    similar_region = similar_age[similar_age['Region'] == region]
                    
                    # Calculate engagement rate for this demographic
                    demo_engagement_rate = similar_region['WillEngage'].mean() * 100
                    
                    st.markdown(f"**Similar Demographic Engagement Rate**: {demo_engagement_rate:.1f}%")
                    
                    # Compare to overall population
                    overall_rate = user_df['WillEngage'].mean() * 100
                    diff = demo_engagement_rate - overall_rate
                    
                    if diff > 5:
                        st.markdown(f"This demographic engages at a rate **{diff:.1f}% higher** than the overall population.")
                    elif diff < -5:
                        st.markdown(f"This demographic engages at a rate **{abs(diff):.1f}% lower** than the overall population.")
                    else:
                        st.markdown("This demographic engages at a rate similar to the overall population.")
                    
                    # Effectiveness of different outreach methods for this demographic
                    st.subheader("Recommended Outreach Channels")
                    
                    # Simulate effectiveness scores based on demographic profile
                    # In a real application, these would be based on empirical data
                    outreach_methods = [
                        'Email Campaigns', 'Social Media Ads', 'Push Notifications', 
                        'In-App Messages', 'Content Marketing', 'Direct Sales', 
                        'Webinars/Events', 'Video Tutorials', 'User Communities'
                    ]
                    
                    # Adjust effectiveness based on age
                    effectiveness = []
                    if age < 30:
                        effectiveness = [40, 85, 80, 75, 65, 30, 50, 70, 75]
                    elif age < 50:
                        effectiveness = [65, 70, 60, 65, 75, 60, 70, 60, 55]
                    else:
                        effectiveness = [75, 50, 35, 45, 60, 70, 65, 55, 40]
                    
                    # Further adjust based on region
                    if region == 'Rural':
                        # Boost direct contact and email for rural
                        for i, method in enumerate(outreach_methods):
                            if method in ['Email Campaigns', 'Direct Sales', 'Webinars/Events']:
                                effectiveness[i] = min(effectiveness[i] + 15, 100)
                            elif method in ['Social Media Ads', 'Push Notifications']:
                                effectiveness[i] = max(effectiveness[i] - 10, 10)
                    
                    outreach_df = pd.DataFrame({
                        'Method': outreach_methods,
                        'Effectiveness': effectiveness
                    }).sort_values('Effectiveness', ascending=False)
                    
                    # Bar chart of outreach effectiveness
                    outreach_fig = px.bar(
                        outreach_df,
                        x='Effectiveness',
                        y='Method',
                        orientation='h',
                        title="Outreach Method Effectiveness for This Demographic",
                        color='Effectiveness',
                        color_continuous_scale='blues'
                    )
                    outreach_fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(outreach_fig, use_container_width=True)

# Policy Impact Simulator page
def show_policy_simulator():
    st.title("Policy Impact Simulator")
    st.markdown("""
    This module demonstrates how AI can simulate the potential impact of policy decisions across multiple dimensions.
    Using data modeling and predictive analytics, we can forecast how policy changes might affect various metrics
    including public opinion, economic factors, and electoral outcomes.
    """)
    
    # Create tabs for different simulation features
    tab1, tab2 = st.tabs(["Policy Impact Analysis", "Comparative Scenario Planner"])
    
    with tab1:
        st.subheader("Policy Impact Analyzer")
        
        # Sample policies
        policies = [
            "Universal Healthcare Program",
            "Tax Reform Package",
            "Infrastructure Investment Plan",
            "Climate Action Initiative",
            "Education Funding Increase"
        ]
        
        # Select policy
        selected_policy = st.selectbox("Select Policy to Analyze", policies)
        
        # Policy parameters based on selection
        if selected_policy == "Universal Healthcare Program":
            col1, col2 = st.columns(2)
            with col1:
                coverage_level = st.slider("Coverage Level (%)", 50, 100, 85)
                implementation_timeline = st.slider("Implementation Timeline (Years)", 1, 10, 3)
            with col2:
                tax_increase = st.slider("Associated Tax Change (%)", -5.0, 5.0, 1.5, 0.1)
                private_option = st.checkbox("Include Private Insurance Option", True)
                
            parameters = {
                "Coverage": coverage_level,
                "Timeline": implementation_timeline,
                "Tax Change": tax_increase,
                "Private Option": private_option
            }
            
        elif selected_policy == "Tax Reform Package":
            col1, col2 = st.columns(2)
            with col1:
                corp_tax_change = st.slider("Corporate Tax Change (%)", -10.0, 10.0, -3.0, 0.5)
                income_tax_top = st.slider("Top Income Bracket Change (%)", -10.0, 10.0, 2.5, 0.5)
            with col2:
                income_tax_middle = st.slider("Middle Income Bracket Change (%)", -10.0, 10.0, -1.0, 0.5)
                deduction_cap = st.checkbox("Cap Deductions", True)
                
            parameters = {
                "Corporate Tax": corp_tax_change,
                "Top Bracket": income_tax_top,
                "Middle Bracket": income_tax_middle,
                "Deduction Cap": deduction_cap
            }
            
        elif selected_policy == "Infrastructure Investment Plan":
            col1, col2 = st.columns(2)
            with col1:
                investment_amount = st.slider("Investment Size ($ Billions)", 100, 2000, 500, 50)
                timeframe = st.slider("Project Timeframe (Years)", 1, 15, 5)
            with col2:
                funding_method = st.selectbox("Primary Funding Method", ["Deficit Spending", "Tax Increase", "Public-Private Partnership"])
                green_focus = st.slider("Green Infrastructure Focus (%)", 0, 100, 40)
                
            parameters = {
                "Investment": investment_amount,
                "Timeframe": timeframe,
                "Funding": funding_method,
                "Green Focus": green_focus
            }
            
        else:
            # Default parameters for other policies
            col1, col2 = st.columns(2)
            with col1:
                budget_change = st.slider("Budget Change (%)", -20.0, 50.0, 15.0, 5.0)
                implementation_speed = st.slider("Implementation Speed", 1, 10, 5)
            with col2:
                public_support = st.slider("Current Public Support (%)", 0, 100, 45)
                partisan_approach = st.selectbox("Partisan Approach", ["Bipartisan", "Party-line", "Executive Action"])
                
            parameters = {
                "Budget": budget_change,
                "Speed": implementation_speed,
                "Support": public_support,
                "Approach": partisan_approach
            }
        
        # Run simulation button
        if st.button("Run Impact Simulation"):
            with st.spinner("Simulating policy impacts..."):
                # Simulate processing time
                time.sleep(2)
                
                # Display results
                st.success(f"Impact simulation completed for {selected_policy}!")
                
                # Simulated impact metrics
                st.subheader("Projected Impact Metrics")
                
                # Create impact scores based on policy parameters
                # In a real application, these would be calculated by sophisticated models
                
                # Define impact dimensions
                dimensions = [
                    "Public Approval", "Economic Growth", "Budget Impact", 
                    "Implementation Feasibility", "Long-term Sustainability"
                ]
                
                # Generate pseudo-random but somewhat logical impact scores based on parameters
                np.random.seed(hash(str(parameters)) % 10000)
                
                # Base scores
                base_scores = np.random.normal(50, 15, len(dimensions))
                
                # Adjust based on policy type and parameters
                if selected_policy == "Universal Healthcare Program":
                    # Higher coverage increases approval but decreases budget impact
                    base_scores[0] += (coverage_level - 75) * 0.3  # Public Approval
                    base_scores[2] -= (coverage_level - 75) * 0.4  # Budget Impact
                    
                    # Longer timeline improves feasibility but reduces approval
                    base_scores[3] += (implementation_timeline - 5) * 2  # Feasibility
                    base_scores[0] -= (implementation_timeline - 3) * 1.5  # Public Approval
                    
                    # Tax increases hurt approval and economic growth but help budget
                    base_scores[0] -= tax_increase * 3  # Public Approval
                    base_scores[1] -= tax_increase * 2  # Economic Growth
                    base_scores[2] += tax_increase * 4  # Budget Impact
                    
                    # Private option improves feasibility and economic growth
                    if private_option:
                        base_scores[1] += 5  # Economic Growth
                        base_scores[3] += 7  # Feasibility
                    
                elif selected_policy == "Tax Reform Package":
                    # Corporate tax cuts boost economic growth but hurt budget
                    base_scores[1] -= corp_tax_change * 2  # Economic Growth (negative change = positive impact)
                    base_scores[2] += corp_tax_change * 3  # Budget Impact (negative change = negative impact)
                    
                    # Top bracket increases help budget but may hurt approval
                    base_scores[2] += income_tax_top * 2  # Budget Impact
                    base_scores[0] -= income_tax_top * 1.5  # Public Approval
                    
                    # Middle bracket cuts boost approval but hurt budget
                    base_scores[0] -= income_tax_middle * 3  # Public Approval (negative change = positive impact)
                    base_scores[2] += income_tax_middle * 2.5  # Budget Impact (negative change = negative impact)
                
                elif selected_policy == "Infrastructure Investment Plan":
                    # Larger investments boost economic growth and approval but hurt budget
                    investment_factor = (investment_amount - 500) / 500
                    base_scores[0] += investment_factor * 10  # Public Approval
                    base_scores[1] += investment_factor * 15  # Economic Growth
                    base_scores[2] -= investment_factor * 20  # Budget Impact
                    
                    # Funding method effects
                    if funding_method == "Deficit Spending":
                        base_scores[2] -= 15  # Budget Impact
                        base_scores[4] -= 10  # Long-term Sustainability
                    elif funding_method == "Tax Increase":
                        base_scores[0] -= 10  # Public Approval
                        base_scores[2] += 15  # Budget Impact
                    else:  # Public-Private
                        base_scores[3] += 10  # Feasibility
                        base_scores[1] += 5   # Economic Growth
                    
                    # Green focus effects
                    green_factor = (green_focus - 50) / 50
                    base_scores[0] += green_factor * 10  # Public Approval
                    base_scores[4] += green_factor * 15  # Long-term Sustainability
                
                # General adjustments for other policies
                else:
                    # Budget changes
                    budget_factor = budget_change / 10
                    base_scores[0] += budget_factor * 3  # Public Approval
                    base_scores[2] -= budget_factor * 5  # Budget Impact
                    
                    # Implementation speed
                    speed_factor = (implementation_speed - 5) / 5
                    base_scores[3] -= speed_factor * 10  # Feasibility (faster = harder)
                    
                    # Public support helps feasibility
                    support_factor = (public_support - 50) / 10
                    base_scores[3] += support_factor * 5  # Feasibility
                    
                    # Partisan approach
                    if partisan_approach == "Bipartisan":
                        base_scores[3] += 15  # Feasibility
                        base_scores[4] += 10  # Long-term Sustainability
                    elif partisan_approach == "Executive Action":
                        base_scores[3] += 5   # Feasibility
                        base_scores[4] -= 15  # Long-term Sustainability
                
                # Clip all scores to 0-100 range
                impact_scores = np.clip(base_scores, 0, 100)
                
                # Create DataFrame for visualization
                impact_df = pd.DataFrame({
                    'Dimension': dimensions,
                    'Score': impact_scores
                })
                
                # Calculate overall viability score
                viability_score = np.average(impact_scores, weights=[0.25, 0.2, 0.2, 0.15, 0.2])
                
                # Display overall viability score with gauge
                gauge_fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=viability_score,
                    title={'text': "Overall Policy Viability Score"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#4c78a8"},
                        'steps': [
                            {'range': [0, 40], 'color': "#f8d7da"},
                            {'range': [40, 70], 'color': "#fff3cd"},
                            {'range': [70, 100], 'color': "#d1e7dd"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                
                gauge_fig.update_layout(
                    height=300,
                    margin=dict(l=10, r=10, t=50, b=10),
                )
                
                st.plotly_chart(gauge_fig, use_container_width=True)
                
                # Display dimensional breakdown
                st.subheader("Impact by Dimension")
                
                # Create radar chart
                categories = impact_df['Dimension'].tolist()
                fig = go.Figure()
                
                fig.add_trace(go.Scatterpolar(
                    r=impact_df['Score'].tolist(),
                    theta=categories,
                    fill='toself',
                    name=selected_policy,
                    line_color='#4c78a8'
                ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 100]
                        )
                    ),
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Bar chart with color coding
                bar_fig = px.bar(
                    impact_df,
                    x='Dimension',
                    y='Score',
                    color='Score',
                    color_continuous_scale='RdYlGn',
                    range_color=[0, 100],
                    title="Dimensional Impact Analysis"
                )
                
                # Add reference line at 50
                bar_fig.add_hline(
                    y=50,
                    line_dash="dash",
                    line_color="gray",
                    annotation_text="Threshold",
                    annotation_position="bottom right"
                )
                
                st.plotly_chart(bar_fig, use_container_width=True)
                
                # Strategic insights
                st.subheader("Strategic Analysis & Recommendations")
                
                # Calculate strengths and weaknesses
                strengths = impact_df[impact_df['Score'] >= 70].sort_values('Score', ascending=False)
                weaknesses = impact_df[impact_df['Score'] <= 40].sort_values('Score')
                
                # Display strengths
                if not strengths.empty:
                    st.markdown("#### Policy Strengths")
                    for _, row in strengths.iterrows():
                        st.success(f"**{row['Dimension']}** ({row['Score']:.1f}/100): This policy performs strongly in this dimension.")
                else:
                    st.info("No significant strengths identified (>70 points).")
                
                # Display weaknesses
                if not weaknesses.empty:
                    st.markdown("#### Policy Challenges")
                    for _, row in weaknesses.iterrows():
                        st.error(f"**{row['Dimension']}** ({row['Score']:.1f}/100): This dimension requires attention to improve overall viability.")
                else:
                    st.info("No significant challenges identified (<40 points).")
                
                # Overall recommendation
                st.markdown("#### Executive Summary")
                if viability_score >= 70:
                    st.success(f"""
                    This policy shows strong overall viability ({viability_score:.1f}/100) and is recommended for implementation.
                    Consider addressing minor challenges, but the policy is well-positioned for success.
                    """)
                elif viability_score >= 50:
                    st.warning(f"""
                    This policy shows moderate viability ({viability_score:.1f}/100) and would benefit from refinement.
                    Focus on addressing key challenges before full implementation.
                    """)
                else:
                    st.error(f"""
                    This policy shows below-threshold viability ({viability_score:.1f}/100) and significant revision is recommended.
                    Consider alternative approaches or substantial modification of the current proposal.
                    """)
                
                # Estimated electoral impact
                st.subheader("Projected Electoral Impact")
                
                # Simulate electoral impact
                base_support = 48  # Base support percentage
                approval_effect = (impact_df.loc[impact_df['Dimension'] == 'Public Approval', 'Score'].values[0] - 50) / 10
                economic_effect = (impact_df.loc[impact_df['Dimension'] == 'Economic Growth', 'Score'].values[0] - 50) / 15
                
                # Calculate support change
                support_change = approval_effect + economic_effect
                new_support = base_support + support_change
                
                # Create before/after chart
                support_data = pd.DataFrame({
                    'Stage': ['Before Policy', 'After Policy'],
                    'Support': [base_support, new_support]
                })
                
                support_fig = px.bar(
                    support_data,
                    x='Stage',
                    y='Support',
                    color='Stage',
                    color_discrete_map={
                        'Before Policy': '#b3cde3',
                        'After Policy': '#4c78a8' if new_support > base_support else '#e15759'
                    },
                    title="Projected Support Level Change",
                    text_auto='.1f'
                )
                
                support_fig.update_traces(texttemplate='%{y:.1f}%', textposition='outside')
                support_fig.update_layout(yaxis_range=[max(0, min(base_support, new_support) - 10), min(100, max(base_support, new_support) + 10)])
                
                st.plotly_chart(support_fig, use_container_width=True)
                
                # Support by demographic
                st.subheader("Support Change by Demographic")
                
                # Simulate demographic impacts
                demographics = ['18-29', '30-44', '45-64', '65+', 'Urban', 'Suburban', 'Rural', 'Low Income', 'Middle Income', 'High Income']
                
                # Base support by demographic
                base_supports = [55, 52, 46, 42, 58, 48, 38, 60, 50, 40]
                
                # Different policies affect demographics differently
                if selected_policy == "Universal Healthcare Program":
                    # Healthcare tends to affect older and lower income more
                    demographic_effects = [
                        support_change * 0.7,  # 18-29
                        support_change * 0.8,  # 30-44
                        support_change * 1.2,  # 45-64
                        support_change * 1.5,  # 65+
                        support_change * 1.0,  # Urban
                        support_change * 1.0,  # Suburban
                        support_change * 1.0,  # Rural
                        support_change * 1.5,  # Low Income
                        support_change * 1.2,  # Middle Income
                        support_change * 0.6   # High Income
                    ]
                elif selected_policy == "Tax Reform Package":
                    # Tax policy affects different income groups differently
                    demographic_effects = [
                        support_change * 0.9,  # 18-29
                        support_change * 1.0,  # 30-44
                        support_change * 1.1,  # 45-64
                        support_change * 1.0,  # 65+
                        support_change * 1.0,  # Urban
                        support_change * 1.1,  # Suburban
                        support_change * 0.9,  # Rural
                        support_change * 1.3,  # Low Income
                        support_change * 1.0,  # Middle Income
                        support_change * 0.7   # High Income
                    ]
                elif selected_policy == "Infrastructure Investment Plan":
                    # Infrastructure tends to affect rural and suburban areas more
                    demographic_effects = [
                        support_change * 0.9,  # 18-29
                        support_change * 1.0,  # 30-44
                        support_change * 1.0,  # 45-64
                        support_change * 1.1,  # 65+
                        support_change * 0.8,  # Urban
                        support_change * 1.2,  # Suburban
                        support_change * 1.5,  # Rural
                        support_change * 1.0,  # Low Income
                        support_change * 1.0,  # Middle Income
                        support_change * 1.0   # High Income
                    ]
                else:
                    # Default demographic effects
                    demographic_effects = [support_change] * len(demographics)
                
                # Calculate new support levels
                new_supports = [min(100, max(0, base + effect)) for base, effect in zip(base_supports, demographic_effects)]
                support_changes = [new - base for new, base in zip(new_supports, base_supports)]
                
                # Create DataFrame for visualization
                demo_df = pd.DataFrame({
                    'Demographic': demographics,
                    'Base Support': base_supports,
                    'New Support': new_supports,
                    'Change': support_changes
                })
                
                # Create interactive bar chart with change indicators
                demo_fig = go.Figure()
                
                # Add base support bars
                demo_fig.add_trace(go.Bar(
                    name='Base Support',
                    x=demo_df['Demographic'],
                    y=demo_df['Base Support'],
                    marker_color='#b3cde3'
                ))
                
                # Add change bars (positive or negative)
                for i, row in demo_df.iterrows():
                    if row['Change'] > 0:
                        demo_fig.add_trace(go.Bar(
                            name='Increase',
                            x=[row['Demographic']],
                            y=[row['Change']],
                            marker_color='#4c78a8',
                            base=row['Base Support'],
                            showlegend=i==0  # Only show legend for first item
                        ))
                    elif row['Change'] < 0:
                        demo_fig.add_trace(go.Bar(
                            name='Decrease',
                            x=[row['Demographic']],
                            y=[row['Change']],
                            marker_color='#e15759',
                            base=row['Base Support'],
                            showlegend=i==0  # Only show legend for first item
                        ))
                
                demo_fig.update_layout(
                    title="Support Change by Demographic Group",
                    barmode='relative',
                    yaxis_title="Support (%)",
                    yaxis=dict(range=[0, 100]),
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
                st.plotly_chart(demo_fig, use_container_width=True)
                
                # Key takeaways
                st.markdown("#### Targeting Recommendations")
                
                # Identify most positively and negatively affected groups
                most_positive = demo_df.nlargest(2, 'Change')
                most_negative = demo_df.nsmallest(2, 'Change')
                
                st.markdown("**Opportunity Groups:**")
                for _, row in most_positive.iterrows():
                    st.markdown(f"- **{row['Demographic']}**: +{row['Change']:.1f}% support increase")
                
                st.markdown("**Risk Groups:**")
                for _, row in most_negative.iterrows():
                    st.markdown(f"- **{row['Demographic']}**: {row['Change']:.1f}% support change")
    
    with tab2:
        st.subheader("Comparative Policy Scenario Planner")
        st.markdown("""
        This tool allows you to compare multiple policy approaches and examine trade-offs
        across different dimensions. Use it to evaluate alternative strategies and determine
        the optimal approach.
        """)
        
        # Policy type selection
        policy_area = st.selectbox(
            "Select Policy Area",
            ["Economic Policy", "Healthcare Policy", "Energy & Environment", "Education Reform"]
        )
        
        # Set up policy comparison based on selection
        if policy_area == "Economic Policy":
            policy_options = [
                "Progressive Tax Reform",
                "Corporate Tax Reduction",
                "Middle Class Tax Relief",
                "Balanced Approach"
            ]
            
            dimensions = [
                "Economic Growth", "Deficit Impact", "Income Inequality", 
                "Job Creation", "Public Approval", "Political Feasibility"
            ]
            
            # Pre-defined scores for each policy on each dimension
            scores = {
                "Progressive Tax Reform": [45, 70, 85, 55, 62, 40],
                "Corporate Tax Reduction": [75, 35, 30, 70, 48, 65],
                "Middle Class Tax Relief": [65, 45, 70, 60, 78, 60],
                "Balanced Approach": [60, 60, 55, 62, 68, 75]
            }
            
        elif policy_area == "Healthcare Policy":
            policy_options = [
                "Single Payer System",
                "Public Option",
                "Market-Based Reform",
                "Status Quo"
            ]
            
            dimensions = [
                "Coverage Expansion", "Cost Control", "Quality of Care", 
                "Budget Impact", "Implementation Complexity", "Public Approval"
            ]
            
            # Pre-defined scores for each policy on each dimension
            scores = {
                "Single Payer System": [95, 80, 65, 40, 25, 55],
                "Public Option": [75, 65, 70, 55, 55, 70],
                "Market-Based Reform": [45, 55, 65, 75, 70, 45],
                "Status Quo": [50, 30, 55, 70, 90, 35]
            }
            
        elif policy_area == "Energy & Environment":
            policy_options = [
                "Green New Deal",
                "Carbon Tax",
                "Regulatory Approach",
                "Market Incentives"
            ]
            
            dimensions = [
                "Emissions Reduction", "Economic Cost", "Innovation", 
                "Implementation Speed", "International Leadership", "Political Viability"
            ]
            
            # Pre-defined scores for each policy on each dimension
            scores = {
                "Green New Deal": [90, 35, 80, 45, 85, 40],
                "Carbon Tax": [75, 60, 75, 70, 80, 45],
                "Regulatory Approach": [65, 55, 50, 60, 60, 55],
                "Market Incentives": [60, 75, 85, 50, 55, 70]
            }
            
        else:  # Education Reform
            policy_options = [
                "Universal Pre-K",
                "School Choice Expansion",
                "Teacher Pay Increase",
                "Higher Ed Affordability"
            ]
            
            dimensions = [
                "Educational Outcomes", "Equity", "Cost", 
                "Implementation Timeline", "Teacher Support", "Parental Support"
            ]
            
            # Pre-defined scores for each policy on each dimension
            scores = {
                "Universal Pre-K": [80, 85, 40, 55, 75, 70],
                "School Choice Expansion": [65, 45, 70, 75, 30, 75],
                "Teacher Pay Increase": [65, 60, 45, 85, 95, 60],
                "Higher Ed Affordability": [70, 80, 35, 65, 70, 75]
            }
        
        # Policy selection
        col1, col2 = st.columns(2)
        
        with col1:
            selected_policy_1 = st.selectbox("Select First Policy Option", policy_options, index=0)
        
        with col2:
            remaining_options = [p for p in policy_options if p != selected_policy_1]
            selected_policy_2 = st.selectbox("Select Second Policy Option", remaining_options, index=0)
        
        # Compare button
        if st.button("Compare Policy Options"):
            with st.spinner("Generating comparative analysis..."):
                # Simulate processing time
                time.sleep(1.5)
                
                # Get scores for selected policies
                scores_1 = scores[selected_policy_1]
                scores_2 = scores[selected_policy_2]
                
                # Create DataFrame for visualization
                comparison_df = pd.DataFrame({
                    'Dimension': dimensions,
                    selected_policy_1: scores_1,
                    selected_policy_2: scores_2
                })
                
                # Calculate overall scores (simple average)
                overall_1 = np.mean(scores_1)
                overall_2 = np.mean(scores_2)
                
                # Display overall comparison
                st.subheader("Overall Policy Comparison")
                
                # Create overall score comparison
                overall_df = pd.DataFrame({
                    'Policy': [selected_policy_1, selected_policy_2],
                    'Overall Score': [overall_1, overall_2]
                })
                
                overall_fig = px.bar(
                    overall_df,
                    x='Policy',
                    y='Overall Score',
                    color='Policy',
                    text_auto='.1f',
                    title="Overall Policy Comparison",
                    color_discrete_sequence=['#4c78a8', '#f58518']
                )
                
                overall_fig.update_traces(textposition='outside')
                overall_fig.update_layout(yaxis_range=[0, 100])
                
                st.plotly_chart(overall_fig, use_container_width=True)
                
                # Radar chart comparison
                st.subheader("Dimensional Comparison")
                
                radar_fig = go.Figure()
                
                radar_fig.add_trace(go.Scatterpolar(
                    r=scores_1,
                    theta=dimensions,
                    fill='toself',
                    name=selected_policy_1,
                    line_color='#4c78a8'
                ))
                
                radar_fig.add_trace(go.Scatterpolar(
                    r=scores_2,
                    theta=dimensions,
                    fill='toself',
                    name=selected_policy_2,
                    line_color='#f58518'
                ))
                
                radar_fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 100]
                        )
                    ),
                    showlegend=True
                )
                
                st.plotly_chart(radar_fig, use_container_width=True)
                
                # Bar chart comparison by dimension
                st.subheader("Dimension-by-Dimension Comparison")
                
                # Prepare data for bar chart
                bar_data = []
                for i, dim in enumerate(dimensions):
                    bar_data.append({'Dimension': dim, 'Score': scores_1[i], 'Policy': selected_policy_1})
                    bar_data.append({'Dimension': dim, 'Score': scores_2[i], 'Policy': selected_policy_2})
                
                bar_df = pd.DataFrame(bar_data)
                
                bar_fig = px.bar(
                    bar_df,
                    x='Dimension',
                    y='Score',
                    color='Policy',
                    barmode='group',
                    title="Policy Performance by Dimension",
                    color_discrete_sequence=['#4c78a8', '#f58518']
                )
                
                bar_fig.update_layout(yaxis_range=[0, 100])
                
                st.plotly_chart(bar_fig, use_container_width=True)
                
                # Dimension-level difference analysis
                st.subheader("Key Differentiators")
                
                # Calculate differences
                comparison_df['Difference'] = comparison_df[selected_policy_1] - comparison_df[selected_policy_2]
                
                # Sort by absolute difference
                comparison_df['AbsDiff'] = comparison_df['Difference'].abs()
                sorted_diff = comparison_df.sort_values('AbsDiff', ascending=False)
                
                # Display top differentiators
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"**{selected_policy_1} Advantages:**")
                    advantages_1 = sorted_diff[sorted_diff['Difference'] > 10].head(3)
                    if not advantages_1.empty:
                        for _, row in advantages_1.iterrows():
                            st.success(f"""
                            **{row['Dimension']}**: +{row['Difference']:.0f} points  
                            {selected_policy_1}: {row[selected_policy_1]:.0f}/100  
                            {selected_policy_2}: {row[selected_policy_2]:.0f}/100
                            """)
                    else:
                        st.info("No significant advantages (>10 points).")
                
                with col2:
                    st.markdown(f"**{selected_policy_2} Advantages:**")
                    advantages_2 = sorted_diff[sorted_diff['Difference'] < -10].head(3)
                    if not advantages_2.empty:
                        for _, row in advantages_2.iterrows():
                            st.success(f"""
                            **{row['Dimension']}**: +{-row['Difference']:.0f} points  
                            {selected_policy_2}: {row[selected_policy_2]:.0f}/100  
                            {selected_policy_1}: {row[selected_policy_1]:.0f}/100
                            """)
                    else:
                        st.info("No significant advantages (>10 points).")
                
                # Strategic recommendation
                st.subheader("Strategic Recommendation")
                
                if overall_1 > overall_2 + 5:
                    recommendation = f"Based on this analysis, **{selected_policy_1}** is recommended as the superior approach with an overall advantage of {overall_1 - overall_2:.1f} points."
                elif overall_2 > overall_1 + 5:
                    recommendation = f"Based on this analysis, **{selected_policy_2}** is recommended as the superior approach with an overall advantage of {overall_2 - overall_1:.1f} points."
                else:
                    recommendation = f"Both **{selected_policy_1}** and **{selected_policy_2}** have comparable overall scores (within 5 points). Consider a hybrid approach that combines strengths of both options."
                
                st.markdown(recommendation)
                
                # Hybrid approach suggestion if scores are close
                if abs(overall_1 - overall_2) <= 10:
                    st.markdown("#### Potential Hybrid Approach")
                    
                    # Identify top dimensions for each policy
                    top_dims_1 = comparison_df.nlargest(3, selected_policy_1)[['Dimension', selected_policy_1]]
                    top_dims_1.columns = ['Dimension', 'Score']
                    top_dims_1['Policy'] = selected_policy_1
                    
                    top_dims_2 = comparison_df.nlargest(3, selected_policy_2)[['Dimension', selected_policy_2]]
                    top_dims_2.columns = ['Dimension', 'Score']
                    top_dims_2['Policy'] = selected_policy_2
                    
                    # Combine top dimensions from both policies
                    combined_tops = pd.concat([top_dims_1, top_dims_2])
                    combined_tops = combined_tops.sort_values('Score', ascending=False).drop_duplicates('Dimension')
                    
                    # Hybrid description
                    st.markdown("""
                    A hybrid approach could incorporate the strongest elements of both policies:
                    """)
                    
                    for _, row in combined_tops.head(4).iterrows():
                        st.markdown(f"- **{row['Dimension']}** component from **{row['Policy']}** ({row['Score']:.0f}/100)")
                    
                    # Estimate hybrid score (simple average of best dimensions)
                    hybrid_score = combined_tops['Score'].head(4).mean()
                    
                    st.success(f"Estimated hybrid approach overall score: **{hybrid_score:.1f}/100**")
                
                # Electoral impact
                st.subheader("Electoral Impact Comparison")
                
                # Simple model relating policy scores to electoral impact
                # Higher public approval and economic metrics contribute more to electoral success
                
                # Create weights for dimensions (simplified model)
                weights = {}
                if policy_area == "Economic Policy":
                    weights = {
                        "Economic Growth": 0.25,
                        "Job Creation": 0.2,
                        "Public Approval": 0.3,
                        "Income Inequality": 0.1,
                        "Deficit Impact": 0.05,
                        "Political Feasibility": 0.1
                    }
                elif policy_area == "Healthcare Policy":
                    weights = {
                        "Coverage Expansion": 0.15,
                        "Cost Control": 0.2,
                        "Quality of Care": 0.15,
                        "Budget Impact": 0.1,
                        "Implementation Complexity": 0.1,
                        "Public Approval": 0.3
                    }
                elif policy_area == "Energy & Environment":
                    weights = {
                        "Emissions Reduction": 0.15,
                        "Economic Cost": 0.2,
                        "Innovation": 0.15,
                        "Implementation Speed": 0.1,
                        "International Leadership": 0.05,
                        "Political Viability": 0.35
                    }
                else:  # Education
                    weights = {
                        "Educational Outcomes": 0.2,
                        "Equity": 0.15,
                        "Cost": 0.1,
                        "Implementation Timeline": 0.05,
                        "Teacher Support": 0.2,
                        "Parental Support": 0.3
                    }
                
                # Calculate weighted scores
                weighted_score_1 = sum(scores_1[i] * weights[dim] for i, dim in enumerate(dimensions))
                weighted_score_2 = sum(scores_2[i] * weights[dim] for i, dim in enumerate(dimensions))
                
                # Convert to electoral impact
                # Base support of 48%, with each 10 points of weighted score contributing 1% change
                electoral_impact_1 = 48 + (weighted_score_1 - 60) / 10
                electoral_impact_2 = 48 + (weighted_score_2 - 60) / 10
                
                # Create comparison dataframe
                electoral_df = pd.DataFrame({
                    'Policy': [selected_policy_1, selected_policy_2],
                    'Electoral Impact': [electoral_impact_1, electoral_impact_2],
                    'Weighted Score': [weighted_score_1, weighted_score_2]
                })
                
                # Create electoral impact chart
                electoral_fig = px.bar(
                    electoral_df,
                    x='Policy',
                    y='Electoral Impact',
                    color='Policy',
                    text_auto='.1f',
                    title="Projected Electoral Support (%)",
                    color_discrete_sequence=['#4c78a8', '#f58518']
                )
                
                electoral_fig.update_traces(texttemplate='%{y:.1f}%', textposition='outside')
                electoral_fig.update_layout(yaxis_range=[40, 55])
                
                st.plotly_chart(electoral_fig, use_container_width=True)
                
                # Final comparative insights
                st.markdown("#### Electoral Strategy Implications")
                
                if electoral_impact_1 > electoral_impact_2 + 1:
                    st.markdown(f"""
                    **{selected_policy_1}** is projected to generate stronger electoral support (+{electoral_impact_1 - electoral_impact_2:.1f}%).
                    This policy more effectively addresses the dimensions that voters weigh most heavily in their voting decisions.
                    """)
                elif electoral_impact_2 > electoral_impact_1 + 1:
                    st.markdown(f"""
                    **{selected_policy_2}** is projected to generate stronger electoral support (+{electoral_impact_2 - electoral_impact_1:.1f}%).
                    This policy more effectively addresses the dimensions that voters weigh most heavily in their voting decisions.
                    """)
                else:
                    st.markdown("""
                    Both policies are projected to have similar electoral impacts (within 1%).
                    The choice between them may depend more on ideological alignment, implementation considerations,
                    or long-term strategic goals rather than immediate electoral calculations.
                    """)

# Main application
def main():
    # Create navigation
    selected = create_navigation()
    
    # Render selected page
    if selected == "Profile":
        show_profile()
    elif selected == "Sentiment Analysis":
        show_sentiment_analysis()
    elif selected == "Behavior Prediction":
        show_behavior_prediction()
    elif selected == "Impact Simulator":
        show_policy_simulator()
    # Additional pages to be implemented in the next sections
    else:
        st.title(f"{selected}")
        st.info("This section is under development. More advanced features coming soon!")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "### Data-Driven Decision Hub | Created by Anirudh Dev | 2025",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
