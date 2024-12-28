import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import altair as alt

# Load Data
@st.cache_data
def load_data():
    data = pd.read_csv('cleaned_avocado_data.csv')
    data['Date'] = pd.to_datetime(data['Date'])
    return data

data = load_data()

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Choose a page:", ["Home", "Historical Patterns", "Predictive Analysis", "Insights", "About"])

# Home Page
if page == "Home":
    st.title("CadoPlug")
    st.write("""
    Welcome to CadoPlug! This interactive app explores avocado price trends, predicts future prices, 
    and provides actionable insights to optimize pricing strategies.
    """)
    st.image("logo.jpeg", use_column_width=True)
    st.write("### Data Snapshot")
    st.dataframe(data.head(10))

# Historical Patterns Page
if page == "Historical Patterns":
    st.title("Historical Patterns")
    region = st.selectbox("Select a Region", sorted(data['region'].unique()))
    region_data = data[data['region'] == region]

    st.subheader(f"Average Price Over Time in {region}")
    avg_price = region_data.groupby('Date')['AveragePrice'].mean().reset_index()
    chart = alt.Chart(avg_price).mark_line().encode(
        x='Date:T',
        y='AveragePrice:Q',
        tooltip=['Date', 'AveragePrice']
    ).properties(
        title=f"Price Trends in {region}"
    ).interactive()
    st.altair_chart(chart, use_container_width=True)

    st.subheader("Regional Price Comparison")
    region_avg = data.groupby('region')['AveragePrice'].mean().sort_values(ascending=False)
    fig = px.bar(region_avg, x=region_avg.index, y=region_avg.values, 
                 title="Average Price by Region",
                 labels={'x': 'Region', 'y': 'Average Price'})
    st.plotly_chart(fig)
# Seasonal Trends
    st.subheader(f"Seasonal Trends in {region}")
    region_data['Month'] = region_data['Date'].dt.month
    monthly_avg_price = region_data.groupby('Month')['AveragePrice'].mean().reset_index()
    monthly_chart = alt.Chart(monthly_avg_price).mark_line(point=True).encode(
        x=alt.X('Month:O', title='Month'),
        y=alt.Y('AveragePrice:Q', title='Average Price'),
        tooltip=['Month', 'AveragePrice']
    ).properties(
        title=f"Seasonal Price Trends in {region}"
    ).interactive()
    st.altair_chart(monthly_chart, use_container_width=True)
# Historical Patterns Page
if page == "Historical Patterns":
    st.title("Historical Patterns")
    region = st.selectbox("Select a Region", sorted(data['region'].unique()))
    region_data = data[data['region'] == region]

    # Average Price Over Time
    st.subheader(f"Average Price Over Time in {region}")
    avg_price = region_data.groupby('Date')['AveragePrice'].mean().reset_index()
    chart = alt.Chart(avg_price).mark_line().encode(
        x='Date:T',
        y='AveragePrice:Q',
        tooltip=['Date', 'AveragePrice']
    ).properties(
        title=f"Price Trends in {region}"
    ).interactive()
    st.altair_chart(chart, use_container_width=True)

    # Regional Price Comparison
    st.subheader("Regional Price Comparison")
    region_avg = data.groupby('region')['AveragePrice'].mean().sort_values(ascending=False)
    fig = px.bar(region_avg, x=region_avg.index, y=region_avg.values, 
                 title="Average Price by Region",
                 labels={'x': 'Region', 'y': 'Average Price'})
    st.plotly_chart(fig)

    # Seasonal Trends
    st.subheader(f"Seasonal Trends in {region}")
    region_data['Month'] = region_data['Date'].dt.month
    monthly_avg_price = region_data.groupby('Month')['AveragePrice'].mean().reset_index()
    monthly_chart = alt.Chart(monthly_avg_price).mark_line(point=True).encode(
        x=alt.X('Month:O', title='Month'),
        y=alt.Y('AveragePrice:Q', title='Average Price'),
        tooltip=['Month', 'AveragePrice']
    ).properties(
        title=f"Seasonal Price Trends in {region}"
    ).interactive()
    st.altair_chart(monthly_chart, use_container_width=True)

    # Volume vs Price
    st.subheader(f"Volume vs Price in {region}")
    scatter_chart = alt.Chart(region_data).mark_circle(size=60, opacity=0.7).encode(
        x=alt.X('AveragePrice:Q', title='Average Price'),
        y=alt.Y('TotalVolume:Q', title='Total Volume'),
        tooltip=['Date', 'AveragePrice', 'TotalVolume'],
        color=alt.Color('Month:O', title='Month', scale=alt.Scale(scheme='viridis'))
    ).properties(
        title=f"Volume vs Price Relationship in {region}"
    ).interactive()
    st.altair_chart(scatter_chart, use_container_width=True)
    
# Predictive Analysis Page
if page == "Predictive Analysis":
    st.title("Predictive Analysis")
    features = st.multiselect("Choose features for prediction:", 
                               ['TotalVolume', 'plu4046', 'plu4225', 'plu4770'], 
                               default=['TotalVolume', 'plu4046'])
    if features:
        X = data[features]
        y = data['AveragePrice']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression().fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        st.write(f"R² Score: {model.score(X_test, y_test):.2f}")
        st.write("### Predicted vs Actual Prices")
        fig = px.scatter(x=y_test, y=predictions, labels={"x": "Actual", "y": "Predicted"})
        fig.add_shape(type='line', x0=min(y_test), y0=min(y_test), x1=max(y_test), y1=max(y_test),
                      line=dict(color="Red", dash="dash"))
        st.plotly_chart(fig)

# Insights Page
if page == "Insights":
    st.title("Insights and Recommendations")
    st.markdown("""
    ### Key Insights
    - **Seasonal Trends**: Prices peak during summer and decline in winter.
    - **Regional Preferences**: Bulk avocados are more popular in urban regions.
    - **Volume vs Price**: Higher sales volumes generally correspond to lower prices.
    
    ### Recommendations
    - Focus marketing efforts on bulk sales during winter in urban areas.
    - Use predictive insights to set competitive prices in high-demand regions.
    - Consider diversifying packaging sizes based on regional preferences.
    """)

# About Page
if page == "About":
    st.title("About This App")
    st.write("""
    This app was developed as part of the Mastering Avocado Pricing project. It showcases my skills in:
    - Data Analysis
    - Machine Learning
    - Interactive Dashboard Development
    """)
    st.image("keneilwe.jpg", caption="Keneilwe Patricia", use_column_width=True)
    st.write("""
    **Developed by:** Keneilwe Patricia  
    **Email:** [patricia001105@gmail.com](mailto:patricia001105@gmail.com)  
    **LinkedIn:** [Keneilwe Rangwaga](https://www.linkedin.com/in/keneilwe-rangwaga14112004)  
    """)
