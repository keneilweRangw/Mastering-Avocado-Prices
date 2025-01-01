import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import altair as alt
import pickle

# Load the saved Random Forest model
with open('best_random_forest_model.pkl', 'rb') as file:
    best_rf_model = pickle.load(file)
    
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



# Define the Predictive Analysis page
if page == "Predictive Analysis":
    st.title("Predictive Analysis")

    # User input for feature selection
    st.subheader("Choose Features for Prediction:")
    selected_features = st.multiselect("Select features to include:", 
                                        ['AveragePrice', 'Month', 'Year'])

    if selected_features:
        st.write(f"You selected: {', '.join(selected_features)}")

        # Allow users to input feature values
        st.subheader("Input Feature Values:")
        feature_inputs = {}
        for feature in selected_features:
            feature_value = st.number_input(f"Enter value for {feature}:", 
                                            value=0 if feature != 'Month' else 1,
                                            step=1)
            feature_inputs[feature] = feature_value

        # Convert inputs into a DataFrame
        input_df = pd.DataFrame([feature_inputs])

        # Display the input data
        st.subheader("Input Data:")
        st.write(input_df)

        # Make predictions using the loaded model
        st.subheader("Prediction:")
        try:
            prediction = best_rf_model.predict(input_df)[0]
            st.success(f"Predicted Outcome: {prediction}")
        except Exception as e:
            st.error(f"Error in prediction: {e}")

        # Add Prescriptive Insights
        st.subheader("Prescriptive Model - Recommended Actions:")
        if 'AveragePrice' in selected_features:
            if prediction > 1.5:  # Example threshold
                st.write("**Insight:** High average price predicted. Consider increasing inventory levels to meet potential demand.")
            else:
                st.write("**Insight:** Low average price predicted. Optimize inventory to avoid overstocking.")

        # Example: Prescriptive suggestion based on Month
        if 'Month' in selected_features:
            if feature_inputs['Month'] in [11, 12]:  # Holiday season
                st.write("**Insight:** It's the holiday season. Stock up on popular items to meet festive demand.")
            else:
                st.write("**Insight:** Regular season. Maintain normal inventory levels.")
    else:
        st.warning("Please select at least one feature to proceed.")


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
    
    **Developed by:** Keneilwe Patricia  
""")
    st.image("keneilwe.jpg", caption="Keneilwe Patricia", use_column_width=True)
    st.write("""
    **Email:** [patricia001105@gmail.com](mailto:patricia001105@gmail.com)  
    **LinkedIn:** [Keneilwe Rangwaga](https://www.linkedin.com/in/keneilwe-rangwaga14112004)  
    """)
