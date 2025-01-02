import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import altair as alt
import pickle
# Page configuration
st.set_page_config(
    page_title="CadoPlug - Avocado Trends",
    page_icon="🥑"
)

# Add custom CSS for avocado colors
st.markdown(
    """
    <style>
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: linear-gradient(to bottom, #b9fbc2, #fff176); /* Light green to yellow gradient */
    }
    .sidebar .css-17eq0hr {
        color: #2e7d32; /* Dark green text for sidebar */
    }

    /* Main content styling */
    .main-content {
        background-color: #fdf3e3; /* Light beige background for main content */
    }

    /* Header styling */
    .css-18e3th9 h1, .css-18e3th9 h2, .css-18e3th9 h3 {
        color: #3e2723; /* Dark brown headers */
    }

    /* Button and interactive element styling */
    .css-1q8dd3e, .css-1q8dd3e:hover {
        background-color: #aed581; /* Light green button background */
        color: #ffffff; /* White text */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Load the saved Random Forest model
with open('random_forest_model.pkl', 'rb') as file:
    best_rf_model = pickle.load(file)

# Load Data
@st.cache_data
def load_data():
    data = pd.read_csv('cleaned_avocado_data.csv')
    data['Date'] = pd.to_datetime(data['Date'])
    return data

data = load_data()

st.sidebar.image("logo.jpeg", use_column_width=True) 
# Sidebar Navigation
st.sidebar.title("Navigation")
# Add an image to the sidebar
page = st.sidebar.radio("Choose a page:", ["Home", "Historical Patterns", "Predictive Analysis", "Insights", "About"])

# Home Page
if page == "Home":
    st.title("CadoPlug")
    st.write("""
    Welcome to **CadoPlug**! 🌱 An app designed to plug you with premium `AVO` trends 
    """)
    st.image("logo.jpeg", use_column_width=True)
    st.write("### Data Snapshot")
    st.dataframe(data.head(3))

     # User Guide
    st.header("User Guide")
    st.markdown("This app serves multiple user groups. Find your profile below to discover insights tailored to your needs:")

    # Dictionary of user groups with descriptions
    import streamlit as st

# Define user groups and their respective details
    user_groups = {
    "Farmers and Producers": """
    - **Purpose:**
        - Plan production cycles and optimize pricing strategies.
    - **How to Use:** 
        - View price trends and identify high-demand periods on the Historical Trends page.
    """,
    "Distributors and Retailers": """
    - **Purpose:**
        - Manage inventory and identify regional demand.
    - **How to Use:**
        - Compare prices on the Historical Trends page.
        - Forecast sales on the Predictive Analysis page.
        - Get stocking strategies on the Insights page.
    """,
    "Consumers": """
    - **Purpose:**
        - Plan purchases by identifying affordable times and regions.
    - **How to Use:**
        - Check Historical Trends for low-price periods.
        - Forecast affordability on the Predictive Analysis page.
    """,
    "Policy Makers and Economists": """
    - **Purpose:**
        - Analyze market trends and ensure fair trade practices.
    - **How to Use:**
        - Study long-term trends on Historical Trends.
        - Forecast market conditions on Predictive Analysis.
        - Access strategies on the Insights page.
    """,
    "Sustainability Advocates": """
    - **Purpose:**
        - Promote sustainable practices and reduce waste.
    - **How to Use:**
        - Identify overproduction risks on Historical Trends.
        - Align production with demand on the Insights page.
    """,
    "Business Owners and Decision Makers": """
    - **Purpose:**
        - Improve efficiency and profitability using data insights.
    - **How to Use:**
        - View past trends on Historical Trends.
        - Forecast future sales on Predictive Analysis.
        - Get actionable recommendations on Insights.
    """
}

# Display user guide sections with numbering
    st.title("User Groups Guide")
    for idx, (group, description) in enumerate(user_groups.items(), start=1):
     st.write(f"**{idx}. {group}**")  # Adds a number before each group
     st.markdown(description)

# Getting Started Guide Section
    st.header("General Navigation")
    st.markdown("""
        Follow the steps below to begin exploring.
        1. Select your **user group** from the list above to understand how the app can serve your needs.
        2. Navigate to the relevant pages based on your interests:
          - **Historical Patterns**: View avocado price and sales trends over time and compare prices by region.
          - **Predictive Analysis**: To forecast future avocado sales
          - **Insights**: Get actionable insights and recommendations to optimize pricing and sales strategies based on data trends.
          - **About**: Learn more about the app's purpose, development, and how it can benefit you.
    """)

    st.write("Now, explore the app and make informed decisions!")
# Historical Patterns Page
if page == "Historical Patterns":
    st.title("Historical Patterns")
    region = st.selectbox("Select a Region", sorted(data['region'].unique()))
    region_data = data[data['region'] == region]

    st.subheader(f"Average Price and Total Volume Over Time in {region}")

# Calculate the average price and total volume over time for the selected region
    avg_price_volume = region_data.groupby('Date')[['AveragePrice', 'TotalVolume']].mean().reset_index()

# Create Altair chart
    base = alt.Chart(avg_price_volume).encode(x='Date:T')

# Line chart for Average Price
    price_line = base.mark_line(color='blue').encode(
    y=alt.Y('AveragePrice:Q', title='Average Price'),
    tooltip=['Date', 'AveragePrice']
)

# Line chart for Total Volume
    volume_line = base.mark_line(color='orange').encode(
    y=alt.Y('TotalVolume:Q', title='Total Volume', axis=alt.Axis(titleColor='orange')),
    tooltip=['Date', 'TotalVolume']
)

# Combine the two charts
    chart = alt.layer(price_line, volume_line).resolve_scale(
    y='independent'  # Use independent scales for the y-axes
).properties(
    title=f"Price and Volume Trends in {region}"
).interactive()

    st.altair_chart(chart, use_container_width=True)

# Seasonal Trends
    st.subheader(f"Seasonal Trends in {region}")
# Extract month and calculate monthly averages for AveragePrice and TotalVolume
    region_data['Month'] = region_data['Date'].dt.month
    monthly_avg = region_data.groupby('Month')[['AveragePrice', 'TotalVolume']].mean().reset_index()

# Create Altair chart
    base = alt.Chart(monthly_avg).encode(
    x=alt.X('Month:O', title='Month')
)

# Line chart for Average Price
    price_line = base.mark_line(point=True, color='blue').encode(
    y=alt.Y('AveragePrice:Q', title='Average Price'),
    tooltip=['Month', 'AveragePrice']
)

# Line chart for Total Volume
    volume_line = base.mark_line(point=True, color='orange').encode(
    y=alt.Y('TotalVolume:Q', title='Total Volume', axis=alt.Axis(titleColor='orange')),
    tooltip=['Month', 'TotalVolume']
)

# Combine the two charts
    seasonal_chart = alt.layer(price_line, volume_line).resolve_scale(
    y='independent'  # Use independent scales for the y-axes
).properties(
    title=f"Seasonal Price and Volume Trends in {region}"
).interactive()

    st.altair_chart(seasonal_chart, use_container_width=True)
    import altair as alt
    import pandas as pd

# Regional Sales Volume Comparison
    st.subheader("Regional Sales Volume Comparison")
    region_avg_volume = data.groupby('region')['TotalVolume'].mean().sort_values(ascending=False).reset_index()

# Create the interactive chart for Regional Sales Volume
    volume_chart = alt.Chart(region_avg_volume).mark_bar(color='skyblue').encode(
    x=alt.X('region:N', sort='-y'),  # Sort regions by TotalVolume
    y='TotalVolume:Q',
    tooltip=['region:N', 'TotalVolume:Q']
).properties(
    title="Total Volume by Region",
    width=800,
    height=400
)

    st.altair_chart(volume_chart, use_container_width=True)

# Regional Price Comparison
    st.subheader("Regional Price Comparison")
    region_avg_price = data.groupby('region')['AveragePrice'].mean().sort_values(ascending=False).reset_index()

# Create the interactive chart for Regional Price Comparison
    price_chart = alt.Chart(region_avg_price).mark_bar(color='lightgreen').encode(
    x=alt.X('region:N', sort='-y'),  # Sort regions by AveragePrice
    y='AveragePrice:Q',
    tooltip=['Region:N', 'AveragePrice:Q']
).properties(
    title="Average Price by Region",
    width=800,
    height=400
)

    st.altair_chart(price_chart, use_container_width=True)



# Define the Predictive Analysis page
if page == "Predictive Analysis":
    st.title("Predictive Analysis")

    # User input for feature selection
    st.subheader("Choose Features for Prediction:")
    selected_features = st.multiselect("Select features to include:", 
                                        ['AveragePrice', 'Month', 'Year'])

    if selected_features:
        st.write(f"You selected: {', '.join(selected_features)}")

        # Allow users to input feature values with limits
        st.subheader("Input Feature Values:")
        feature_inputs = {}
        for feature in selected_features:
            if feature == 'AveragePrice':
                feature_value = st.number_input(f"Enter value for {feature}:", 
                                                value=0.5,  # Default value
                                                min_value=0.0,  # Min limit
                                                max_value=10.0,  # Max limit
                                                step=0.1)  # Step size
            elif feature == 'Month':
                feature_value = st.number_input(f"Enter value for {feature}:", 
                                                value=1,  # Default value
                                                min_value=1,  # Min value (January)
                                                max_value=12,  # Max value (December)
                                                step=1)  # Step size
            elif feature == 'Year':
                feature_value = st.number_input(f"Enter value for {feature}:", 
                                                value=2023,  # Default value
                                                min_value=2015,  # Min value (start year)
                                                max_value=2025,  # Max value (current or near future)
                                                step=1)  # Step size
            feature_inputs[feature] = feature_value

        # Convert inputs into a DataFrame
        input_df = pd.DataFrame([feature_inputs])

        # Display the input data
        st.subheader("Input Data:")
        st.write(input_df)

        # Make predictions using the loaded model
        st.subheader("Sales Prediction:")
        try:
            prediction = best_rf_model.predict(input_df)[0]
            st.success(f"Predicted Outcome: {prediction}")
        except Exception as e:
            st.error(f"Error in prediction: {e}")

        # Add Prescriptive Insights
        st.subheader("Recommended Actions:")
        if 'AveragePrice' in selected_features:
            if prediction < 10:  # threshold for high price prediction
                st.write("**Insight:** High average price predicted. Consider increasing inventory levels to meet potential demand.")
            else:
                st.write("**Insight:** Low average price predicted. Optimize inventory to avoid overstocking.")

        # Prescriptive suggestion based on Month
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
    This app was developed as part of the Mastering Avocado Pricing project. It explores avocado price trends, predicts future prices, and provides actionable insights to optimize pricing strategies.
    
    **Developed by:** Keneilwe Patricia Rangwaga  
""")
    st.image("keneilwe.jpg", caption="Keneilwe",  width=200)
    st.write("""
    **Email:** [patricia001105@gmail.com](mailto:patricia001105@gmail.com)  
    **LinkedIn:** [Keneilwe Rangwaga](https://www.linkedin.com/in/keneilwe-rangwaga14112004)
    **Portfolio:** [Here](https://keneilwerangw.github.io/My_Portfolio/home.html)
    """)
