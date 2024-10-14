import streamlit as st
import joblib
import pandas as pd

# Load the model
model = joblib.load('flight_price_model.pkl')
data = pd.read_excel('Flight_Tickets_data.xlsx')

# Streamlit app
st.set_page_config(page_title="Flight Price Prediction", page_icon="✈️", layout="centered")
st.title('Flight Price Prediction')

# Input fields
st.markdown("### Enter Flight Details")
airline = st.selectbox('Airline', data['Airline'].unique())
source = st.selectbox('Source', data['Source'].unique())
destination = st.selectbox('Destination', data['Destination'].unique())
total_stops = st.selectbox('Total Stops', data['Total-Stop'].unique())

# Generate route based on source and destination
route = f'{source[:3].upper()} → {destination[:3].upper()}'
st.text(f'Route: {route}')  # Display the generated route

if st.button('Predict'):
    # Filter data for the selected route and other parameters
    filtered_data = data[
        (data['Airline'] == airline) &
        (data['Source'] == source) &
        (data['Destination'] == destination) &
        (data['Total-Stop'] == total_stops)
    ]

    if filtered_data.empty:
        st.error("No data found for the selected route.")
    else:
        # Sort by Date_of_Journey to get the latest one
        filtered_data = filtered_data.sort_values(by='Date_of_Journey', ascending=False)

        # Select row with the latest Date_of_Journey
        latest_row = filtered_data.iloc[0]

        # Extract the corresponding duration and price
        route_duration = latest_row['Duration']
        predicted_price = model.predict(filtered_data[['Airline', 'Source', 'Destination', 'Route', 'Total-Stop']])

        st.success(f'Predicted Price: ₹{float(predicted_price[0]):,.2f}')
        st.info(f'Actual Price for Selected Route: ₹{latest_row["Price"]:.2f}')
        st.info(f'Duration for Selected Route: {route_duration}')

# Add some spacing
st.markdown("<br><br>", unsafe_allow_html=True)

