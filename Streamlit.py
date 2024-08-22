import pandas as pd
import pickle
import streamlit as st
import sklearn
print(sklearn.__version__)

import os 
# print(os.path.abspath(os.getcwd()))

print("path", os.path.join(os.getcwd(), "scaling.pkl"))
from sklearn.preprocessing import StandardScaler


# Load the models and scalers
scaler_model = pickle.load(open('scaling.pkl', 'rb'))
rf_model = pickle.load(open('rf_model.pkl', 'rb'))

# Function to make predictions
def predict_price(input_data, model, scaler):
    # Initialize an empty DataFrame
    df_input2 = pd.DataFrame([input_data])
    # Scale the data
    X = scaler.transform(df_input2)
    # Make predictions
    y_pred = model.predict(X)
    return y_pred[0]

# Streamlit app
def main():
    st.title("Flight Price Prediction")

    with st.form(key='prediction_form'):
        st.subheader("Select a Boarding City")
        boarding_city = st.radio(
            "Select a boarding city",
            ['Aracaju', 'Brasilia', 'Campo_Grande', 'Florianopolis', 'Natal', 'Recife', 'Rio_de_Janeiro', 'Salvador', 'Sao_Paulo']
        )
        
        st.subheader("Select a Destination City")
        destination_city = st.radio(
            "Select a destination city",
            ['Aracaju', 'Brasilia', 'Campo_Grande', 'Florianopolis', 'Natal', 'Recife', 'Rio_de_Janeiro', 'Salvador', 'Sao_Paulo']
        )

        st.subheader("Select a Flight Type")
        flight_type = st.radio(
            "Select a flight type",
            ['premium', 'economic', 'firstClass']
        )

        st.subheader("Select Agency")
        agency = st.radio(
            "Select an agency",
            ['FlyingDrops', 'Rainbow', 'CloudFy']
        )

        st.subheader("Travel Details")
        day = st.number_input("Travel Day", min_value=1, max_value=31, value=5)
        week_no = st.number_input("Travel Week No", min_value=1, max_value=53, value=7)
        week_day = st.number_input("Travel Week Day", min_value=1, max_value=7, value=5)

        submit_button = st.form_submit_button("Predict")

    if submit_button:
        boarding = 'from_' + boarding_city
        destination = 'destination_' + destination_city
        flight_type = 'flightType_' + flight_type
        agency = 'agency_' + agency

        boarding_city_list = ['from_Florianopolis (SC)', 'from_Sao_Paulo (SP)', 'from_Salvador (BH)', 'from_Brasilia (DF)', 'from_Rio_de_Janeiro (RJ)', 'from_Campo_Grande (MS)', 'from_Aracaju (SE)', 'from_Natal (RN)', 'from_Recife (PE)']
        destination_city_list = ['destination_Florianopolis (SC)', 'destination_Sao_Paulo (SP)', 'destination_Salvador (BH)', 'destination_Brasilia (DF)', 'destination_Rio_de_Janeiro (RJ)', 'destination_Campo_Grande (MS)', 'destination_Aracaju (SE)', 'destination_Natal (RN)', 'destination_Recife (PE)']
        class_list = ['flightType_economic', 'flightType_firstClass', 'flightType_premium']
        agency_list = ['agency_Rainbow', 'agency_CloudFy', 'agency_FlyingDrops']

        travel_dict = dict()

        for city in boarding_city_list:
            travel_dict[city] = 1 if city == boarding else 0
        for city in destination_city_list:
            travel_dict[city] = 1 if city == destination else 0
        for flight_class in class_list:
            travel_dict[flight_class] = 1 if flight_class == flight_type else 0
        for agency in agency_list:
            travel_dict[agency] = 1 if agency == agency else 0
        travel_dict['week_no'] = week_no
        travel_dict['week_day'] = week_day
        travel_dict['day'] = day

        # Make prediction
        predicted_price = predict_price(travel_dict, rf_model, scaler_model)
        st.write(f"Predicted Flight Price Per Person: ${predicted_price:.2f}")

if __name__ == "__main__":
    main()
