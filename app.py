import streamlit as st
import pandas as pd
import joblib

df=pd.read_csv("laptop_price.csv",encoding="latin1")
model=joblib.load("ridge_laptopPrice_model.pkl")
scaler=joblib.load("scaler_laptopPrice.pkl")
expected_columns=joblib.load("feature_columns_laptopPrice.pkl")

st.title("Laptop Price Prediction 💻")
st.markdown("Provide the following details to predict the laptop price.")

# 'Company', 'Product', 'TypeName', 'Inches',
#        'ScreenResolution', 'Cpu', 'Ram', 'Memory', 'Gpu', 'OpSys', 'Weight'

#collect user input
company=st.selectbox("Company", options=df["Company"].unique())
# product=st.selectbox("Product", options=df["Product"].unique())
type_name=st.selectbox("Type Name", options=df["TypeName"].unique())
inches=st.number_input("Inches", min_value=0.0, value=15.6,step=0.1)
screen_resolution=st.selectbox("Screen Resolution", options=df["ScreenResolution"].unique())
cpu=st.selectbox("CPU", options=df["Cpu"].unique())
ram=st.selectbox("RAM", options=df["Ram"].unique())
memory=st.selectbox("Memory", options=df["Memory"].unique())
gpu=st.selectbox("GPU", options=df["Gpu"].unique())
osys=st.selectbox("Operating System", options=df["OpSys"].unique())
weight=st.number_input("Weight", min_value=0.0, value=2.0,step=0.1)

#when predict is clicked
if st.button("Predict Laptop Price"):
    raw_input={
        "Company":company,
        # "Product":product,
        "TypeName":type_name,
        "Inches":inches,
        "ScreenResolution":screen_resolution,
        "Cpu":cpu,
        "Ram":ram,
        "Memory":memory,
        "Gpu":gpu,
        "OpSys":osys,
        "Weight":weight
    }

    input_df = pd.DataFrame([raw_input])
    
    # Convert text categories into dummy/one-hot columns to match your training data
    input_df = pd.get_dummies(input_df)

    # Add any missing columns with 0 and ensure exact column order
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0
            
    input_df = input_df[expected_columns]

    # Scale features and get the predicted price
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]

    # Set a realistic minimum floor
    if prediction <= 0:
        st.warning("⚠️ This specific hardware combination falls outside the model's normal training range.")
        prediction = max(150.0, abs(prediction))

    # Display the predicted laptop price
    st.success(f"💻 Estimated Laptop Price: €{prediction:,.2f}")  