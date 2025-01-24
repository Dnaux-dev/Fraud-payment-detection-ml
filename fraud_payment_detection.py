import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Title and description with animation
st.markdown(
    "<h1 style='text-align: center; color: #4CAF50;'>Fraud Payment Detection App</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    """
    <div style="text-align: center; color: #fff;">
        <p>Detect fraudulent transactions by entering details manually or uploading a dataset.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# User selection: Enter details or upload a dataset
option = st.radio("Choose input method:", ["Enter Details", "Upload Dataset"])

# Updated default dataset
default_data = pd.DataFrame({
    'amount': [1000, 200, 5000],
    'old_balance': [5000, 800, 10000],
    'new_balance': [4000, 600, 5000],
    'transaction_type': [0, 1, 0],  # TRANSFER (0), CASH_OUT (1)
    'is_fraud': [0, 0, 1]
})

# Define the model
model = RandomForestClassifier(random_state=42)

if option == "Enter Details":
    # Sidebar for manual transaction details input
    st.sidebar.header("Input Transaction Details")
    amount = st.sidebar.number_input("Transaction Amount", min_value=0.0, step=0.01)
    old_balance = st.sidebar.number_input("Old Balance", min_value=0.0, step=0.01)
    new_balance = st.sidebar.number_input("New Balance", min_value=0.0, step=0.01)
    transaction_type = st.sidebar.selectbox("Transaction Type", ["TRANSFER", "CASH_OUT"])

    # Prepare input data
    input_data = pd.DataFrame({
        'amount': [amount],
        'old_balance': [old_balance],
        'new_balance': [new_balance],
        'transaction_type': [0 if transaction_type == "TRANSFER" else 1]
    })

    # Logical validation
    logical_check = new_balance - old_balance == amount

    # Train the model on default data
    X = default_data[['amount', 'old_balance', 'new_balance', 'transaction_type']]
    y = default_data['is_fraud']
    model.fit(X, y)

    # Prediction based on manual input
    if st.button("Check for Fraudulent Payment"):
        if logical_check:
            st.subheader("Prediction Result")
            st.write("✅ Transaction is Legitimate.")
        else:
            prediction = model.predict(input_data)[0]
            st.subheader("Prediction Result")
            if prediction == 1:
                st.write("🚨 Fraudulent Transaction Detected!")
            else:
                st.write("✅ Transaction is Legitimate.")

elif option == "Upload Dataset":
    # File uploader for dataset
    uploaded_file = st.file_uploader("Upload your dataset (CSV)", type="csv")

    if uploaded_file is not None:
        # Load dataset
        data = pd.read_csv(uploaded_file)

        st.subheader("Dataset Preview")
        st.write(data.head())

        # Check if required columns are present
        required_columns = ['amount', 'old_balance', 'new_balance', 'transaction_type']
        if all(column in data.columns for column in required_columns):
            # Preprocess the data
            X = data[['amount', 'old_balance', 'new_balance', 'transaction_type']]

            # Train the model using default data
            y_default = default_data['is_fraud']
            model.fit(default_data[['amount', 'old_balance', 'new_balance', 'transaction_type']], y_default)

            # Predict fraud for the uploaded dataset
            predictions = model.predict(X)
            data['is_fraud'] = predictions  # Add predictions as a new column

            st.subheader("Predicted Results")
            st.write(data.head())

            # Allow download of the dataset with predictions
            csv = data.to_csv(index=False)
            st.download_button(
                label="Download Dataset with Predictions",
                data=csv,
                file_name="predicted_fraud_dataset.csv",
                mime="text/csv",
            )
        else:
            st.write("Please make sure your dataset contains the following columns:", required_columns)
    else:
        st.write("Awaiting CSV file upload.")

# Footer with credit
st.markdown(
    """
    <hr>
    <div style="text-align: center;">
        <p>Made with ❤️ by <b>Ajilore Daniel Okikiola</b></p>
    </div>
    """,
    unsafe_allow_html=True,
)
