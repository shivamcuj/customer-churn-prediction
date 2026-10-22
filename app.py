
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Customer Churn Analytics",
    page_icon="📊",
    layout="wide"
)

@st.cache_resource
def load_model():
    return joblib.load("churn_model.joblib")

model = load_model()

st.title("Customer Churn Prediction")
st.write(
    "Analyze customer churn patterns and estimate "
    "the churn risk for each customer."
)

uploaded_file = st.file_uploader(
    "Upload customer CSV",
    type=["csv"]
)

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.subheader("Dataset overview")
    col1, col2 = st.columns(2)
    col1.metric("Customers", len(data))
    col2.metric("Columns", len(data.columns))

    if "Churn" in data.columns:
        churn_counts = data["Churn"].value_counts()
        st.bar_chart(churn_counts)

    # Remove identifiers and labels before prediction
    
    features = data.drop(
        columns=["customerID", "Churn"],
        errors="ignore"
    ).copy()

    # Convert TotalCharges to numeric
    if "TotalCharges" in features.columns:
        features["TotalCharges"] = pd.to_numeric(
            features["TotalCharges"],
            errors="coerce"
        )

    # Apply the same zero-tenure cleaning used during training
    if "tenure" in features.columns and "TotalCharges" in features.columns:
        mask = (
            features["tenure"].eq(0)
            & features["TotalCharges"].isna()
        )
        features.loc[mask, "TotalCharges"] = 0

    # Generate predictions
    probabilities = model.predict_proba(features)[:, 1]


    try:
        probabilities = model.predict_proba(features)[:, 1]

        results = data.copy()
        results["Predicted_Churn_Probability"] = probabilities
        results["Predicted_Churn"] = [
            "Yes" if p >= 0.5 else "No"
            for p in probabilities
        ]

        st.subheader("Prediction results")
        st.dataframe(results.head(100), use_container_width=True)

        st.download_button(
            "Download predictions CSV",
            data=results.to_csv(index=False).encode("utf-8"),
            file_name="churn_predictions.csv",
            mime="text/csv"
        )

    except (ValueError, KeyError) as error:
        st.error(
            "CSV columns ya data types expected model input "
            "se match nahi karte. Original dataset ka schema "
            "check karo."
        )
        st.caption(str(error))
else:
    st.info(
        "Upload a CSV with the same feature columns "
        "used during training."
    )
