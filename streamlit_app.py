import streamlit as st
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB

# ---------------------------------------------------------
# CreditWise - Loan Approval Prediction System
# ---------------------------------------------------------

st.set_page_config(
    page_title="CreditWise | Loan Approval",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_PATH = "loan_approval_data.csv"

st.markdown(
    """
    <style>
    .main {background-color: #f7f9fc;}
    .block-container {padding-top: 2rem; padding-bottom: 2rem;}
    .hero {
        padding: 1.5rem 1.8rem;
        border-radius: 16px;
        background: linear-gradient(110deg, #102a43, #176b87);
        color: white;
        margin-bottom: 1.2rem;
    }
    .hero h1 {color: white; margin-bottom: .2rem;}
    .hero p {color: #e4f1f7; margin-bottom: 0;}
    div[data-testid="stMetric"] {
        background: white; padding: 14px; border-radius: 12px;
        border: 1px solid #e7edf3;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_data
def load_data():
    data = pd.read_csv(DATA_PATH)
    return data

@st.cache_resource
def train_model(data):
    # The dataset contains some rows without a known target.
    # Such rows cannot be used for supervised training.
    training_data = data.dropna(subset=["Loan_Approved"]).copy()
    training_data["Loan_Approved"] = (
        training_data["Loan_Approved"].astype(str).str.strip().str.title()
    )
    training_data = training_data[
        training_data["Loan_Approved"].isin(["Yes", "No"])
    ]

    y = (training_data["Loan_Approved"] == "Yes").astype(int)

    X = training_data.drop(
        columns=["Loan_Approved", "Applicant_ID"],
        errors="ignore",
    )

    # Match the feature engineering used in the notebook.
    X["DTI_Ratio_sq"] = pd.to_numeric(X["DTI_Ratio"], errors="coerce") ** 2
    X["Credit_Score_sq"] = (
        pd.to_numeric(X["Credit_Score"], errors="coerce") ** 2
    )
    X = X.drop(columns=["DTI_Ratio", "Credit_Score"], errors="ignore")

    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    # Education_Level was label-encoded in the original notebook.
    education_feature = (
        ["Education_Level"] if "Education_Level" in categorical_features else []
    )
    onehot_features = [
        col for col in categorical_features if col not in education_feature
    ]

    transformers = []

    if numeric_features:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="mean")),
            ]
        )
        transformers.append(("numeric", numeric_pipeline, numeric_features))

    if education_feature:
        education_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "ordinal",
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value",
                        unknown_value=-1,
                    ),
                ),
            ]
        )
        transformers.append(
            ("education", education_pipeline, education_feature)
        )

    if onehot_features:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "onehot",
                    OneHotEncoder(
                        drop="first",
                        handle_unknown="ignore",
                        sparse_output=False,
                    ),
                ),
            ]
        )
        transformers.append(
            ("categorical", categorical_pipeline, onehot_features)
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("scaler", StandardScaler()),
            ("classifier", GaussianNB()),
        ]
    )

    model.fit(X, y)
    return model, X, training_data, numeric_features, categorical_features


def add_engineered_features(input_df):
    result = input_df.copy()
    result["DTI_Ratio_sq"] = (
        pd.to_numeric(result["DTI_Ratio"], errors="coerce") ** 2
    )
    result["Credit_Score_sq"] = (
        pd.to_numeric(result["Credit_Score"], errors="coerce") ** 2
    )
    return result.drop(
        columns=["DTI_Ratio", "Credit_Score"],
        errors="ignore",
    )


# -------------------- Header --------------------

st.markdown(
    """
    <div class="hero">
        <h1>💳 CreditWise</h1>
        <p>Intelligent Loan Approval Prediction System</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write(
    "Evaluate applicant information using a machine learning model. "
    "Enter the applicant's details below to generate a prediction."
)

try:
    raw_data = load_data()
    model, X_train, training_data, numeric_features, categorical_features = (
        train_model(raw_data)
    )
except FileNotFoundError:
    st.error(
        "Dataset not found. Upload **loan_approval_data.csv** to the same "
        "GitHub repository/folder as streamlit_app.py, then redeploy."
    )
    st.stop()
except Exception as exc:
    st.error(f"The dataset or model could not be loaded: {exc}")
    st.stop()

# -------------------- Sidebar --------------------

with st.sidebar:
    st.title("About CreditWise")
    st.write(
        "CreditWise uses historical loan application data and a "
        "Gaussian Naive Bayes classifier to estimate loan approval."
    )
    st.divider()
    st.caption("Model: Gaussian Naive Bayes")
    st.caption("Training data: loan_approval_data.csv")
    st.caption(f"Training records: {len(training_data):,}")

# -------------------- Dataset overview --------------------

approved_count = int(
    (training_data["Loan_Approved"].astype(str).str.title() == "Yes").sum()
)
rejected_count = int(
    (training_data["Loan_Approved"].astype(str).str.title() == "No").sum()
)

metric1, metric2, metric3 = st.columns(3)
metric1.metric("Training applications", f"{len(training_data):,}")
metric2.metric("Previously approved", f"{approved_count:,}")
metric3.metric("Previously rejected", f"{rejected_count:,}")

st.divider()
st.subheader("Applicant Information")
st.caption("Fields marked with an asterisk are required.")

# Get sensible defaults and dropdown choices from the uploaded dataset.
def numeric_default(column, fallback=0.0):
    values = pd.to_numeric(raw_data[column], errors="coerce").dropna()
    return float(values.median()) if not values.empty else fallback

def category_options(column, fallback):
    values = raw_data[column].dropna().astype(str).unique().tolist()
    values = sorted(values)
    return values if values else fallback

with st.form("loan_application_form"):
    st.markdown("#### 1. Personal and employment details")
    c1, c2, c3 = st.columns(3)

    with c1:
        age = st.number_input(
            "Applicant age *", min_value=18, max_value=100,
            value=int(numeric_default("Age", 30)), step=1
        )
        gender_options = category_options("Gender", ["Male", "Female"])
        gender = st.selectbox("Gender", gender_options)

    with c2:
        marital_options = category_options(
            "Marital_Status", ["Single", "Married"]
        )
        marital_status = st.selectbox("Marital status", marital_options)
        dependents = st.number_input(
            "Number of dependents", min_value=0, max_value=20,
            value=int(numeric_default("Dependents", 0)), step=1
        )

    with c3:
        education_options = category_options(
            "Education_Level", ["Graduate", "Not Graduate"]
        )
        education = st.selectbox("Education level", education_options)
        employment_options = category_options(
            "Employment_Status",
            ["Salaried", "Self-employed", "Contract", "Unemployed"],
        )
        employment = st.selectbox("Employment status", employment_options)

    employer_options = category_options(
        "Employer_Category",
        ["Government", "Private", "MNC", "Business", "Unemployed"],
    )
    employer_category = st.selectbox(
        "Employer category", employer_options
    )

    st.markdown("#### 2. Financial information")
    c1, c2, c3 = st.columns(3)

    with c1:
        applicant_income = st.number_input(
            "Applicant monthly income (₹) *",
            min_value=0.0,
            value=numeric_default("Applicant_Income"),
            step=1000.0,
        )
        coapplicant_income = st.number_input(
            "Co-applicant monthly income (₹)",
            min_value=0.0,
            value=numeric_default("Coapplicant_Income"),
            step=1000.0,
        )

    with c2:
        credit_score = st.number_input(
            "Credit score *", min_value=0.0, max_value=1000.0,
            value=numeric_default("Credit_Score", 650.0), step=1.0
        )
        savings = st.number_input(
            "Savings balance (₹)", min_value=0.0,
            value=numeric_default("Savings"), step=1000.0
        )

    with c3:
        existing_loans = st.number_input(
            "Existing loans", min_value=0, max_value=50,
            value=int(numeric_default("Existing_Loans", 0)), step=1
        )
        dti_ratio = st.number_input(
            "Debt-to-income ratio", min_value=0.0, max_value=10.0,
            value=numeric_default("DTI_Ratio", 0.3), step=0.01,
            help="Enter the ratio in the same format used in your dataset."
        )

    st.markdown("#### 3. Loan details")
    c1, c2, c3 = st.columns(3)

    with c1:
        loan_amount = st.number_input(
            "Requested loan amount (₹) *", min_value=0.0,
            value=numeric_default("Loan_Amount"), step=5000.0
        )
        loan_term = st.number_input(
            "Loan term (months)", min_value=1, max_value=600,
            value=int(numeric_default("Loan_Term", 120)), step=1
        )

    with c2:
        collateral_value = st.number_input(
            "Collateral value (₹)", min_value=0.0,
            value=numeric_default("Collateral_Value"), step=5000.0
        )
        purpose_options = category_options(
            "Loan_Purpose", ["Home", "Education", "Personal", "Business"]
        )
        loan_purpose = st.selectbox("Loan purpose", purpose_options)

    with c3:
        area_options = category_options(
            "Property_Area", ["Urban", "Semiurban", "Rural"]
        )
        property_area = st.selectbox("Property area", area_options)

    submitted = st.form_submit_button(
        "Predict Loan Approval", type="primary", use_container_width=True
    )

# -------------------- Prediction --------------------

if submitted:
    if loan_amount <= 0:
        st.error("Please enter a loan amount greater than zero.")
    else:
        applicant = pd.DataFrame(
            [
                {
                    "Applicant_Income": applicant_income,
                    "Coapplicant_Income": coapplicant_income,
                    "Employment_Status": employment,
                    "Age": age,
                    "Marital_Status": marital_status,
                    "Dependents": dependents,
                    "Credit_Score": credit_score,
                    "Existing_Loans": existing_loans,
                    "DTI_Ratio": dti_ratio,
                    "Savings": savings,
                    "Collateral_Value": collateral_value,
                    "Loan_Amount": loan_amount,
                    "Loan_Term": loan_term,
                    "Loan_Purpose": loan_purpose,
                    "Property_Area": property_area,
                    "Education_Level": education,
                    "Gender": gender,
                    "Employer_Category": employer_category,
                }
            ]
        )

        try:
            model_input = add_engineered_features(applicant)
            prediction = int(model.predict(model_input)[0])
            probabilities = model.predict_proba(model_input)[0]
            approval_probability = float(probabilities[1])

            st.divider()
            st.subheader("Prediction Result")

            if prediction == 1:
                st.success("### Prediction: Loan Approved")
                st.write(
                    "The model predicts that this application belongs to "
                    "the loan-approval class."
                )
            else:
                st.error("### Prediction: Loan Rejected")
                st.write(
                    "The model predicts that this application belongs to "
                    "the loan-rejection class."
                )

            st.metric(
                "Estimated approval probability",
                f"{approval_probability * 100:.1f}%"
            )
            st.progress(approval_probability)

            with st.expander("View submitted applicant details"):
                st.dataframe(applicant, use_container_width=True)

            st.warning(
                "This is an experimental machine learning prediction, not "
                "a real lending decision or a guarantee of approval. "
                "The probability is a model estimate, not a verified "
                "financial risk score. Actual loan decisions require "
                "document verification, regulatory checks, and human review."
            )

        except Exception as exc:
            st.error(f"Unable to generate prediction: {exc}")

st.divider()
st.caption(
    "CreditWise | Machine Learning Project | "
    "Predictions are for educational and demonstration purposes only."
)
