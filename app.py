"""ICT336 Smart Savings Advisor — Streamlit deployment."""
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Smart Savings Advisor", page_icon="🏦", layout="wide")
BASE = Path(__file__).resolve().parent

@st.cache_resource
def load_models():
    savings = joblib.load(BASE / "savings_model.joblib")
    saver = joblib.load(BASE / "saver_model.joblib")
    return savings, saver

@st.cache_data
def load_customers():
    for name in ("customers_with_groups.csv", "customer_with_group.csv", "customer_with_groups.csv"):
        path = BASE / name
        if path.exists():
            return pd.read_csv(path)
    return None

def unpack_model(bundle, label):
    """Support the teacher's saved dictionary and a plain sklearn estimator."""
    if isinstance(bundle, dict):
        estimator = bundle.get("model")
        features = bundle.get("features")
        if estimator is None or not hasattr(estimator, "predict"):
            raise ValueError(f"{label} file has no usable 'model' entry.")
        if features is None:
            features = getattr(estimator, "feature_names_in_", None)
        return estimator, list(features) if features is not None else None
    if not hasattr(bundle, "predict"):
        raise ValueError(f"{label} file is not a model or a model dictionary.")
    features = getattr(bundle, "feature_names_in_", None)
    return bundle, list(features) if features is not None else None

BANK_NUMERIC = {
    "age": (35, 18, 100), "campaign": (1, 1, 100),
    "pdays": (999, 0, 999), "previous": (0, 0, 100),
    "emp.var.rate": (1.1, -5.0, 5.0), "cons.price.idx": (93.0, 70.0, 120.0),
    "cons.conf.idx": (-40.0, -80.0, 20.0), "euribor3m": (3.0, 0.0, 10.0),
    "nr.employed": (5100.0, 4000.0, 6000.0), "duration": (100, 0, 5000),
}
BANK_CATEGORIES = {
    "job": ["admin.", "blue-collar", "entrepreneur", "housemaid", "management", "retired", "self-employed", "services", "student", "technician", "unemployed", "unknown"],
    "marital": ["married", "single", "divorced", "unknown"],
    "education": ["university.degree", "high.school", "basic.9y", "professional.course", "basic.4y", "basic.6y", "illiterate", "unknown"],
    "default": ["no", "yes", "unknown"], "housing": ["no", "yes", "unknown"],
    "loan": ["no", "yes", "unknown"], "contact": ["cellular", "telephone"],
    "month": ["may", "jun", "jul", "aug", "nov", "apr", "oct", "sep", "mar", "dec"],
    "day_of_week": ["mon", "tue", "wed", "thu", "fri"],
    "poutcome": ["nonexistent", "failure", "success"],
}

st.title("🏦 Smart Savings Advisor")
st.caption("ICT336 | Bank deposit prediction and customer segmentation | Educational demonstration")

try:
    savings_bundle, saver_bundle = load_models()
    savings_estimator, savings_features = unpack_model(savings_bundle, "Savings")
    saver_estimator, saver_features = unpack_model(saver_bundle, "Saver")
except FileNotFoundError as exc:
    st.error(f"Missing model file: {exc.filename}. Place both .joblib files beside app.py.")
    st.stop()
except Exception as exc:
    st.error(f"Could not load a saved model: {exc}")
    st.info("Check the saved file structure and match the scikit-learn version used in Colab.")
    st.stop()

customers = load_customers()

st.header("1. Fixed-Term Deposit Prediction")
st.write("Enter the customer features used during model training.")
if savings_features is None:
    st.error("The savings model does not contain a feature list. Please save 'features' with the model in Colab.")
else:
    with st.form("deposit_form"):
        values = {}
        columns = st.columns(3)
        for i, name in enumerate(savings_features):
            with columns[i % 3]:
                if name in BANK_CATEGORIES:
                    options = (savings_bundle.get("choices", {}).get(name, BANK_CATEGORIES[name])
                               if isinstance(savings_bundle, dict) else BANK_CATEGORIES[name])
                    values[name] = st.selectbox(name, list(options))
                elif name in BANK_NUMERIC:
                    default, low, high = BANK_NUMERIC[name]
                    if isinstance(default, int):
                        values[name] = st.number_input(name, min_value=int(low), max_value=int(high), value=default, step=1)
                    else:
                        values[name] = st.number_input(name, min_value=float(low), max_value=float(high), value=float(default))
                else:
                    values[name] = st.text_input(name, value="unknown", help="Check this feature's required type in the training notebook")
        predict_deposit = st.form_submit_button("Predict deposit subscription", type="primary")

    if predict_deposit:
        input_df = pd.DataFrame([values], columns=savings_features)
        try:
            if hasattr(savings_estimator, "predict_proba"):
                probabilities = savings_estimator.predict_proba(input_df)[0]
                classes = list(savings_estimator.classes_)
                positive_index = next((i for i, label in enumerate(classes)
                                       if str(label).lower() in ("1", "yes", "true")), None)
                if positive_index is None:
                    st.error(f"Could not identify the Yes class. Saved model classes: {classes}")
                else:
                    chance = float(probabilities[positive_index])
                    st.metric("Estimated chance of subscription", f"{chance:.1%}")
                    st.progress(max(0.0, min(1.0, chance)))
                    st.caption("Model-estimated probability, not a guarantee of customer behavior.")
            else:
                predicted = savings_estimator.predict(input_df)[0]
                st.metric("Predicted subscription", str(predicted))
                st.info("This model does not support probability predictions.")
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            st.info("Check that feature names, categories and types match the training notebook.")

st.divider()
st.header("2. Customer Segmentation")
st.write("Enter annual income and spending score to identify a customer cluster.")
with st.form("cluster_form"):
    income = st.number_input("Annual income (thousands)", min_value=0.0, value=80.0, step=1.0)
    spending = st.slider("Spending score", min_value=1, max_value=100, value=20)
    predict_group = st.form_submit_button("Predict customer group", type="primary")

if predict_group:
    if saver_features is None or len(saver_features) != 2:
        st.error(f"The saved clustering model must specify two feature names. Found: {saver_features}")
    else:
        person = pd.DataFrame([{
            "Annual Income (k$)": income,
            "Spending Score (1-100)": spending,
        }])
        missing = [col for col in saver_features if col not in person.columns]
        if missing:
            st.error(f"Clustering features do not match the website inputs: {missing}")
        else:
            try:
                group = int(saver_estimator.predict(person[saver_features])[0])
                names = saver_bundle.get("names", {}) if isinstance(saver_bundle, dict) else {}
                tips = saver_bundle.get("tips", {}) if isinstance(saver_bundle, dict) else {}
                group_name = names.get(group, f"Group {group}") if isinstance(names, dict) else (names[group] if group < len(names) else f"Group {group}")
                st.success(f"Predicted customer group: {group} — {group_name}")
                if isinstance(tips, dict) and group in tips:
                    st.info(f"Savings tip: {tips[group]}")
                elif isinstance(tips, (list, tuple)) and group < len(tips):
                    st.info(f"Savings tip: {tips[group]}")
                st.caption("Cluster labels are assigned by the trained K-Means model.")
            except Exception as exc:
                st.error(f"Clustering prediction failed: {exc}")

st.divider()
st.header("3. Customer Groups Dataset")
if customers is None:
    st.info("Add customers_with_groups.csv to this folder to display the segmented customers.")
else:
    st.write(f"{len(customers):,} customer records")
    st.dataframe(customers, width="stretch")
    st.download_button("Download grouped customers CSV", data=customers.to_csv(index=False).encode("utf-8"), file_name="customers_with_groups.csv", mime="text/csv")
    numeric_cols = customers.select_dtypes(include="number").columns.tolist()
    group_col = next((c for c in customers.columns if c.lower() in ("group", "cluster", "cluster_id")), None)
    if group_col:
        summary_cols = [c for c in numeric_cols if c != group_col]
        if summary_cols:
            st.subheader("Average values by group")
            st.dataframe(customers.groupby(group_col)[summary_cols].mean().round(2), width="stretch")
