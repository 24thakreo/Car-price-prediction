
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AutoPrice | Car Intelligence",
    page_icon="🚘",
    layout="wide"
)

BASE = Path(__file__).resolve().parent
MODEL_FILE = BASE / "new_car_price_model.pkl"
CONFIG_FILE = BASE / "car_feature_config.json"
FEATURES_FILE = BASE / "car_model_features.json"
DATA_FILE = BASE / "cars_cleaned.csv"
COMPARISON_FILE = BASE / "car_model_comparison.csv"
BEST_MODEL_FILE = BASE / "best_car_model_name.txt"

TARGET = "Ex-Showroom_Price"

# ---------------- DESIGN ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
.stApp {
    background: #f4f7fb;
}
[data-testid="stSidebar"] {
    background: #101827;
}
[data-testid="stSidebar"] * {
    color: #e5eaf3 !important;
}
.hero {
    background: linear-gradient(120deg,#111827,#1d4ed8);
    padding: 30px;
    border-radius: 18px;
    color: white;
    margin: 15px 0 25px 0;
}
.hero h1 {
    color: white;
    font-size: 30px;
    margin: 0;
}
.hero p {
    color: #dbeafe;
    margin-bottom: 0;
}
.metric {
    background: white;
    border: 1px solid #e5eaf2;
    padding: 19px;
    border-radius: 15px;
    box-shadow: 0 4px 15px #0f172a08;
}
.metric-label {
    color: #64748b;
    font-size: 12px;
    text-transform: uppercase;
}
.metric-value {
    color: #111827;
    font-size: 25px;
    font-weight: 800;
    margin-top: 8px;
}
.result {
    background: #eaf2ff;
    border: 1px solid #bfdbfe;
    padding: 25px;
    border-radius: 16px;
    margin-top: 20px;
}
.result-price {
    color: #1d4ed8;
    font-size: 36px;
    font-weight: 800;
}
div[data-testid="stForm"] {
    background: white;
    padding: 20px;
    border-radius: 16px;
    border: 1px solid #e5eaf2;
}
</style>
""", unsafe_allow_html=True)


# ---------------- LOAD PROJECT FILES ----------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)

@st.cache_data
def load_config():
    with open(CONFIG_FILE, encoding="utf-8") as f:
        return json.load(f)

@st.cache_data
def load_features():
    with open(FEATURES_FILE, encoding="utf-8") as f:
        return json.load(f)

@st.cache_data
def load_dataset():
    data = pd.read_csv(DATA_FILE)
    data[TARGET] = pd.to_numeric(data[TARGET], errors="coerce")
    return data.dropna(subset=[TARGET])

@st.cache_data
def load_comparison():
    if COMPARISON_FILE.exists():
        return pd.read_csv(COMPARISON_FILE)
    return pd.DataFrame()


required = [
    MODEL_FILE, CONFIG_FILE, FEATURES_FILE, DATA_FILE
]
missing = [f.name for f in required if not f.exists()]

if missing:
    st.error("Missing project files: " + ", ".join(missing))
    st.stop()

try:
    model = load_model()
    config = load_config()
    features = load_features()
    df = load_dataset()
    comparison = load_comparison()
except Exception as e:
    st.error("Unable to load project files.")
    st.exception(e)
    st.stop()

if not isinstance(features, list):
    st.error("car_model_features.json must contain a list.")
    st.stop()

best_model = "Saved regression pipeline"
if BEST_MODEL_FILE.exists():
    best_model = BEST_MODEL_FILE.read_text(encoding="utf-8").strip() or best_model

def money(value):
    return "₹{:,.0f}".format(float(value))

def metric(label, value, note):
    st.markdown(f"""
    <div class="metric">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div style="color:#94a3b8;font-size:11px">{note}</div>
    </div>
    """, unsafe_allow_html=True)

def header(title, subtitle):
    st.markdown("""
    <h2 style="margin-bottom:0;color:#111827">
    Auto<span style="color:#2563eb">Price</span>
    </h2>
    """, unsafe_allow_html=True)
    st.caption(subtitle)
    st.markdown(f"""
    <div class="hero">
        <h1>{title}</h1>
        <p>New car price prediction and data intelligence dashboard</p>
    </div>
    """, unsafe_allow_html=True)


# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("# 🚘 AutoPrice")
    st.caption("NEW CAR INTELLIGENCE")
    st.markdown("---")

    page = st.radio(
        "NAVIGATION",
        [
            "Overview",
            "Price Prediction",
            "Market Analytics",
            "Model Insights",
            "Dataset Explorer"
        ]
    )

    st.markdown("---")
    st.caption("PROJECT MODEL")
    st.write(best_model)
    st.caption("Target: Ex-Showroom Price")


# =====================================================
# OVERVIEW
# =====================================================
if page == "Overview":

    header("New Car Price Intelligence", "Dashboard overview")

    prices = df[TARGET]

    a, b, c, d = st.columns(4)

    with a:
        metric("Total cars", f"{len(df):,}", "Dataset records")
    with b:
        metric("Average price", money(prices.mean()), "Average ex-showroom price")
    with c:
        metric("Median price", money(prices.median()), "Median ex-showroom price")
    with d:
        metric("Model features", len(features), "Input variables")

    st.write("")
    left, right = st.columns([1.4, 1])

    with left:
        st.subheader("Car price distribution")
        fig = px.histogram(
            df,
            x=TARGET,
            nbins=35,
            marginal="box",
            template="plotly_white",
            color_discrete_sequence=["#2563eb"]
        )
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Price statistics")
        summary = pd.DataFrame({
            "Measure": ["Minimum", "Maximum", "Average", "Median"],
            "Price": [
                money(prices.min()),
                money(prices.max()),
                money(prices.mean()),
                money(prices.median())
            ]
        })
        st.dataframe(summary, hide_index=True, use_container_width=True)

        st.subheader("Project workflow")
        st.markdown("""
        1. Data cleaning
        2. Statistical analysis
        3. Regression model training
        4. Model comparison
        5. Price prediction
        """)

    st.info(
        "Predictions are estimates based on historical dataset records, "
        "not live dealer quotations."
    )


# =====================================================
# PRICE PREDICTION
# =====================================================
elif page == "Price Prediction":

    header("Predict Your Car Price", "Enter car specifications")

    st.subheader("Vehicle configuration")

    if any(f not in config for f in features):
        st.error("Some model features are missing from car_feature_config.json.")
        st.stop()

    with st.form("car_prediction_form"):

        values = {}
        columns = st.columns(2)

        for i, feature in enumerate(features):
            item = config[feature]
            kind = str(item.get("type", "")).lower()

            with columns[i % 2]:

                if kind in ["number", "numeric", "float", "int"]:
                    minimum = float(item.get("min", 0))
                    maximum = float(item.get("max", 100))
                    default = float(item.get("default", minimum))

                    if maximum < minimum:
                        maximum = minimum + 100

                    default = min(max(default, minimum), maximum)

                    values[feature] = st.number_input(
                        feature.replace("_", " "),
                        min_value=minimum,
                        max_value=maximum,
                        value=default,
                        key="input_" + feature
                    )

                else:
                    options = [
                        str(x) for x in item.get("options", [])
                        if pd.notna(x)
                    ]

                    default = str(item.get(
                        "default",
                        options[0] if options else "Unknown"
                    ))

                    if not options:
                        options = [default]

                    if default not in options:
                        options.insert(0, default)

                    values[feature] = st.selectbox(
                        feature.replace("_", " "),
                        options,
                        index=options.index(default),
                        key="input_" + feature
                    )

        submitted = st.form_submit_button(
            "⚡ Predict Car Price",
            type="primary",
            use_container_width=True
        )

    if submitted:
        try:
            input_df = pd.DataFrame([values], columns=features)
            prediction = float(model.predict(input_df)[0])

            if not np.isfinite(prediction):
                raise ValueError("Invalid prediction returned by model.")

            st.session_state["last_prediction"] = prediction

            if "history" not in st.session_state:
                st.session_state["history"] = []

            st.session_state["history"].insert(0, {
                "Predicted Price": prediction,
                **values
            })
            st.session_state["history"] = st.session_state["history"][:20]

            st.markdown(f"""
            <div class="result">
                <div style="color:#475569;font-size:13px">
                ESTIMATED EX-SHOWROOM PRICE
                </div>
                <div class="result-price">{money(prediction)}</div>
                <div style="color:#64748b;font-size:12px">
                Model: {best_model}
                </div>
            </div>
            """, unsafe_allow_html=True)

            with st.expander("View selected car details"):
                st.dataframe(
                    pd.DataFrame([values]).T,
                    use_container_width=True
                )

        except Exception as e:
            st.error("Prediction failed.")
            st.exception(e)

    if st.session_state.get("history"):
        st.subheader("Recent prediction history")
        st.dataframe(
            pd.DataFrame(st.session_state["history"]),
            use_container_width=True,
            hide_index=True
        )

        if st.button("Clear history"):
            st.session_state["history"] = []
            st.rerun()


# =====================================================
# MARKET ANALYTICS
# =====================================================
elif page == "Market Analytics":

    header("Market Analytics", "Explore car price patterns")

    category_options = [
        c for c in ["Make", "Fuel_Type", "Body_Type", "Type", "Drivetrain"]
        if c in df.columns
    ]

    numeric_options = [
        c for c in [
            "Displacement", "Cylinders", "Power", "Torque",
            "Kerb_Weight", "Seating_Capacity", "Fuel_Tank_Capacity"
        ]
        if c in df.columns
    ]

    left, right = st.columns(2)

    with left:
        category = (
            st.selectbox("Compare by category", category_options)
            if category_options else None
        )

    with right:
        numeric = (
            st.selectbox("Compare specification with price", numeric_options)
            if numeric_options else None
        )

    a, b = st.columns(2)

    with a:
        st.subheader("Average price by category")

        if category:
            grouped = (
                df.groupby(category)[TARGET]
                .mean()
                .sort_values(ascending=False)
                .head(15)
                .reset_index()
            )

            fig = px.bar(
                grouped,
                x=category,
                y=TARGET,
                color=TARGET,
                color_continuous_scale="Blues",
                template="plotly_white"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

    with b:
        st.subheader("Specification vs price")

        if numeric:
            plot_df = df[[numeric, TARGET]].copy()
            plot_df[numeric] = pd.to_numeric(
                plot_df[numeric], errors="coerce"
            )
            plot_df = plot_df.dropna()

            fig = px.scatter(
                plot_df,
                x=numeric,
                y=TARGET,
                trendline="ols",
                opacity=.65,
                template="plotly_white",
                color_discrete_sequence=["#2563eb"]
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

    if category:
        st.subheader("Price range across categories")
        fig = px.box(
            df,
            x=category,
            y=TARGET,
            color=category,
            template="plotly_white"
        )
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)


# =====================================================
# MODEL INSIGHTS
# =====================================================
elif page == "Model Insights":

    header("Model Performance", "Regression model comparison")

    if comparison.empty:
        st.warning("car_model_comparison.csv is not available.")
    else:
        st.subheader("Model comparison table")
        st.dataframe(
            comparison,
            use_container_width=True,
            hide_index=True
        )

        model_col = next(
            (c for c in comparison.columns if c.lower() == "model"),
            None
        )
        r2_col = next(
            (c for c in comparison.columns
             if c.lower().replace(" ", "") in ["r2score", "r2"]),
            None
        )

        if model_col and r2_col:
            chart = comparison.copy()
            chart[r2_col] = pd.to_numeric(
                chart[r2_col], errors="coerce"
            )
            chart = chart.dropna(subset=[r2_col])

            fig = px.bar(
                chart.sort_values(r2_col, ascending=False),
                x=model_col,
                y=r2_col,
                text_auto=".3f",
                color=r2_col,
                color_continuous_scale="Blues",
                template="plotly_white",
                title="R² score comparison"
            )
            fig.update_layout(height=400, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

        st.success(f"Selected model: {best_model}")

    st.subheader("Understanding evaluation metrics")
    st.markdown("""
    - **MAE:** Average absolute prediction error.
    - **MSE:** Average squared prediction error.
    - **RMSE:** Square root of MSE.
    - **R²:** Measures how much price variation is explained by the model.
    """)


# =====================================================
# DATASET EXPLORER
# =====================================================
elif page == "Dataset Explorer":

    header("Dataset Explorer", "Explore cleaned car records")

    a, b, c = st.columns(3)

    with a:
        metric("Total records", f"{len(df):,}", "Cleaned records")
    with b:
        metric("Total columns", len(df.columns), "Features and target")
    with c:
        metric("Missing cells", int(df.isna().sum().sum()), "Loaded dataset")

    st.subheader("Filter dataset")

    filtered = df.copy()

    filter_columns = [
        c for c in ["Make", "Fuel_Type", "Body_Type", "Type"]
        if c in df.columns
    ]

    cols = st.columns(2)

    for i, col in enumerate(filter_columns[:2]):
        options = sorted(df[col].dropna().astype(str).unique())

        with cols[i]:
            selected = st.multiselect(
                f"Select {col}",
                options
            )

        if selected:
            filtered = filtered[
                filtered[col].astype(str).isin(selected)
            ]

    st.caption(f"Showing {len(filtered)} of {len(df)} records")

    st.dataframe(
        filtered,
        use_container_width=True,
        height=420
    )

    st.download_button(
        "⬇ Download filtered dataset",
        filtered.to_csv(index=False).encode("utf-8"),
        file_name="filtered_cars_dataset.csv",
        mime="text/csv"
    )

    st.subheader("Missing values report")

    missing = df.isnull().sum()
    missing = missing[missing > 0].reset_index()
    missing.columns = ["Column", "Missing values"]

    if missing.empty:
        st.success("No missing values in the loaded cleaned dataset.")
    else:
        st.dataframe(missing, hide_index=True, use_container_width=True)


# ---------------- FOOTER ----------------
st.markdown("---")
st.caption(
    "AutoPrice | New Car Price Prediction System | "
    "Machine Learning + Streamlit"
)
st.caption(
    "Predictions are estimates based on historical dataset information."
)
