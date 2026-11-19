import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

st.set_page_config(page_title="Smart House Price Prediction", page_icon="🏡", layout="wide")

st.title("🏡 Smart House Price Prediction")
st.caption("Compare Simple, Multiple and Polynomial Linear Regression models using California housing data.")

@st.cache_data
def load_data():
    return pd.read_csv("housing.csv")

@st.cache_resource
def train_models(data):
    data = data.drop_duplicates().copy()
    required = {
        "longitude", "latitude", "housing_median_age", "total_rooms",
        "total_bedrooms", "population", "households", "median_income",
        "ocean_proximity", "median_house_value"
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError("housing.csv is missing columns: " + ", ".join(sorted(missing)))

    X = data.drop(columns=["median_house_value"])
    y = data["median_house_value"]
    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    simple_model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("regression", LinearRegression())
    ])
    simple_model.fit(X_train[["median_income"]], y_train)
    simple_pred = simple_model.predict(X_test[["median_income"]])

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])
    multiple_model = Pipeline([
        ("preprocessor", preprocessor),
        ("regression", LinearRegression())
    ])
    multiple_model.fit(X_train, y_train)
    multiple_pred = multiple_model.predict(X_test)

    poly_income = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("poly", PolynomialFeatures(degree=2, include_bias=False))
    ])
    other_numeric = [c for c in numeric_features if c != "median_income"]
    poly_preprocessor = ColumnTransformer([
        ("income", poly_income, ["median_income"]),
        ("num", numeric_transformer, other_numeric),
        ("cat", categorical_transformer, categorical_features)
    ])
    polynomial_model = Pipeline([
        ("preprocessor", poly_preprocessor),
        ("regression", LinearRegression())
    ])
    polynomial_model.fit(X_train, y_train)
    poly_pred = polynomial_model.predict(X_test)

    models = {
        "Simple Linear Regression": (simple_model, simple_pred),
        "Multiple Linear Regression": (multiple_model, multiple_pred),
        "Polynomial Regression": (polynomial_model, poly_pred),
    }
    rows = []
    for name, (_, pred) in models.items():
        mse = mean_squared_error(y_test, pred)
        rows.append({
            "Model": name,
            "MSE": mse,
            "RMSE": np.sqrt(mse),
            "MAE": mean_absolute_error(y_test, pred),
            "R2 Score": r2_score(y_test, pred)
        })
    metrics = pd.DataFrame(rows)
    best_name = metrics.loc[metrics["MSE"].idxmin(), "Model"]
    return data, models, metrics, best_name

try:
    data = load_data()
    data, models, metrics, best_model_name = train_models(data)
except FileNotFoundError:
    st.error("Dataset not found. Upload housing.csv to the same GitHub repository as app.py.")
    st.stop()
except Exception as exc:
    st.error(f"Could not load or train the models: {exc}")
    st.stop()

st.success(f"Best model based on the lowest test MSE: {best_model_name}")

tab1, tab2, tab3 = st.tabs(["Predict House Price", "Model Comparison", "About Dataset"])

with tab1:
    st.subheader("Enter house details")
    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            longitude = st.number_input("Longitude", min_value=-125.0, max_value=-113.0, value=-122.23, step=0.01)
            latitude = st.number_input("Latitude", min_value=32.0, max_value=42.0, value=37.88, step=0.01)
            age = st.number_input("Housing Median Age (years)", min_value=0.0, value=30.0, step=1.0)
        with c2:
            rooms = st.number_input("Total Rooms", min_value=1.0, value=2000.0, step=100.0)
            bedrooms = st.number_input("Total Bedrooms", min_value=0.0, value=400.0, step=10.0)
            population = st.number_input("Population", min_value=1.0, value=1000.0, step=50.0)
        with c3:
            households = st.number_input("Households", min_value=1.0, value=350.0, step=10.0)
            income = st.number_input("Median Income (in tens of thousands)", min_value=0.0, value=5.0, step=0.1)
            ocean_options = sorted(data["ocean_proximity"].dropna().astype(str).unique())
            ocean = st.selectbox("Ocean Proximity", ocean_options)

        submitted = st.form_submit_button("Predict House Price", type="primary")

    if submitted:
        new_house = pd.DataFrame([{
            "longitude": longitude,
            "latitude": latitude,
            "housing_median_age": age,
            "total_rooms": rooms,
            "total_bedrooms": bedrooms,
            "population": population,
            "households": households,
            "median_income": income,
            "ocean_proximity": ocean
        }])
        model, _ = models[best_model_name]
        if best_model_name == "Simple Linear Regression":
            predicted_price = model.predict(new_house[["median_income"]])[0]
        else:
            predicted_price = model.predict(new_house)[0]
        st.metric("Estimated Median House Value", f"${predicted_price:,.2f}")
        st.caption(f"Prediction generated using {best_model_name}. This is an estimate, not a property valuation.")

with tab2:
    st.subheader("Regression model performance")
    st.dataframe(metrics.style.format({
        "MSE": "{:,.2f}", "RMSE": "${:,.2f}",
        "MAE": "${:,.2f}", "R2 Score": "{:.4f}"
    }), use_container_width=True, hide_index=True)
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.barplot(data=metrics, x="Model", y="R2 Score", ax=ax)
    ax.set_title("R² Score Comparison")
    ax.tick_params(axis="x", rotation=12)
    st.pyplot(fig)
    st.caption("The best model is selected automatically using the lowest Mean Squared Error (MSE) on the test set.")

with tab3:
    st.subheader("California Housing Dataset")
    a, b, c = st.columns(3)
    a.metric("Dataset rows", f"{len(data):,}")
    b.metric("Input features", "9")
    c.metric("Target", "Median house value")
    st.write("The project uses California housing attributes including location, housing age, rooms, bedrooms, population, households, median income and ocean proximity.")
    st.write("Preview")
    st.dataframe(data.head(10), use_container_width=True)
