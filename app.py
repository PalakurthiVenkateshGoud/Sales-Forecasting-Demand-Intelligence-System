"""
End-to-End Sales Forecasting & Demand Intelligence System
Interactive Streamlit Dashboard

Intern: Palakurthi Venkatesh Goud
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import zscore

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Optional heavier dependencies - degrade gracefully if not installed
try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False

try:
    from prophet import Prophet
    HAS_PROPHET = True
except Exception:
    HAS_PROPHET = False

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    HAS_SARIMA = True
except Exception:
    HAS_SARIMA = False


# ============================================================
# PAGE CONFIG (must be the first Streamlit command)
# ============================================================
st.set_page_config(
    page_title="Sales Forecasting & Demand Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_PATH_DEFAULT = "data/train.csv"


# ============================================================
# GLOBAL STYLES - premium KPI cards
# ============================================================
st.markdown(
    """
    <style>
    .kpi-card {
        border-radius: 16px;
        padding: 20px 22px;
        height: 150px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        overflow: hidden;
        box-sizing: border-box;
        box-shadow: 0 6px 16px rgba(0,0,0,0.28);
        border: 1px solid rgba(255,255,255,0.08);
        transition: transform 0.18s ease, box-shadow 0.18s ease;
        margin: 0 10px 22px 10px;
    }
    div[data-testid="column"] {
        padding-left: 4px;
        padding-right: 4px;
    }
    .kpi-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 10px 24px rgba(0,0,0,0.38);
    }
    .kpi-title {
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: rgba(255,255,255,0.85);
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .kpi-value {
        font-size: 25px;
        font-weight: 800;
        color: #ffffff;
        line-height: 1.15;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        overflow: hidden;
        word-break: break-word;
    }
    .kpi-subtitle {
        font-size: 12px;
        font-weight: 500;
        color: rgba(255,255,255,0.78);
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def kpi_card(title, value, subtitle, gradient):
    st.markdown(
        f"""
        <div class="kpi-card" style="background: linear-gradient(135deg, {gradient[0]}, {gradient[1]});">
            <div class="kpi-title">{title}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


KPI_GRADIENTS = {
    "sales":     ("#12B981", "#047857"),
    "avg":       ("#3B82F6", "#1D4ED8"),
    "orders":    ("#A855F7", "#7E22CE"),
    "customers": ("#FB923C", "#C2410C"),
    "products":  ("#22B8CF", "#0E7490"),
    "region":    ("#22C55E", "#15803D"),
    "category":  ("#F59E0B", "#B45309"),
    "segment":   ("#EC4899", "#BE185D"),
}


# ============================================================
# DATA LOADING & PREPARATION
# ============================================================
@st.cache_data(show_spinner=False)
def load_data(file):
    """Load the Superstore-style dataset from an uploaded file or default path."""
    df = pd.read_csv(file, encoding="latin-1")

    rename_map = {c: c.strip() for c in df.columns if c.strip() != c}
    if rename_map:
        df = df.rename(columns=rename_map)

    for date_col in ["Order Date", "Ship Date"]:
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")

    df = df.drop_duplicates()

    if "Sales" in df.columns:
        df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce")
    df = df.dropna(subset=["Sales", "Order Date"])

    df["Year"] = df["Order Date"].dt.year
    df["Month"] = df["Order Date"].dt.month
    df["Month Name"] = df["Order Date"].dt.strftime("%b")
    df["Quarter"] = df["Order Date"].dt.quarter
    df["Day"] = df["Order Date"].dt.day
    df["Day Name"] = df["Order Date"].dt.day_name()
    df["Weekday"] = df["Order Date"].dt.weekday

    return df


@st.cache_data(show_spinner=False)
def build_daily_sales(df):
    return (
        df.groupby("Order Date")["Sales"]
        .sum()
        .reset_index()
        .sort_values("Order Date")
    )


@st.cache_data(show_spinner=False)
def build_forecast_dataset(daily_sales):
    fdf = daily_sales.copy()
    fdf["Year"] = fdf["Order Date"].dt.year
    fdf["Month"] = fdf["Order Date"].dt.month
    fdf["Quarter"] = fdf["Order Date"].dt.quarter
    fdf["Week"] = fdf["Order Date"].dt.isocalendar().week.astype(int)
    fdf["Day"] = fdf["Order Date"].dt.day
    fdf["DayOfWeek"] = fdf["Order Date"].dt.dayofweek
    fdf["DayOfYear"] = fdf["Order Date"].dt.dayofyear
    fdf["IsWeekend"] = (fdf["DayOfWeek"] >= 5).astype(int)
    fdf["Lag_1"] = fdf["Sales"].shift(1)
    fdf["Lag_7"] = fdf["Sales"].shift(7)
    fdf["Lag_30"] = fdf["Sales"].shift(30)
    fdf["Rolling_7"] = fdf["Sales"].rolling(7).mean()
    fdf["Rolling_30"] = fdf["Sales"].rolling(30).mean()
    fdf = fdf.dropna().reset_index(drop=True)
    return fdf


FEATURES = [
    "Year", "Month", "Quarter", "Week", "Day", "DayOfWeek",
    "DayOfYear", "IsWeekend", "Lag_1", "Lag_7", "Lag_30",
    "Rolling_7", "Rolling_30",
]


@st.cache_data(show_spinner=False)
def compute_demand_score(df):
    product_score = (
        df.groupby("Product Name")
        .agg(
            Total_Sales=("Sales", "sum"),
            Orders=("Order ID", "count") if "Order ID" in df.columns else ("Sales", "count"),
            Avg_Sale=("Sales", "mean"),
        )
    )
    product_score["Demand Score"] = (
        product_score["Total_Sales"] * 0.6
        + product_score["Orders"] * 0.3
        + product_score["Avg_Sale"] * 0.1
    )
    return product_score.sort_values("Demand Score", ascending=False)


@st.cache_data(show_spinner=False)
def detect_anomalies(daily_sales):
    adf = daily_sales.copy()
    adf["Z_Score"] = zscore(adf["Sales"])
    adf["Anomaly"] = adf["Z_Score"].abs() > 3
    return adf


# ============================================================
# MODEL TRAINING (cached per model to avoid recompute)
# ============================================================
@st.cache_resource(show_spinner=False)
def train_ml_models(forecast_df):
    split = int(len(forecast_df) * 0.80)
    train = forecast_df.iloc[:split]
    test = forecast_df.iloc[split:]

    X_train, y_train = train[FEATURES], train["Sales"]
    X_test, y_test = test[FEATURES], test["Sales"]

    results = {}
    dates_test = test["Order Date"].values

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    results["Linear Regression"] = {
        "pred": lr_pred, "actual": y_test.values, "dates": dates_test,
        "mae": mean_absolute_error(y_test, lr_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, lr_pred)),
        "r2": r2_score(y_test, lr_pred),
    }

    rf = RandomForestRegressor(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    results["Random Forest"] = {
        "pred": rf_pred, "actual": y_test.values, "dates": dates_test,
        "mae": mean_absolute_error(y_test, rf_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, rf_pred)),
        "r2": r2_score(y_test, rf_pred),
        "importance": pd.DataFrame({"Feature": FEATURES, "Importance": rf.feature_importances_})
                        .sort_values("Importance", ascending=False),
    }

    if HAS_XGBOOST:
        xgb = XGBRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
        )
        xgb.fit(X_train, y_train)
        xgb_pred = xgb.predict(X_test)
        results["XGBoost"] = {
            "pred": xgb_pred, "actual": y_test.values, "dates": dates_test,
            "mae": mean_absolute_error(y_test, xgb_pred),
            "rmse": np.sqrt(mean_squared_error(y_test, xgb_pred)),
            "r2": r2_score(y_test, xgb_pred),
            "importance": pd.DataFrame({"Feature": FEATURES, "Importance": xgb.feature_importances_})
                            .sort_values("Importance", ascending=False),
        }

    return results


@st.cache_resource(show_spinner=False)
def train_prophet(daily_sales):
    if not HAS_PROPHET:
        return None
    pdf = daily_sales.rename(columns={"Order Date": "ds", "Sales": "y"})
    train_size = int(len(pdf) * 0.80)
    train, test = pdf.iloc[:train_size], pdf.iloc[train_size:]

    model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    model.fit(train)

    future = model.make_future_dataframe(periods=len(test), freq="D")
    forecast = model.predict(future)
    predictions = forecast.tail(len(test))[["ds", "yhat"]]

    mae = mean_absolute_error(test["y"], predictions["yhat"])
    rmse = np.sqrt(mean_squared_error(test["y"], predictions["yhat"]))
    r2 = r2_score(test["y"], predictions["yhat"])

    return {
        "pred": predictions["yhat"].values, "actual": test["y"].values,
        "dates": test["ds"].values, "mae": mae, "rmse": rmse, "r2": r2,
    }


@st.cache_resource(show_spinner=False)
def train_sarima(daily_sales):
    if not HAS_SARIMA:
        return None
    ts = daily_sales.set_index("Order Date").asfreq("D")
    ts["Sales"] = ts["Sales"].fillna(0)

    train_size = int(len(ts) * 0.80)
    train, test = ts.iloc[:train_size], ts.iloc[train_size:]

    model = SARIMAX(
        train["Sales"], order=(1, 1, 1), seasonal_order=(1, 1, 1, 7),
        enforce_stationarity=False, enforce_invertibility=False,
    )
    sarima = model.fit(disp=False)
    forecast = sarima.forecast(steps=len(test))

    mae = mean_absolute_error(test["Sales"], forecast)
    rmse = np.sqrt(mean_squared_error(test["Sales"], forecast))
    r2 = r2_score(test["Sales"], forecast)

    return {
        "pred": forecast.values, "actual": test["Sales"].values,
        "dates": test.index.values, "mae": mae, "rmse": rmse, "r2": r2,
    }


# ============================================================
# DATA SOURCE (sidebar - renders BELOW the native nav automatically)
# ============================================================
uploaded_file = st.sidebar.file_uploader("Upload Superstore CSV", type=["csv"])
data_source = uploaded_file if uploaded_file is not None else DATA_PATH_DEFAULT

try:
    df_raw = load_data(data_source)
    data_ok = True
except FileNotFoundError:
    st.sidebar.warning("No dataset found at `data/train.csv`. Please upload a CSV file to continue.")
    data_ok = False
except Exception as e:
    st.sidebar.error(f"Could not load data: {e}")
    data_ok = False

if data_ok:
    st.sidebar.caption(f"Rows loaded: {len(df_raw):,}")
    st.sidebar.caption(f"Date range: {df_raw['Order Date'].min().date()} to {df_raw['Order Date'].max().date()}")


# ============================================================
# PAGES
# ============================================================
def page_home():
    st.title("Executive Dashboard")
    st.caption("High-level KPIs summarizing overall business performance.")

    df = df_raw
    total_sales = df["Sales"].sum()
    avg_sales = df["Sales"].mean()
    total_orders = df["Order ID"].nunique() if "Order ID" in df.columns else len(df)
    total_customers = df["Customer ID"].nunique() if "Customer ID" in df.columns else None
    total_products = df["Product ID"].nunique() if "Product ID" in df.columns else (
        df["Product Name"].nunique() if "Product Name" in df.columns else None
    )

    best_region = df.groupby("Region")["Sales"].sum().idxmax() if "Region" in df.columns else "N/A"
    best_category = df.groupby("Category")["Sales"].sum().idxmax() if "Category" in df.columns else "N/A"
    best_segment = df.groupby("Segment")["Sales"].sum().idxmax() if "Segment" in df.columns else "N/A"

    row1 = st.columns(4, gap="large")
    with row1[0]:
        kpi_card("Total Sales", f"${total_sales:,.0f}", "Business Revenue", KPI_GRADIENTS["sales"])
    with row1[1]:
        kpi_card("Average Sale", f"${avg_sales:,.2f}", "Per Transaction", KPI_GRADIENTS["avg"])
    with row1[2]:
        kpi_card("Orders", f"{total_orders:,}", "Completed Orders", KPI_GRADIENTS["orders"])
    with row1[3]:
        kpi_card("Customers", f"{total_customers:,}" if total_customers is not None else "N/A",
                  "Unique Customers", KPI_GRADIENTS["customers"])

    row2 = st.columns(4, gap="large")
    with row2[0]:
        kpi_card("Products", f"{total_products:,}" if total_products is not None else "N/A",
                  "Available Products", KPI_GRADIENTS["products"])
    with row2[1]:
        kpi_card("Best Region", best_region, "Highest Sales", KPI_GRADIENTS["region"])
    with row2[2]:
        kpi_card("Top Category", best_category, "Best Performer", KPI_GRADIENTS["category"])
    with row2[3]:
        kpi_card("Top Segment", best_segment, "Highest Revenue", KPI_GRADIENTS["segment"])

    st.write("")
    st.write("")

    c1, c2 = st.columns(2)
    with c1:
        daily = build_daily_sales(df)
        fig = px.line(daily, x="Order Date", y="Sales", title="Daily Sales Trend")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        yearly = df.groupby("Year")["Sales"].sum().reset_index()
        fig = px.bar(yearly, x="Year", y="Sales", title="Yearly Sales", text_auto=".2s")
        st.plotly_chart(fig, use_container_width=True)


def page_sales_analytics():
    st.title("Sales Analytics")
    st.caption("Filter and explore sales performance across dimensions.")

    df = df_raw
    with st.expander("Filters", expanded=True):
        f1, f2, f3 = st.columns(3)
        f4, f5, f6 = st.columns(3)

        years = sorted(df["Year"].dropna().unique().tolist())
        sel_years = f1.multiselect("Year", years, default=years)

        regions = sorted(df["Region"].dropna().unique().tolist()) if "Region" in df.columns else []
        sel_regions = f2.multiselect("Region", regions, default=regions)

        states = sorted(df["State"].dropna().unique().tolist()) if "State" in df.columns else []
        sel_states = f3.multiselect("State", states, default=[])

        categories = sorted(df["Category"].dropna().unique().tolist()) if "Category" in df.columns else []
        sel_categories = f4.multiselect("Category", categories, default=categories)

        subcats = sorted(df["Sub-Category"].dropna().unique().tolist()) if "Sub-Category" in df.columns else []
        sel_subcats = f5.multiselect("Sub-Category", subcats, default=[])

        segments = sorted(df["Segment"].dropna().unique().tolist()) if "Segment" in df.columns else []
        sel_segments = f6.multiselect("Customer Segment", segments, default=segments)

    fdf = df[df["Year"].isin(sel_years)]
    if regions:
        fdf = fdf[fdf["Region"].isin(sel_regions)]
    if sel_states:
        fdf = fdf[fdf["State"].isin(sel_states)]
    if categories:
        fdf = fdf[fdf["Category"].isin(sel_categories)]
    if sel_subcats:
        fdf = fdf[fdf["Sub-Category"].isin(sel_subcats)]
    if segments:
        fdf = fdf[fdf["Segment"].isin(sel_segments)]

    if fdf.empty:
        st.warning("No records match the selected filters.")
        return

    st.markdown(f"**{len(fdf):,} records** match your filters — Total Sales: **${fdf['Sales'].sum():,.2f}**")

    c1, c2 = st.columns(2)
    with c1:
        monthly = fdf.groupby(["Year", "Month"])["Sales"].sum().reset_index()
        monthly["Date"] = pd.to_datetime(monthly[["Year", "Month"]].assign(day=1))
        fig = px.line(monthly.sort_values("Date"), x="Date", y="Sales", title="Monthly Sales", markers=True)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        yearly = fdf.groupby("Year")["Sales"].sum().reset_index()
        fig = px.bar(yearly, x="Year", y="Sales", title="Yearly Sales", text_auto=".2s")
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        if "Category" in fdf.columns:
            cat_sales = fdf.groupby("Category")["Sales"].sum().sort_values(ascending=False).reset_index()
            fig = px.bar(cat_sales, x="Category", y="Sales", title="Sales by Category", text_auto=".2s")
            st.plotly_chart(fig, use_container_width=True)
    with c4:
        if "Region" in fdf.columns:
            reg_sales = fdf.groupby("Region")["Sales"].sum().sort_values(ascending=False).reset_index()
            fig = px.bar(reg_sales, x="Region", y="Sales", title="Sales by Region", text_auto=".2s")
            st.plotly_chart(fig, use_container_width=True)

    c5, c6 = st.columns(2)
    with c5:
        if "Product Name" in fdf.columns:
            top_products = fdf.groupby("Product Name")["Sales"].sum().sort_values(ascending=False).head(10).reset_index()
            fig = px.bar(top_products.sort_values("Sales"), x="Sales", y="Product Name",
                         orientation="h", title="Top 10 Products by Sales")
            st.plotly_chart(fig, use_container_width=True)
    with c6:
        if "Customer Name" in fdf.columns:
            top_customers = fdf.groupby("Customer Name")["Sales"].sum().sort_values(ascending=False).head(10).reset_index()
            fig = px.bar(top_customers.sort_values("Sales"), x="Sales", y="Customer Name",
                         orientation="h", title="Top 10 Customers by Sales")
            st.plotly_chart(fig, use_container_width=True)


def page_forecasting():
    st.title("Forecasting")
    st.caption("Compare five forecasting approaches trained on historical daily sales.")

    df = df_raw
    daily = build_daily_sales(df)
    forecast_df = build_forecast_dataset(daily)

    if len(forecast_df) < 40:
        st.warning("Not enough historical daily records to train reliable forecasting models.")
        return

    model_options = ["Linear Regression", "Random Forest"]
    if HAS_XGBOOST:
        model_options.append("XGBoost")
    if HAS_PROPHET:
        model_options.append("Prophet")
    if HAS_SARIMA:
        model_options.append("SARIMA")

    with st.spinner("Training models..."):
        ml_results = train_ml_models(forecast_df)
        all_results = dict(ml_results)
        if HAS_PROPHET:
            prophet_res = train_prophet(daily)
            if prophet_res:
                all_results["Prophet"] = prophet_res
        if HAS_SARIMA:
            sarima_res = train_sarima(daily)
            if sarima_res:
                all_results["SARIMA"] = sarima_res

    selected_model = st.selectbox("Select Forecasting Model", model_options)
    res = all_results.get(selected_model)

    if res is None:
        st.info(f"{selected_model} is not available in this environment (optional dependency not installed).")
        return

    m1, m2, m3 = st.columns(3)
    m1.metric("MAE", f"{res['mae']:,.2f}")
    m2.metric("RMSE", f"{res['rmse']:,.2f}")
    m3.metric("R² Score", f"{res['r2']:.4f}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=res["dates"], y=res["actual"], mode="lines", name="Actual", line=dict(color="crimson")))
    fig.add_trace(go.Scatter(x=res["dates"], y=res["pred"], mode="lines", name="Predicted", line=dict(color="royalblue")))
    fig.update_layout(title=f"{selected_model}: Actual vs Predicted Daily Sales",
                       xaxis_title="Date", yaxis_title="Sales ($)")
    st.plotly_chart(fig, use_container_width=True)

    if "importance" in res:
        st.subheader("Feature Importance")
        imp = res["importance"].head(10).sort_values("Importance")
        fig2 = px.bar(imp, x="Importance", y="Feature", orientation="h")
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Model Comparison")
    comp = pd.DataFrame([
        {"Model": name, "MAE": r["mae"], "RMSE": r["rmse"], "R²": r["r2"]}
        for name, r in all_results.items()
    ]).sort_values("RMSE")
    best_model = comp.iloc[0]["Model"]
    st.dataframe(comp.style.format({"MAE": "{:,.2f}", "RMSE": "{:,.2f}", "R²": "{:.4f}"}), use_container_width=True)
    st.success(f"Best performing model (lowest RMSE): {best_model}")


def page_anomaly_detection():
    st.title("Anomaly Detection")
    st.caption("Z-score based detection of unusual daily sales spikes (|Z| > 3).")

    df = df_raw
    daily = build_daily_sales(df)
    adf = detect_anomalies(daily)

    total_days = len(adf)
    n_anomalies = int(adf["Anomaly"].sum())

    m1, m2 = st.columns(2)
    m1.metric("Total Days Analyzed", f"{total_days:,}")
    m2.metric("Anomalies Detected", f"{n_anomalies:,}")

    anomalies = adf[adf["Anomaly"]]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=adf["Order Date"], y=adf["Sales"], mode="lines",
                              name="Daily Sales", line=dict(color="steelblue")))
    fig.add_trace(go.Scatter(x=anomalies["Order Date"], y=anomalies["Sales"], mode="markers",
                              name="Anomaly", marker=dict(color="red", size=9)))
    fig.update_layout(title="Sales Anomaly Detection", xaxis_title="Date", yaxis_title="Sales ($)")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top Anomalies")
    st.dataframe(
        anomalies.sort_values("Sales", ascending=False).head(20).style.format({
            "Sales": "${:,.2f}", "Z_Score": "{:.2f}"
        }),
        use_container_width=True,
    )


def page_demand_intelligence():
    st.title("Demand Intelligence")
    st.caption("Identify the products, regions, states, and segments driving the most demand.")

    df = df_raw
    if "Product Name" in df.columns:
        product_score = compute_demand_score(df)
        st.subheader("Top 20 Products by Demand Score")
        st.dataframe(
            product_score.head(20).style.format({
                "Total_Sales": "${:,.2f}", "Avg_Sale": "${:,.2f}", "Demand Score": "{:,.1f}"
            }),
            use_container_width=True,
        )

    c1, c2 = st.columns(2)
    with c1:
        if "State" in df.columns:
            top_states = df.groupby("State")["Sales"].sum().sort_values(ascending=False).head(10).reset_index()
            fig = px.bar(top_states.sort_values("Sales"), x="Sales", y="State", orientation="h",
                         title="Top 10 States by Demand")
            st.plotly_chart(fig, use_container_width=True)
    with c2:
        if "Region" in df.columns:
            region_demand = df.groupby("Region")["Sales"].sum().sort_values(ascending=False).reset_index()
            fig = px.bar(region_demand, x="Region", y="Sales", title="Regional Demand", text_auto=".2s")
            st.plotly_chart(fig, use_container_width=True)

    if "Segment" in df.columns:
        st.subheader("Demand by Customer Segment")
        seg = df.groupby("Segment")["Sales"].agg(["sum", "mean", "count"]).sort_values("sum", ascending=False)
        seg.columns = ["Total Sales", "Average Sale", "Order Count"]
        st.dataframe(seg.style.format({"Total Sales": "${:,.2f}", "Average Sale": "${:,.2f}"}),
                     use_container_width=True)


def page_recommendations():
    st.title("AI Business Recommendation Engine")
    st.caption("Rule-based recommendations generated directly from your data's statistics.")

    df = df_raw
    cards = []

    if "Category" in df.columns:
        top_category = df.groupby("Category")["Sales"].sum().idxmax()
        cards.append(("Inventory", f"Increase inventory investment in **{top_category}** — it generates the highest sales of any category."))

    if "Region" in df.columns:
        best_region = df.groupby("Region")["Sales"].sum().idxmax()
        cards.append(("Marketing", f"Prioritize marketing campaigns in the **{best_region}** region due to its highest revenue contribution."))

    if "Segment" in df.columns:
        best_segment = df.groupby("Segment")["Sales"].sum().idxmax()
        cards.append(("Customer", f"Focus customer loyalty programs on the **{best_segment}** segment."))

    if "Ship Mode" in df.columns:
        best_ship = df.groupby("Ship Mode")["Sales"].sum().idxmax()
        cards.append(("Logistics", f"Maintain operational efficiency for **{best_ship}** shipping — it handles the majority of business volume."))

    if "State" in df.columns:
        best_state = df.groupby("State")["Sales"].sum().idxmax()
        cards.append(("Expansion", f"Expand sales operations in **{best_state}** since it contributes the highest revenue."))

    if "Product Name" in df.columns:
        product_score = compute_demand_score(df)
        best_product = product_score.index[0]
        cards.append(("Top Product", f"Maintain higher inventory levels for **{best_product}** — it has the highest demand score."))

    cols = st.columns(2)
    for i, (label, text) in enumerate(cards):
        with cols[i % 2]:
            st.info(f"**{label}**\n\n{text}")

    st.subheader("Overall Business Strategy")
    st.markdown(
        "- Increase inventory for high-demand products.\n"
        "- Focus marketing spend on high-performing regions.\n"
        "- Strengthen retention programs for the top-performing customer segment.\n"
        "- Optimize logistics around the dominant shipping method.\n"
        "- Monitor lower-performing regions and states for growth opportunities."
    )


def page_about():
    st.title("About This Project")
    st.markdown("""
### Problem Statement
Retail and e-commerce companies rely on accurate demand forecasting to ensure the right products
are available at the right time, in the right locations. This project builds an end-to-end
analytics system that forecasts sales, detects abnormal demand, scores product-level demand,
and generates actionable business recommendations.

### Dataset
- **Primary:** Superstore Sales Dataset — four years of daily, order-level retail transactions
  across multiple regions, categories, and customer segments.

### Analytics Cycle
1. Data Loading & Quality Assessment
2. Exploratory Data Analysis
3. Time Series Analysis
4. Feature Engineering (calendar, lag, and rolling-window features)
5. Machine Learning & Statistical Forecasting
6. Anomaly Detection (Z-score based)
7. Demand Intelligence Scoring
8. AI-Generated Business Recommendations
9. Interactive Dashboard Delivery

### Technologies Used
`Python` · `Pandas` · `NumPy` · `Scikit-learn` · `XGBoost` · `Prophet` · `Statsmodels` ·
`Plotly` · `Streamlit`

### Models Used
Linear Regression · Random Forest · XGBoost · Prophet · SARIMA

### Author
**Palakurthi Venkatesh Goud**
B.Tech Computer Science & Data Science, MLR Institute of Technology & Management

### GitHub
_Repository link to be added._
""")


def page_need_data():
    st.title("End-to-End Sales Forecasting & Demand Intelligence System")
    st.info("Upload your Superstore-style sales CSV from the sidebar to get started.")


# ============================================================
# NATIVE NAVIGATION (single source of truth - replaces any
# auto-discovered pages/ nav and removes duplicate sidebar lists)
# ============================================================
if data_ok:
    pages = {
        "App": [
            st.Page(page_home, title="Executive Dashboard", default=True),
            st.Page(page_sales_analytics, title="Sales Analytics"),
            st.Page(page_forecasting, title="Forecasting"),
            st.Page(page_anomaly_detection, title="Anomaly Detection"),
            st.Page(page_demand_intelligence, title="Demand Intelligence"),
            st.Page(page_recommendations, title="AI Recommendations"),
            st.Page(page_about, title="About Project"),
        ]
    }
else:
    pages = {"App": [st.Page(page_need_data, title="Get Started", default=True)]}

pg = st.navigation(pages)
pg.run()