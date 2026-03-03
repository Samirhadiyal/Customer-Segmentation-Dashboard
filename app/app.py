import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(page_title="Customer Segmentation Dashboard", layout="wide")

st.title("Customer Segmentation Analytics Dashboard")
st.info("Upload a dataset with customer related numeric attributes such as age, income, spending, purchase amount, etc.")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    # --------------------------------------------------
    # Dataset Preview
    # --------------------------------------------------
    st.subheader("Dataset Preview")
    with st.expander("View Uploaded Data"):
        st.dataframe(df.head())

    # --------------------------------------------------
    # Automatic Data Quality Report
    # --------------------------------------------------
    st.markdown("---")
    st.subheader("Automatic Data Quality Report")

    total_rows = df.shape[0]
    total_columns = df.shape[1]
    missing_values = df.isnull().sum().sum()
    duplicate_rows = df.duplicated().sum()
    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Rows", total_rows)
    col2.metric("Total Columns", total_columns)
    col3.metric("Missing Values", missing_values)
    col4.metric("Duplicate Rows", duplicate_rows)

    if missing_values > 0:
        st.warning("Dataset contains missing values. Consider cleaning data.")

    if len(numeric_columns) < 2:
        st.error("Dataset must contain at least two numeric columns.")
        st.stop()

    # --------------------------------------------------
    # Sidebar Controls
    # --------------------------------------------------
    with st.sidebar:
        st.header("Segmentation Settings")

        auto_exclude = [col for col in numeric_columns if "id" in col.lower()]
        usable_numeric_cols = [col for col in numeric_columns if col not in auto_exclude]

        selected_features = st.multiselect(
            "Select numeric attributes",
            usable_numeric_cols,
            default=usable_numeric_cols
        )

        segmentation_mode = st.selectbox(
            "Choose segmentation detail",
            ["Basic (3 Groups)", "Balanced (5 Groups)", "Detailed (7 Groups)", "Custom"]
        )

        if segmentation_mode == "Basic (3 Groups)":
            k = 3
        elif segmentation_mode == "Balanced (5 Groups)":
            k = 5
        elif segmentation_mode == "Detailed (7 Groups)":
            k = 7
        else:
            k = st.slider("Select number of groups", 2, 15, 4)

        run_button = st.button("Run Segmentation")

    # --------------------------------------------------
    # Segmentation Logic
    # --------------------------------------------------
    if run_button:

        if len(selected_features) < 2:
            st.error("Please select at least two numeric attributes.")
            st.stop()

        for col in selected_features:
            if df[col].nunique() <= 1:
                st.error(f"Column '{col}' has no variation.")
                st.stop()

        X = df[selected_features].copy()

        for col in X.select_dtypes(exclude=np.number).columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        kmeans = KMeans(n_clusters=k, random_state=42)
        df["Cluster"] = kmeans.fit_predict(X_scaled)

        summary = df.groupby("Cluster")[selected_features].mean()
        overall_means = df[selected_features].mean()

        # --------------------------------------------------
        # Intelligent Segment Naming
        # --------------------------------------------------
        cluster_names = {}

        for cluster_id in summary.index:
            group_data = summary.loc[cluster_id]
            name_parts = []

            for feature in selected_features:

                level = "High" if group_data[feature] > overall_means[feature] else "Low"

                if "age" in feature.lower():
                    name_parts.append("Young" if level == "Low" else "Older")

                elif "income" in feature.lower():
                    name_parts.append("High Income" if level == "High" else "Low Income")

                elif "spend" in feature.lower() or "score" in feature.lower():
                    name_parts.append("High Spenders" if level == "High" else "Low Spenders")

                else:
                    name_parts.append(f"{level} {feature}")

            cluster_names[cluster_id] = " | ".join(name_parts)

        df["Customer Group"] = df["Cluster"].map(cluster_names)

        st.success("Segmentation Completed Successfully")

        # --------------------------------------------------
        # Overview
        # --------------------------------------------------
        st.markdown("---")
        st.subheader("Overview")

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Customers", len(df))
        col2.metric("Groups Created", df["Customer Group"].nunique())
        col3.metric("Largest Group", df["Customer Group"].value_counts().idxmax())

        # --------------------------------------------------
        # Distribution
        # --------------------------------------------------
        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            fig1 = px.pie(
                df,
                names="Customer Group",
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            group_counts = df["Customer Group"].value_counts().reset_index()
            group_counts.columns = ["Customer Group", "Count"]

            fig_bar = px.bar(
                group_counts,
                x="Customer Group",
                y="Count",
                color="Customer Group",
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        # --------------------------------------------------
        # Scatter Visualization
        # --------------------------------------------------
        st.markdown("---")
        st.subheader("Group Visualization")

        fig2 = px.scatter(
            df,
            x=selected_features[0],
            y=selected_features[1],
            color="Customer Group",
            opacity=0.8,
            color_discrete_sequence=px.colors.qualitative.Set2
        )

        fig2.update_traces(marker=dict(size=9))

        fig2.update_layout(
            height=700,
            margin=dict(l=40, r=40, t=60, b=40),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            )
        )

        st.plotly_chart(fig2, use_container_width=True)

        # --------------------------------------------------
        # Feature Comparison
        # --------------------------------------------------
        st.markdown("---")
        st.subheader("Average Feature Comparison Across Groups")

        group_summary = df.groupby("Customer Group")[selected_features].mean().reset_index()

        for feature in selected_features:
            fig = px.bar(
                group_summary,
                x="Customer Group",
                y=feature,
                color="Customer Group",
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            st.plotly_chart(fig, use_container_width=True)

        # --------------------------------------------------
        # Live Insights
        # --------------------------------------------------
        st.markdown("---")
        st.subheader("Live Customer Group Insights")

        detailed_summary = df.groupby("Customer Group")[selected_features].mean()

        for group in detailed_summary.index:
            st.markdown(f"### {group}")
            group_data = detailed_summary.loc[group]

            insights = []
            for feature in selected_features:
                insights.append(f"Average {feature} is {round(group_data[feature],2)}")

            st.write(", ".join(insights))

        # --------------------------------------------------
        # Download
        # --------------------------------------------------
        st.markdown("---")
        csv = df.to_csv(index=False)

        st.download_button(
            "Download Segmented Data",
            csv,
            "segmented_customers.csv"
        )