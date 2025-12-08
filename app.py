import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="Customer Segmentation MATCH NOTEBOOK",
                   page_icon="📊",
                   layout="wide")

st.title("📊 Customer Segmentation Dashboard (MATCH NOTEBOOK VERSION)")


# ======================================================================
# 1. LOAD + FULL PREPROCESSING (100% MATCH NOTEBOOK)
# ======================================================================
@st.cache_data
def load_and_clean():
    # Load dataset
    url = "https://drive.google.com/uc?id=1V2IdrRQ8XQmJlzb2PAlJw0ziJQg-13QW"
    df = pd.read_csv(url)

    # MATCH NOTEBOOK:
    df["Description"] = df["Description"].fillna("Tidak ada deskripsi")
    df = df.dropna(subset=["Customer ID"])
    df["Customer ID"] = df["Customer ID"].astype(str)

    # Duplicate removal
    df = df.drop_duplicates()

    # Fix datetime
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    # REMOVE NEGATIVE quantity & price
    df = df[(df["Quantity"] > 0) & (df["Price"] > 0)]

    # IQR OUTLIER REMOVAL FOR QUANTITY
    Q1 = df["Quantity"].quantile(0.25)
    Q3 = df["Quantity"].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df["Quantity"] >= Q1 - 1.5*IQR) & (df["Quantity"] <= Q3 + 1.5*IQR)]

    # QUANTILE OUTLIER REMOVAL FOR PRICE (1% – 99%)
    p1 = df["Price"].quantile(0.01)
    p99 = df["Price"].quantile(0.99)
    df_clean_price = df[(df["Price"] >= p1) & (df["Price"] <= p99)]

    return df, df_clean_price


df, df_clean_price = load_and_clean()


# ======================================================================
# SIDEBAR NAVIGATION
# ======================================================================
menu = st.sidebar.radio(
    "Navigation",
    [
        "Dataset Overview",
        "EDA Visualizations",
        "RFM Analysis",
        "Clustering",
        "Business Insights"
    ]
)

# ======================================================================
# PAGE 1 — DATASET OVERVIEW
# ======================================================================
if menu == "Dataset Overview":
    st.header("📂 Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Top 10 Countries")
    st.write(df["Country"].value_counts().head(10))


# ======================================================================
# PAGE 2 — VISUALIZATIONS (MATCH NOTEBOOK)
# ======================================================================
elif menu == "EDA Visualizations":

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Distribusi Quantity",
        "Distribusi Price (Cleaned Quantile)",
        "Jumlah Transaksi per Hari",
        "Top 10 Produk",
        "Transaksi per Jam",
        "Transaksi per Bulan",
        "PCA Preview (MATCH NOTEBOOK)"
    ])

    # DISTRIBUSI QUANTITY
    with tab1:
        st.subheader("Distribusi Quantity")
        fig = plt.figure(figsize=(8,4))
        sns.histplot(df["Quantity"], kde=True)
        st.pyplot(fig)

    # DISTRIBUSI PRICE — CLEANED 1%–99%
    with tab2:
        st.subheader("Distribusi Price (Quantile 1%–99%)")
        fig = plt.figure(figsize=(8,4))
        sns.histplot(df_clean_price["Price"], kde=True, bins=50)
        st.pyplot(fig)

    # TRANSAKSI PER HARI
    with tab3:
        st.subheader("Jumlah Transaksi per Hari")
        df["InvoiceDate_only"] = df["InvoiceDate"].dt.date
        trans_day = df.groupby("InvoiceDate_only").size()

        fig = plt.figure(figsize=(12,4))
        plt.plot(trans_day)
        st.pyplot(fig)

    # TOP PRODUK
    with tab4:
        st.subheader("Top 10 Produk")
        prod = df.groupby("Description")["Quantity"].sum().sort_values(ascending=False).head(10)
        st.write(prod)

    # TRANSAKSI PER JAM
    with tab5:
        st.subheader("Transaksi per Jam")
        df["Hour"] = df["InvoiceDate"].dt.hour

        fig = plt.figure(figsize=(7,4))
        sns.countplot(x="Hour", data=df)
        st.pyplot(fig)

    # TRANSAKSI PER BULAN
    with tab6:
        st.subheader("Transaksi per Bulan")
        df["Month"] = df["InvoiceDate"].dt.month

        fig = plt.figure(figsize=(7,4))
        sns.countplot(x="Month", data=df)
        st.pyplot(fig)

    # PCA PREVIEW
    with tab7:
        st.subheader("PCA Preview — MATCH NOTEBOOK")

        sample = df.sample(3000, random_state=42)

        scaler = StandardScaler()
        scaled = scaler.fit_transform(sample[["Quantity","Price"]])

        pca = PCA(2)
        comp = pca.fit_transform(scaled)

        fig = plt.figure(figsize=(7,5))
        sns.scatterplot(x=comp[:,0], y=comp[:,1], alpha=0.4)
        st.pyplot(fig)


# ======================================================================
# PAGE 3 — RFM ANALYSIS (MATCH NOTEBOOK)
# ======================================================================
elif menu == "RFM Analysis":

    st.header("📊 RFM — MATCH NOTEBOOK")

    latest = df["InvoiceDate"].max()
    df["Recency"] = (latest - df["InvoiceDate"]).dt.days

    rfm = df.groupby("Customer ID").agg({
        "Recency": "min",
        "InvoiceDate": "nunique",
        "Price": "sum"
    }).reset_index()

    rfm.columns = ["Customer ID", "Recency", "Frequency", "Monetary"]

    # IQR outlier removal for RFM
    def remove_iqr(data, col):
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        return data[(data[col] >= Q1 - 1.5*IQR) & (data[col] <= Q3 + 1.5*IQR)]

    rfm = remove_iqr(rfm, "Recency")
    rfm = remove_iqr(rfm, "Frequency")
    rfm = remove_iqr(rfm, "Monetary")

    st.session_state["rfm"] = rfm

    st.subheader("Preview RFM")
    st.dataframe(rfm.head())

    # Subplot Distribusi
    fig, ax = plt.subplots(1,3, figsize=(15,4))

    ax[0].hist(rfm["Recency"], bins=20, color="skyblue")
    ax[0].set_title("Distribusi Recency")

    ax[1].hist(rfm["Frequency"], bins=20, color="orange")
    ax[1].set_title("Distribusi Frequency")

    ax[2].hist(rfm["Monetary"], bins=20, color="green")
    ax[2].set_title("Distribusi Monetary")

    st.pyplot(fig)

# ======================================================================
# PAGE 4 — CLUSTERING (FINAL MATCH NOTEBOOK)
# ======================================================================
elif menu == "Clustering":

    st.header("🎯 KMeans Clustering — MATCH NOTEBOOK")

    if "rfm" not in st.session_state:
        st.warning("Jalankan RFM Analysis dulu.")
        st.stop()

    # COPY EXACT NOTEBOOK BEHAVIOR
    rfm = st.session_state["rfm"].copy()

    # 1. SCALING
    scaler = StandardScaler()
    scaled = scaler.fit_transform(rfm[["Recency", "Frequency", "Monetary"]])

    # 2. PCA EXACT
    pca = PCA(n_components=2)
    comps = pca.fit_transform(scaled)
    rfm["PCA1"], rfm["PCA2"] = comps[:,0], comps[:,1]

    # 3. SAME K AS NOTEBOOK (default = 3)
    k = st.slider("Jumlah cluster", 2, 10, 3)

    # 4. KMeans EXACT match
    kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300,
                    n_init=10, random_state=42)
    rfm["Cluster"] = kmeans.fit_predict(scaled)

    # SIMPAN
    st.session_state["clustered"] = rfm

    # PCA PLOT
    st.subheader("Visualisasi PCA — MATCH NOTEBOOK")
    fig = plt.figure(figsize=(8,6))
    sns.scatterplot(
        x=rfm["PCA1"], y=rfm["PCA2"],
        hue=rfm["Cluster"], palette="viridis",
        s=30, alpha=0.7
    )
    plt.title("Cluster Visualization after PCA (MATCH NOTEBOOK)")
    st.pyplot(fig)

    st.write("Jumlah anggota per cluster:")
    st.write(rfm["Cluster"].value_counts())

# ======================================================================
# PAGE 5 — BUSINESS INSIGHTS (FINAL MATCH NOTEBOOK)
# ======================================================================
elif menu == "Business Insights":

    if "clustered" not in st.session_state:
        st.warning("Jalankan Clustering dulu.")
        st.stop()

    rfm = st.session_state["clustered"]

    st.header("💡 Business Insights — MATCH NOTEBOOK")

    # === EXACT SEGMENTATION LOGIC (MATCH NOTEBOOK) ===
    rec_med = rfm["Recency"].median()
    freq_med = rfm["Frequency"].median()
    mon_med = rfm["Monetary"].median()

    def assign_segment(row):
        if row["Recency"] < rec_med and row["Frequency"] > freq_med:
            return "Champions"
        elif row["Frequency"] > freq_med:
            return "Loyalists"
        elif row["Monetary"] > mon_med:
            return "Big Spenders, low frequency"
        elif row["Recency"] > rec_med:
            return "At-risk"
        else:
            return "New but promising"

    rfm["Segment"] = rfm.apply(assign_segment, axis=1)

    # === HEATMAP EXACT MATCH ===
    pivot = pd.crosstab(rfm["Cluster"], rfm["Segment"])

    st.subheader("Distribusi Segmen Pelanggan per Cluster — MATCH NOTEBOOK")
    fig = plt.figure(figsize=(10,6))
    sns.heatmap(pivot, annot=True, cmap="YlGnBu", fmt="g")
    st.pyplot(fig)

    st.subheader("Interpretasi:")
    st.write("""
    • **Champions** → Aktif & profitable, fokuskan loyalty program  
    • **Loyalists** → Sering beli, target upselling/cross-selling  
    • **Big Spenders** → Nilai belanja tinggi tapi jarang beli  
    • **At-risk** → Lama tidak transaksi, perlu reactivation campaign  
    • **New but promising** → Pelanggan baru yang potensial  
    """)

