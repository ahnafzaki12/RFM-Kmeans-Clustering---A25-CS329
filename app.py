import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="Customer Segmentation Dashboard", 
                   page_icon="📊",
                   layout="wide")

st.title("📊 Customer Segmentation Dashboard (RFM + KMeans)")

# ======================================================================
# LOAD DATA
# ======================================================================
@st.cache_data
def load_data():
    url = "https://drive.google.com/uc?id=1V2IdrRQ8XQmJlzb2PAlJw0ziJQg-13QW"
    df = pd.read_csv(url)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df = df.dropna(subset=["Customer ID"])
    df["Customer ID"] = df["Customer ID"].astype(str)
    df = df[(df["Quantity"] > 0) & (df["Price"] > 0)]
    return df

df = load_data()

# ======================================================================
# SIDEBAR
# ======================================================================
menu = st.sidebar.radio(
    "Navigation Menu",
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
    st.dataframe(df.head(20))

    st.subheader("Top 10 Countries by Transaction Count")
    st.write(df["Country"].value_counts().head(10))


# ======================================================================
# PAGE 2 — VISUALISASI EDA
# ======================================================================
elif menu == "EDA Visualizations":

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Distribusi Quantity",
        "Distribusi Price",
        "Transaksi per Hari",
        "Top Produk",
        "Transaksi per Jam",
        "Transaksi per Bulan",
        "PCA Clustering Preview"
    ])

    # --------------------- Distribusi Quantity --------------------------
    with tab1:
        st.subheader("Distribusi Quantity")
        fig = plt.figure(figsize=(8,4))
        sns.histplot(df["Quantity"], kde=True)
        plt.title("Distribusi Quantity")
        st.pyplot(fig)

    # --------------------- Distribusi Price -----------------------------
    with tab2:
        st.subheader("Distribusi Harga (Cleaned 1% - 99%)")
        p1, p99 = df["Price"].quantile([0.01, 0.99])
        df_clean = df[(df["Price"] >= p1) & (df["Price"] <= p99)]

        fig = plt.figure(figsize=(8,4))
        sns.histplot(df_clean["Price"], kde=True, bins=50)
        plt.title("Distribusi Price")
        st.pyplot(fig)

    # --------------------- Transaksi per Hari ---------------------------
    with tab3:
        st.subheader("Jumlah Transaksi per Hari")
        df["InvoiceDate_only"] = df["InvoiceDate"].dt.date
        trans_day = df.groupby("InvoiceDate_only").size()

        fig = plt.figure(figsize=(12,4))
        trans_day.plot()
        plt.title("Jumlah Transaksi per Hari")
        plt.xlabel("Tanggal")
        plt.ylabel("Jumlah")
        st.pyplot(fig)

    # --------------------- Top Produk -----------------------------------
    with tab4:
        st.subheader("Top 10 Produk berdasarkan Quantity")
        top_prod = df.groupby("Description")["Quantity"].sum().sort_values(ascending=False).head(10)
        st.write(top_prod)

    # --------------------- Transaksi per Jam -----------------------------
    with tab5:
        st.subheader("Distribusi Transaksi per Jam")
        df["Hour"] = df["InvoiceDate"].dt.hour

        fig = plt.figure(figsize=(7,4))
        sns.countplot(x="Hour", data=df)
        plt.title("Distribusi Transaksi per Jam")
        st.pyplot(fig)

    # --------------------- Transaksi per Bulan ---------------------------
    with tab6:
        st.subheader("Distribusi Transaksi per Bulan")
        df["Month"] = df["InvoiceDate"].dt.month

        fig = plt.figure(figsize=(7,4))
        sns.countplot(x="Month", data=df)
        plt.title("Distribusi Transaksi per Bulan")
        st.pyplot(fig)

    # --------------------- PCA PREVIEW ----------------------------------
    with tab7:
        st.subheader("PCA Preview (Random Sample)")
        st.write("PCA setelah scaling, sebelum clustering")

        sample = df.sample(2000, random_state=42)

        # scale
        scaler = StandardScaler()
        scaled = scaler.fit_transform(sample[["Quantity", "Price"]])

        # PCA
        pca = PCA(n_components=2)
        comps = pca.fit_transform(scaled)

        fig = plt.figure(figsize=(7,5))
        sns.scatterplot(x=comps[:,0], y=comps[:,1], alpha=0.5)
        plt.title("PCA Scatter")
        st.pyplot(fig)


# ======================================================================
# PAGE 3 — RFM ANALYSIS
# ======================================================================
elif menu == "RFM Analysis":

    st.header("📊 RFM Analysis")

    latest_date = df["InvoiceDate"].max()
    df["Recency"] = (latest_date - df["InvoiceDate"]).dt.days

    rfm = df.groupby("Customer ID").agg({
        "Recency": "min",
        "InvoiceDate": "nunique",
        "Price": "sum"
    }).reset_index()

    rfm.columns = ["Customer ID", "Recency", "Frequency", "Monetary"]
    st.session_state["rfm"] = rfm

    st.dataframe(rfm.head())

    # Subplots RFM
    st.subheader("Distribusi Recency, Frequency, Monetary")

    fig, ax = plt.subplots(1,3, figsize=(15,4))

    ax[0].hist(rfm["Recency"], bins=20, color="skyblue")
    ax[0].set_title("Distribusi Recency")

    ax[1].hist(rfm["Frequency"], bins=20, color="orange")
    ax[1].set_title("Distribusi Frequency")

    ax[2].hist(rfm["Monetary"], bins=20, color="green")
    ax[2].set_title("Distribusi Monetary")

    st.pyplot(fig)


# ======================================================================
# PAGE 4 — KMEANS CLUSTERING
# ======================================================================
elif menu == "Clustering":

    st.header("🎯 Customer Clustering (KMeans)")

    if "rfm" not in st.session_state:
        st.warning("Jalankan RFM Analysis dulu.")
        st.stop()

    rfm = st.session_state["rfm"].copy()

    scaler = StandardScaler()
    scaled = scaler.fit_transform(rfm[["Recency", "Frequency", "Monetary"]])

    # PCA
    pca = PCA(2)
    comps = pca.fit_transform(scaled)
    rfm["PCA1"], rfm["PCA2"] = comps[:,0], comps[:,1]

    # Slider jumlah cluster
    k = st.slider("Jumlah cluster", 2, 10, 3)

    model = KMeans(n_clusters=k, random_state=42)
    rfm["Cluster"] = model.fit_predict(scaled)

    # PCA plot
    st.subheader("Cluster Visualization (PCA)")
    fig = plt.figure(figsize=(8,6))
    sns.scatterplot(data=rfm, x="PCA1", y="PCA2", hue="Cluster", palette="viridis")
    plt.title("Cluster Visualization after PCA")
    st.pyplot(fig)

    st.session_state["clustered"] = rfm


# ======================================================================
# PAGE 5 — BUSINESS INSIGHTS
# ======================================================================
elif menu == "Business Insights":

    st.header("💡 Business Insights dari Clustering")

    if "clustered" not in st.session_state:
        st.warning("Jalankan Clustering dulu.")
        st.stop()

    rfm = st.session_state["clustered"]

    # Heatmap distribusi segmen
    st.subheader("Distribusi Segmen Pelanggan per Cluster")

    def categorize_segment(row):
        if row["Recency"] < rfm["Recency"].median() and row["Frequency"] > rfm["Frequency"].median():
            return "Champions"
        elif row["Frequency"] > rfm["Frequency"].median():
            return "Loyalists"
        elif row["Monetary"] > rfm["Monetary"].median():
            return "Big Spenders, low frequency"
        elif row["Recency"] > rfm["Recency"].median():
            return "At-risk"
        else:
            return "New but promising"

    rfm["Segment"] = rfm.apply(categorize_segment, axis=1)

    pivot = pd.crosstab(rfm["Cluster"], rfm["Segment"])

    fig = plt.figure(figsize=(10,6))
    sns.heatmap(pivot, annot=True, cmap="YlGnBu", fmt="g")
    plt.title("Distribusi Segmen Pelanggan per Cluster")
    st.pyplot(fig)

    st.subheader("Interpretasi Segmen & Rekomendasi")
    st.write("""
    - **Champions** → Tingkatkan loyalitas, program khusus, VIP benefit  
    - **Loyalists** → Dorong upselling, cross-selling  
    - **Big Spenders** → Tawarkan paket premium  
    - **At-risk** → Reminder, voucher, reactivation campaign  
    - **New but promising** → Edukasi produk, penawaran awal  
    """)

