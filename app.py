
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="Customer Segmentation", page_icon="📊", layout="wide")

st.title("📊 Customer Segmentation Dashboard (RFM + KMeans)")
st.write("Aplikasi ini dibuat dari Google Colab → GitHub → Streamlit Cloud")


menu = st.sidebar.radio("Menu", ["Upload Data", "RFM Analysis", "Clustering", "Business Insights"])


if menu == "Upload Data":
    st.header("📂 Upload Data CSV")
    file = st.file_uploader("Upload file CSV", type=["csv"])

    if file:
        df = pd.read_csv(file)
        st.session_state["df"] = df
        st.success("Data berhasil di-upload!")
        st.dataframe(df.head())


elif menu == "RFM Analysis":
    if "df" not in st.session_state:
        st.warning("Upload data dulu.")
        st.stop()

    df = st.session_state["df"].copy()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df = df[(df["Quantity"] > 0) & (df["Price"] > 0)]
    df = df.dropna(subset=["Customer ID"])
    df["Customer ID"] = df["Customer ID"].astype(str)

    latest = df["InvoiceDate"].max()
    df["Recency"] = (latest - df["InvoiceDate"]).dt.days

    rfm = df.groupby("Customer ID").agg({
        "Recency": "min",
        "InvoiceDate": "nunique",
        "Price": "sum"
    }).reset_index()

    rfm.columns = ["Customer ID", "Recency", "Frequency", "Monetary"]

    st.session_state["rfm"] = rfm

    st.header("📊 Hasil RFM")
    st.dataframe(rfm.head())


elif menu == "Clustering":
    if "rfm" not in st.session_state:
        st.warning("Lakukan analisis RFM dulu.")
        st.stop()

    rfm = st.session_state["rfm"].copy()

    scaler = StandardScaler()
    scaled = scaler.fit_transform(rfm[["Recency", "Frequency", "Monetary"]])

    st.header("🔍 Elbow Method")
    wcss = []
    for i in range(2, 10):
        km = KMeans(n_clusters=i, random_state=42)
        km.fit(scaled)
        wcss.append(km.inertia_)

    fig = plt.figure(figsize=(5,4))
    plt.plot(range(2,10), wcss, marker="o")
    st.pyplot(fig)

    k = st.slider("Jumlah cluster:", 2, 10, 3)

    model = KMeans(n_clusters=k, random_state=42)
    rfm["Cluster"] = model.fit_predict(scaled)

    st.session_state["clustered"] = rfm

    st.header("🎨 PCA Visualization")
    pca = PCA(2)
    comp = pca.fit_transform(scaled)
    rfm["PCA1"], rfm["PCA2"] = comp[:,0], comp[:,1]

    fig2 = plt.figure(figsize=(7,5))
    sns.scatterplot(data=rfm, x="PCA1", y="PCA2", hue="Cluster", palette="viridis")
    st.pyplot(fig2)


elif menu == "Business Insights":
    if "clustered" not in st.session_state:
        st.warning("Hitung clustering dulu.")
        st.stop()

    rfm = st.session_state["clustered"]
    st.header("💡 Rekomendasi Bisnis")

    summary = rfm.groupby("Cluster").agg({
        "Recency": "mean",
        "Frequency": "mean",
        "Monetary": "mean",
        "Customer ID": "count"
    })

    st.write("Ringkasan per cluster:")
    st.dataframe(summary)

    st.write("Rekomendasi otomatis:")

    for c in summary.index:
        st.subheader(f"Cluster {c}")

        r = summary.loc[c,"Recency"]
        f = summary.loc[c,"Frequency"]
        m = summary.loc[c,"Monetary"]

        if f > summary["Frequency"].mean() and m > summary["Monetary"].mean():
            st.write("⭐ **Champions** → Beri loyalty program & promo eksklusif")
        elif f > summary["Frequency"].mean():
            st.write("🔁 **Loyal Customers** → Up-selling & cross-selling")
        elif m > summary["Monetary"].mean():
            st.write("💰 **Big Spenders** → Paket premium & membership")
        else:
            st.write("🧊 **Low Value Customers** → Flash sale, voucher diskon")
