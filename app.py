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


@st.cache_data
def load_data():
    url = "https://drive.google.com/uc?id=1V2IdrRQ8XQmJlzb2PAlJw0ziJQg-13QW"
    df = pd.read_csv(url)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df = df.dropna(subset=["Customer ID"])
    df["Customer ID"] = df["Customer ID"].astype(str)
    df = df[(df["Quantity"] > 0) & (df["Price"] > 0)]
    return df

df = load_data()

st.success("Dataset berhasil dimuat otomatis dari Google Drive.")
st.write(f"Jumlah data: {df.shape[0]} rows")


menu = st.sidebar.radio("Menu", ["Dataset", "RFM Analysis", "Clustering", "Business Insights"])

if menu == "Dataset":
    st.header("📂 Dataset Preview")
    st.dataframe(df.head(20))

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

    st.write("Ringkasan RFM:")
    st.dataframe(rfm.head())



elif menu == "Clustering":

    if "rfm" not in st.session_state:
        st.warning("Silakan jalankan RFM Analysis dulu.")
        st.stop()

    rfm = st.session_state["rfm"].copy()

    scaler = StandardScaler()
    scaled = scaler.fit_transform(rfm[["Recency", "Frequency", "Monetary"]])

    st.subheader("🔍 Elbow Method")

    wcss = []
    for i in range(2, 10):
        km = KMeans(n_clusters=i, random_state=42)
        km.fit(scaled)
        wcss.append(km.inertia_)

    fig = plt.figure(figsize=(5,4))
    plt.plot(range(2,10), wcss, marker='o')
    st.pyplot(fig)

    k = st.slider("Jumlah cluster:", 2, 10, 3)

    kmeans = KMeans(n_clusters=k, random_state=42)
    rfm["Cluster"] = kmeans.fit_predict(scaled)

    st.session_state["clustered"] = rfm

   
    st.subheader("🎨 PCA Visualization")
    pca = PCA(2)
    comp = pca.fit_transform(scaled)
    rfm["PCA1"], rfm["PCA2"] = comp[:,0], comp[:,1]

    fig2 = plt.figure(figsize=(7,5))
    sns.scatterplot(data=rfm, x="PCA1", y="PCA2", hue="Cluster", palette="viridis")
    st.pyplot(fig2)


elif menu == "Business Insights":

    if "clustered" not in st.session_state:
        st.warning("Silakan jalankan Clustering dulu.")
        st.stop()

    rfm = st.session_state["clustered"]

    st.header("💡 Business Insights")
    summary = rfm.groupby("Cluster").agg({
        "Recency": "mean",
        "Frequency": "mean",
        "Monetary": "mean",
        "Customer ID": "count"
    })

    st.write("Ringkasan per cluster:")
    st.dataframe(summary)

    st.subheader("📌 Rekomendasi")

    for c in summary.index:
        st.markdown(f"### Cluster {c}")

        r = summary.loc[c, "Recency"]
        f = summary.loc[c, "Frequency"]
        m = summary.loc[c, "Monetary"]

        if f > summary["Frequency"].mean() and m > summary["Monetary"].mean():
            st.write("⭐ **Champions** → Reward program, promo eksklusif.")
        elif f > summary["Frequency"].mean():
            st.write("🔁 **Loyal Customers** → Up-selling & cross-selling.")
        elif m > summary["Monetary"].mean():
            st.write("💰 **Big Spenders** → Paket premium, membership.")
        else:
            st.write("🧊 **Low Value Customers** → Flash sale, voucher diskon.")
