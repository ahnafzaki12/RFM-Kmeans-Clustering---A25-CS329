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
                   layout="wide")

st.title("Customer Segmentation Dashboard")

# =====================================================================
# LOAD + PREPROCESSING IDENTIK NOTEBOOK
# =====================================================================
@st.cache_data
def load_data():
    url = "https://drive.google.com/uc?id=1V2IdrRQ8XQmJlzb2PAlJw0ziJQg-13QW"
    df = pd.read_csv(url)

    # 1. Missing values
    df["Description"] = df["Description"].fillna("Tidak ada deskripsi")
    df = df.dropna(subset=["Customer ID"])
    df["Customer ID"] = df["Customer ID"].astype(str)

    # 2. Drop duplicates
    df = df.drop_duplicates()

    # 3. Convert date
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    # 4. Positive only
    df = df[(df["Quantity"] > 0) & (df["Price"] > 0)]

    # 5. IQR Quantity
    Q1 = df["Quantity"].quantile(0.25)
    Q3 = df["Quantity"].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df["Quantity"] >= Q1 - 1.5 * IQR) & (df["Quantity"] <= Q3 + 1.5 * IQR)]

    # 6. Price quantile 1%–99%
    p1 = df["Price"].quantile(0.01)
    p99 = df["Price"].quantile(0.99)
    df_price_clean = df[(df["Price"] >= p1) & (df["Price"] <= p99)]

    return df, df_price_clean

df, df_clean_price = load_data()

# =====================================================================
# SIDEBAR
# =====================================================================
menu = st.sidebar.radio(
    "Navigation",
    ["Dataset Overview", "EDA Visualizations", "RFM Analysis", "Clustering", "Business Insights"]
)

# =====================================================================
# PAGE 1 — DATASET OVERVIEW
# =====================================================================
if menu == "Dataset Overview":
    st.header("Dataset Preview")
    st.dataframe(df.head())
    st.markdown("""
    ### Interpretasi Notebook

    Cluster 0 → campuran pelanggan sangat bernilai dan pelanggan berisiko tinggi.  
    Cluster 1 → mayoritas pelanggan At-risk dengan aktivitas rendah.  
    Cluster 2 → pelanggan baru yang potensial serta beberapa pelanggan bernilai tinggi.

    Setiap cluster memerlukan strategi berbeda untuk retensi, upselling, dan nurturing pelanggan.
    """)

# =====================================================================
# PAGE 2 — EDA VISUALIZATIONS (MATCH NOTEBOOK)
# =====================================================================
elif menu == "EDA Visualizations":

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Distribusi Quantity",
        "Distribusi Price",
        "Jumlah Transaksi per Hari",
        "Top 10 Produk",
        "Transaksi per Jam",
        "Transaksi per Bulan"
    ])

    # 1. Distribusi Quantity
    with tab1:
        st.subheader("Distribusi Quantity")
        fig = plt.figure(figsize=(8,4))
        sns.histplot(df["Quantity"], kde=True)
        st.pyplot(fig)

        st.markdown("""
        **Deskripsi Notebook:**  
          Distribusi Quantity pada diagram tampak right-skewed, dengan sebagian besar transaksi berada pada jumlah kecil (terutama 1–3 unit), sementara beberapa puncak jelas muncul pada nilai seperti 6, 12, dan 24 yang menunjukkan pola pembelian dalam paket standar. Nilai Quantity besar relatif jarang, sehingga distribusinya tetap wajar dan mencerminkan karakteristik umum transaksi retail, mayoritas pelanggan membeli sedikit, dan sebagian produk dijual dalam kelipatan tertentu.
        """)

    # 2. Distribusi Price cleaned
    with tab2:
        st.subheader("Distribusi Price (1%–99%)")
        fig = plt.figure(figsize=(8,4))
        sns.histplot(df_clean_price["Price"], kde=True, bins=50)
        st.pyplot(fig)

        st.markdown("""
        **Deskripsi Notebook:**  
        Distribusi Price setelah pembersihan menunjukkan pola *right-skewed*,  
        mayoritas harga berada di kisaran 0.5–4, dengan puncak pada 1–2.  
        Produk mahal hanya sedikit muncul di kisaran 10–15.
        """)

    # 3. Transaksi per Hari
    with tab3:
        st.subheader("Jumlah Transaksi per Hari")
        df["InvoiceDate_only"] = df["InvoiceDate"].dt.date
        trans = df.groupby("InvoiceDate_only").size()

        fig = plt.figure(figsize=(12,4))
        plt.plot(trans)
        st.pyplot(fig)

        st.markdown("""
        **Deskripsi Notebook:**  
        Jumlah transaksi harian meningkat signifikan dari 2010 sampai akhir 2011.  
        Awal tahun berkisar 500–1500 transaksi/hari dan mencapai puncak mendekati 3000.  
        Ini mencerminkan peningkatan aktivitas penjualan terutama menjelang musim liburan.
        """)

    # 4. Top Produk
    with tab4:
        st.subheader("Top 10 Produk")
        top10 = df.groupby("Description")["Quantity"].sum().sort_values(ascending=False).head(10)
        st.write(top10)

        st.markdown("""
        **Deskripsi Notebook:**  
        Produk terlaris didominasi barang dekoratif dan perlengkapan rumah kecil.  
        *WHITE HANGING HEART T-LIGHT HOLDER* adalah yang paling laris dengan 27.542 unit.  
        Banyaknya cake cases dan dekorasi menandakan tingginya pembelian barang murah-menengah.
        """)

    # 5. Transaksi per Jam
    with tab5:
        st.subheader("Transaksi per Jam")
        df["Hour"] = df["InvoiceDate"].dt.hour
        fig = plt.figure(figsize=(7,4))
        sns.countplot(x="Hour", data=df)
        st.pyplot(fig)

        st.markdown("""
        **Deskripsi Notebook:**  
        Aktivitas transaksi memuncak pada pukul 11.00–14.00, terutama pukul 12.00.  
        Transaksi rendah pada pagi awal dan sore, menandakan pelanggan memesan saat jam istirahat kerja.
        """)

    # 6. Transaksi per Bulan
    with tab6:
        st.subheader("Transaksi per Bulan")
        df["Month"] = df["InvoiceDate"].dt.month
        fig = plt.figure(figsize=(7,4))
        sns.countplot(x="Month", data=df)
        st.pyplot(fig)

        st.markdown("""
        **Deskripsi Notebook:**  
        Transaksi stabil Januari–Agustus, lalu meningkat tajam September–November  
        dengan puncak pada November karena musim liburan. Desember menurun tetapi masih tinggi.
        """)


# =====================================================================
# PAGE 3 — RFM ANALYSIS (MATCH NOTEBOOK)
# =====================================================================
elif menu == "RFM Analysis":

    st.header("RFM Analysis")

    # Hitung Recency
    latest = df["InvoiceDate"].max()
    df["Recency"] = (latest - df["InvoiceDate"]).dt.days

    # Frequency & Monetary
    frequency_df = df.groupby("Customer ID")["InvoiceDate"].nunique().reset_index()
    frequency_df.columns = ["Customer ID", "Frequency"]

    monetary_df = df.groupby("Customer ID")["Price"].sum().reset_index()
    monetary_df.columns = ["Customer ID", "Monetary"]

    rfm = pd.merge(frequency_df, monetary_df, on="Customer ID")
    rfm = pd.merge(rfm, df[["Customer ID", "Recency"]].drop_duplicates(), on="Customer ID")

    # Outlier removal (IQR)
    def rm_iqr(data, col):
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        return data[(data[col] >= Q1 - 1.5*IQR) & (data[col] <= Q3 + 1.5*IQR)]

    rfm = rm_iqr(rfm, "Recency")
    rfm = rm_iqr(rfm, "Frequency")
    rfm = rm_iqr(rfm, "Monetary")

    st.session_state["rfm"] = rfm

    # Plot distribusi
    fig, ax = plt.subplots(1,3, figsize=(16,4))
    ax[0].hist(rfm["Recency"], bins=20, color="skyblue")
    ax[0].set_title("Distribusi Recency")

    ax[1].hist(rfm["Frequency"], bins=20, color="orange")
    ax[1].set_title("Distribusi Frequency")

    ax[2].hist(rfm["Monetary"], bins=20, color="green")
    ax[2].set_title("Distribusi Monetary")

    st.pyplot(fig)
    st.markdown("""
    ### Deskripsi Notebook

    Distribusi Recency menyebar cukup luas, menandakan banyak pelanggan masih aktif.  
    Frequency sangat *right-skewed*: mayoritas pelanggan hanya bertransaksi sedikit.  
    Monetary juga *right-skewed*: sebagian besar pelanggan memiliki nilai belanja rendah–menengah.

    Ini sesuai karakter umum RFM:  
    banyak pelanggan biasa, sedikit pelanggan sangat bernilai tinggi.
    """)

# =====================================================================
# PAGE 4 — CLUSTERING (MATCH NOTEBOOK)
# =====================================================================
elif menu == "Clustering":

    st.header("🎯 Clustering — MATCH NOTEBOOK")

    if "rfm" not in st.session_state:
        st.warning("Silakan jalankan RFM Analysis terlebih dahulu.")
        st.stop()

    rfm = st.session_state["rfm"].copy()

    # Scaling
    scaler = StandardScaler()
    scaled = scaler.fit_transform(rfm[["Recency","Frequency","Monetary"]])

    # PCA
    pca = PCA(2)
    comp = pca.fit_transform(scaled)
    rfm["PCA1"] = comp[:,0]
    rfm["PCA2"] = comp[:,1]

    # KMeans (Notebook: k = 3)
    kmeans = KMeans(n_clusters=3, init="k-means++", n_init=10, max_iter=300, random_state=42)
    rfm["Cluster"] = kmeans.fit_predict(scaled)

    st.session_state["clustered"] = rfm

    # Plot PCA
    fig = plt.figure(figsize=(8,6))
    sns.scatterplot(x=rfm["PCA1"], y=rfm["PCA2"], hue=rfm["Cluster"], palette="viridis")
    plt.title("Cluster Visualization after PCA — MATCH NOTEBOOK")
    st.pyplot(fig)
    st.markdown("""
    ### Deskripsi Notebook

    Visualisasi PCA menunjukkan pemisahan yang jelas antar cluster.  
    Cluster ungu adalah kelompok terbesar dengan karakteristik umum.  
    Cluster kuning dan hijau lebih terfokus, mencerminkan perbedaan kuat  
    dalam pola belanja dan aktivitas pelanggan.

    Ini membuktikan bahwa model KMeans berhasil melakukan segmentasi pelanggan dengan baik.
    """)


# =====================================================================
# PAGE 5 — BUSINESS INSIGHTS (MATCH NOTEBOOK)
# =====================================================================
elif menu == "Business Insights":

    if "clustered" not in st.session_state:
        st.warning("Silakan jalankan Clustering terlebih dahulu.")
        st.stop()

    rfm = st.session_state["clustered"]

    st.header("💡 Business Insights — MATCH NOTEBOOK")

    # ===== SEGMENTATION (exact Notebook logic) =====
    rec_med = rfm["Recency"].median()
    freq_med = rfm["Frequency"].median()
    mon_med = rfm["Monetary"].median()

    def categorize_segment(row):
        if row['Frequency'] > freq_med and row['Monetary'] > mon_med and row['Recency'] <= rec_med:
            return 'Champions'
        elif row['Frequency'] > freq_med and row['Recency'] <= rec_med:
            return 'Loyalists'
        elif row['Frequency'] <= freq_med and row['Monetary'] > mon_med:
            return 'Big spenders, low frequency'
        elif row['Recency'] <= rec_med and row['Frequency'] <= freq_med and row['Monetary'] <= mon_med:
            return 'New but promising'
        elif row['Recency'] > rec_med:
            return 'At-risk'
        else:
            return 'Low-value, low-engagement'

    rfm["Segment"] = rfm.apply(categorize_segment, axis=1)

    # Heatmap
    segment_counts = rfm.pivot_table(index="Cluster", columns="Segment", aggfunc="size", fill_value=0)

    fig = plt.figure(figsize=(12,6))
    sns.heatmap(segment_counts, annot=True, cmap="YlGnBu", fmt="g", linewidths=1, linecolor="black")
    plt.title("Distribusi Segmen Pelanggan per Cluster — MATCH NOTEBOOK")
    st.pyplot(fig)

    st.subheader("Interpretasi dan Strategi Bisnis")
    st.write("""
    • Champions → Beri reward eksklusif
    • Loyalists → Dorong upselling
    • Big Spenders → Tingkatkan frekuensi
    • New but promising → Nurturing program
    • At-risk → Win-back campaign
    """)

    st.markdown("""
    ### Interpretasi Notebook

    Cluster 0 → campuran pelanggan sangat bernilai dan pelanggan berisiko tinggi.  
    Cluster 1 → mayoritas pelanggan At-risk dengan aktivitas rendah.  
    Cluster 2 → pelanggan baru yang potensial serta beberapa pelanggan bernilai tinggi.

    Setiap cluster memerlukan strategi berbeda untuk retensi, upselling, dan nurturing pelanggan.
    """)
 

