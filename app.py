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

# LOAD

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

# SIDEBAR

menu = st.sidebar.radio(
    "Navigation",
    ["Dataset Overview", "EDA Visualizations", "RFM Analysis", "Clustering", "Business Insights"]
)

# PAGE 1 — DATASET OVERVIEW

if menu == "Dataset Overview":
    st.header("Dataset Preview")
    st.dataframe(df.head())
    st.subheader("Dataset Preview")
    st.markdown("""
    Dataset Online Retail II berisi seluruh transaksi dari sebuah perusahaan ritel daring yang berbasis di Inggris dan beroperasi tanpa toko fisik. Data mencakup periode 1 Desember 2009 hingga 9 Desember 2011, dengan fokus pada penjualan produk giftware (hadiah dan dekorasi) yang unik. Sebagian besar pelanggan merupakan wholesaler yang membeli dalam jumlah besar untuk kebutuhan bisnis.
    Struktur Atribut
    Dataset memiliki delapan atribut utama:
    - InvoiceNo - Nomor faktur unik untuk setiap transaksi.
    - StockCode — Kode unik untuk setiap produk.
    - Description — Nama atau deskripsi produk.
    - Quantity — Jumlah unit produk yang dibeli dalam satu transaksi.
    - InvoiceDate — Tanggal dan waktu transaksi terjadi.
    - UnitPrice — Harga satuan produk dalam Pound Sterling (£).
    - CustomerID — ID pelanggan unik (hanya muncul untuk pelanggan teridentifikasi)
    - Country — Negara tempat pelanggan berada.
    Setiap cluster memerlukan strategi berbeda untuk retensi, upselling, dan nurturing pelanggan.
    """)

# PAGE 2 — EDA VISUALIZATIONS
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
        ### Intrepretasi Distribusi Quantity
          Distribusi Quantity pada diagram tampak right-skewed, dengan sebagian besar transaksi berada pada jumlah kecil (terutama 1–3 unit), sementara beberapa puncak jelas muncul pada nilai seperti 6, 12, dan 24 yang menunjukkan pola pembelian dalam paket standar. Nilai Quantity besar relatif jarang, sehingga distribusinya tetap wajar dan mencerminkan karakteristik umum transaksi retail, mayoritas pelanggan membeli sedikit, dan sebagian produk dijual dalam kelipatan tertentu.
        """)

    # 2. Distribusi Price cleaned
    with tab2:
        st.subheader("Distribusi Price (1%–99%)")
        fig = plt.figure(figsize=(8,4))
        sns.histplot(df_clean_price["Price"], kde=True, bins=50)
        st.pyplot(fig)

        st.markdown("""
        ### Intrepretasi Distribusi Price
          Distribusi Price setelah pembersihan menunjukkan pola right-skewed, di mana sebagian besar harga terkonsentrasi pada rentang sekitar 0.5–4 dengan puncak tertinggi di kisaran 1–2. Setelah itu frekuensi menurun tajam, hanya terlihat beberapa puncak kecil di sekitar harga 4–8, sedangkan harga di atas kisaran tersebut (mendekati 10–15) muncul sangat jarang. Pola ini menunjukkan bahwa mayoritas produk yang terjual berharga murah hingga menengah, sementara produk dengan harga tinggi hanya menyumbang sebagian kecil dari transaksi.
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
        ### Intrepretasi Transaksi per Hari
          Jumlah transaksi per hari menunjukkan pola yang sangat fluktuatif dengan kecenderungan meningkat dari awal 2010 hingga akhir 2011. Pada awal periode, transaksi harian berada di kisaran 500–1500, kemudian mengalami kenaikan signifikan pada akhir 2010 dengan puncak mendekati 3000 transaksi per hari. Setelah sedikit penurunan di awal 2011, jumlah transaksi kembali meningkat menuju akhir tahun dan mencapai level tertinggi menjelang Desember 2011. Pola ini mencerminkan dinamika musim belanja dan pertumbuhan aktivitas penjualan dari waktu ke waktu.
        """)

    # 4. Top Produk
    with tab4:
        st.subheader("Top 10 Produk")
        top10 = df.groupby("Description")["Quantity"].sum().sort_values(ascending=False).head(10)
        st.write(top10)

        st.markdown("""
        ### Intrepretasi Top 10 Produk
          Produk dengan jumlah penjualan tertinggi didominasi oleh barang-barang dekoratif dan kebutuhan rumah tangga kecil, dengan WHITE HANGING HEART T-LIGHT HOLDER menjadi produk paling laris dengan 27.542 unit terjual. Disusul oleh ASSORTED COLOUR BIRD ORNAMENT serta berbagai jenis cake cases yang secara konsisten muncul dalam daftar teratas, menandakan tingginya permintaan untuk perlengkapan baking dan barang dekoratif murah. Produk seperti JUMBO BAG RED RETROSPOT, STRAWBERRY CERAMIC TRINKET BOX, hingga REGENCY CAKESTAND 3 TIER juga menunjukkan volume penjualan besar, mencerminkan bahwa pelanggan cenderung membeli barang-barang hadiah, dekorasi, dan perlengkapan rumah dengan harga terjangkau dalam jumlah tinggi.
        """)

    # 5. Transaksi per Jam
    with tab5:
        st.subheader("Transaksi per Jam")
        df["Hour"] = df["InvoiceDate"].dt.hour
        fig = plt.figure(figsize=(7,4))
        sns.countplot(x="Hour", data=df)
        st.pyplot(fig)

        st.markdown("""
        ### Intrepretasi Transaksi per Jam
          Distribusi transaksi per jam menunjukkan bahwa aktivitas penjualan mencapai puncaknya pada jam kerja tengah hari, khususnya antara pukul 11.00 hingga 14.00, dengan lonjakan tertinggi sekitar pukul 12.00. Sebaliknya, jumlah transaksi sangat rendah pada pagi awal dan sore menjelang malam. Pola ini mengindikasikan bahwa pelanggan cenderung melakukan pemesanan pada periode sibuk kerja atau saat istirahat siang, sehingga transaksi menumpuk di sekitar jam-jam tersebut.
        """)

    # 6. Transaksi per Bulan
    with tab6:
        st.subheader("Transaksi per Bulan")
        df["Month"] = df["InvoiceDate"].dt.month
        fig = plt.figure(figsize=(7,4))
        sns.countplot(x="Month", data=df)
        st.pyplot(fig)

        st.markdown("""
        ### Intrepretasi Transaksi per Bulan
          Distribusi transaksi per bulan memperlihatkan pola musiman yang kuat, di mana transaksi relatif stabil dari bulan Januari hingga Agustus, kemudian mulai meningkat signifikan pada bulan September hingga November, dengan puncak tertinggi pada November. Kenaikan ini mencerminkan periode belanja musiman seperti persiapan liburan dan akhir tahun, yang biasanya mendorong aktivitas pembelian lebih tinggi. Setelah puncak tersebut, transaksi kembali menurun pada Desember, meski masih berada di atas rata-rata bulanan.
        """)


# PAGE 3 — RFM ANALYSIS
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
    st.subheader("Intrepretasi Distribusi RFM")
    st.markdown("""
    ### Distribusi Recency
    Sebagian besar pelanggan memiliki recency rendah hingga menengah, menunjukkan mereka masih cukup aktif. Namun terdapat kelompok dengan recency tinggi yang sudah lama tidak bertransaksi, sehingga berpotensi masuk kategori at-risk.
    ### Distribusi Frequency
    Mayoritas pelanggan hanya bertransaksi 1–5 kali, sementara hanya sedikit pelanggan dengan frekuensi tinggi. Pola ini mencerminkan dominasi pelanggan non-loyal dan sedikit pelanggan sangat loyal yang membawa nilai penting bagi bisnis.
    ### Distribusi Monetary
    Nilai belanja total pelanggan didominasi oleh kelompok bernilai rendah–menengah. Hanya sebagian kecil pelanggan yang memberikan kontribusi pendapatan besar, menunjukkan adanya perilaku Pareto (20% pelanggan menghasilkan sebagian besar revenue).

    """)

# PAGE 4 — CLUSTERING
elif menu == "Clustering":

    st.header("Clustering")

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
    plt.title("Cluster Visualization PCA")
    st.pyplot(fig)
    st.markdown("""
    ### Intrepretasi PCA
      Visualisasi cluster setelah PCA menunjukkan pemisahan yang jelas antara tiga kelompok pelanggan, di mana setiap cluster menempati area berbeda pada ruang PCA1 dan PCA2. Cluster berwarna ungu tampak merupakan kelompok terbesar, mencakup pelanggan dengan karakteristik RFM yang lebih umum. Cluster kuning dan hijau berada pada posisi berbeda dan lebih terfokus, menandakan segmentasi pelanggan berdasarkan pola perilaku yang cukup berbeda, seperti nilai belanja, frekuensi pembelian, atau tingkat aktivitas. Pemisahan yang rapi ini menunjukkan bahwa metode clustering berhasil mengidentifikasi tiga segmen pelanggan yang memiliki perbedaan nyata dalam perilaku mereka
    """)

# PAGE 5 — BUSINESS INSIGHTS

elif menu == "Business Insights":

    if "clustered" not in st.session_state:
        st.warning("Silakan jalankan Clustering terlebih dahulu.")
        st.stop()

    rfm = st.session_state["clustered"]

    st.header("Business Insights")

    # SEGMENTATION
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
    plt.title("Distribusi Segmen Pelanggan per Cluster")
    st.pyplot(fig)

    st.subheader("Interpretasi dan Strategi Bisnis")
    st.markdown("""
    ### Cluster 0

    Karakteristik utama cluster ini berisi campuran pelanggan bernilai tinggi dan pelanggan yang cukup berisiko, dengan jumlah besar di segmen At-risk dan Champions.
    - At-risk (3.086 pelanggan): Banyak pelanggan dalam kelompok ini memiliki potensi churn, sehingga perlu perhatian khusus seperti reminder pembelian, email re-engagement, atau diskon khusus.
    - Big Spenders, Low Frequency (116 pelanggan): Mereka mengeluarkan banyak uang namun tidak sering bertransaksi. Penawaran personal atau program yang meningkatkan frekuensi belanja bisa mendorong mereka lebih aktif.
    - Champions (3.127 pelanggan): Kelompok yang sangat bernilai—mereka aktif dan memiliki Monetary tinggi. Mereka cocok diberikan reward premium, early access produk baru, atau VIP program.
    - Loyalists (50 pelanggan): Pelanggan setia yang rutin membeli. Tingkatkan loyalitas mereka dengan penawaran membership atau hadiah loyalitas.
    - New but Promising (0 pelanggan): Tidak ada pelanggan baru dengan potensi tinggi pada cluster ini.

    Strategi untuk Cluster 0
    Fokus pada retensi pelanggan bernilai tinggi seperti Champions dan Loyalists dengan program eksklusif dan pengalaman yang dipersonalisasi. Sementara itu, pelanggan At-risk perlu segera diaktifkan kembali melalui win-back campaigns, reminder pembelian, atau potongan harga. Targetkan Big Spenders dengan penawaran premium untuk meningkatkan frekuensi.

    ### Cluster 1

    Karakteristik utama cluster ini didominasi oleh pelanggan At-risk, dengan hanya sebagian kecil pelanggan bernilai tinggi.
    - At-risk (8.833 pelanggan): Ini segmen terbesar dalam cluster, sehingga merupakan prioritas utama untuk kampanye reaktivasi. Berikan promosi yang kuat untuk mencegah churn lebih lanjut.
    - Big Spenders, Low Frequency (1.414 pelanggan): Pelanggan yang jarang berbelanja tetapi mengeluarkan uang cukup banyak. Strategi upsell atau rekomendasi produk personal bisa membantu.
    - Champions (119 pelanggan): Walaupun jumlahnya kecil, mereka tetap sangat berharga. Berikan pengalaman premium agar mereka tetap setia.
    - Loyalists (88 pelanggan): Pelanggan dengan engagement stabil. Tawarkan loyalty reward, poin belanja, atau penawaran eksklusif.
    - New but Promising (321 pelanggan): Pelanggan baru yang menunjukkan potensi. Perlu nurturing seperti welcome series email, penawaran pertama, atau edukasi produk.

    Strategi untuk Cluster 1
    Prioritaskan re-engagement massal, terutama untuk segmen At-risk yang sangat besar. Gunakan email marketing, promo besar, atau bundle produk untuk menarik mereka kembali. Pelanggan bernilai tinggi seperti Champions dan Big Spenders perlu dipertahankan melalui program VIP atau rekomendasi produk yang dipersonalisasi. Segmen New but Promising perlu nurturance agar berkembang menjadi pelanggan loyal.

    ### Cluster 2

    Karakteristik utama cluster ini memiliki banyak pelanggan yang baru berkembang dan sejumlah pelanggan bernilai tinggi.
    - At-risk (0 pelanggan): Tidak ada pelanggan yang berisiko pada cluster ini.
    - Big Spenders, Low Frequency (1.368 pelanggan): Pelanggan bernilai besar namun dengan frekuensi rendah. Berikan insentif untuk mendorong repeat purchase.
    - Champions (1.895 pelanggan): Pelanggan setia dan bernilai tinggi. Pertahankan momentum dengan reward eksklusif dan layanan premium.
    - Loyalists (923 pelanggan): Pelanggan aktif yang rajin bertransaksi. Tawarkan program loyalitas lanjutan untuk memperkuat hubungan.
    - New but Promising (5.281 pelanggan): Segmen terbesar dalam cluster ini. Mereka adalah pelanggan baru yang menunjukkan potensi besar dan harus di-nurture agar naik ke segmen loyal.

    Strategi untuk Cluster 2
    Fokus pada mendorong pertumbuhan pelanggan baru melalui edukasi produk, promo sambutan, dan program pengenalan brand. Loyalists dan Champions perlu dipertahankan melalui penghargaan eksklusif, poin loyalitas, dan pengalaman personal. Sementara itu, Big Spenders dapat diarahkan untuk meningkatkan frekuensi belanja melalui rekomendasi produk dan penawaran time-limited
    """)


