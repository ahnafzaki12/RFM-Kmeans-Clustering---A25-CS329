# Customer Segmentation Analysis using RFM & K-Means Clustering

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-KMeans-green)
![Status](https://img.shields.io/badge/Status-Completed-success)

## Deskripsi Singkat Proyek

Proyek ini bertujuan untuk membantu perusahaan ritel online mengoptimalkan strategi pemasaran mereka dengan beralih dari pendekatan *mass marketing* ke **pemasaran terpersonalisasi**. 

Dengan memanfaatkan data transaksi historis, kami menerapkan analisis **RFM (Recency, Frequency, Monetary)** untuk mengevaluasi nilai pelanggan, kemudian menggunakan algoritma **K-Means Clustering** untuk mengelompokkan pelanggan ke dalam segmen-segmen perilaku yang berbeda (seperti *Champions*, *Loyalists*, *At-Risk*, dll).

**Fitur Utama:**
* **Data Cleaning & Preprocessing:** Penanganan *missing values*, duplikat, dan *outlier* (menggunakan Log Transformation/Capping).
* **RFM Modeling:** Perhitungan skor Recency, Frequency, dan Monetary untuk setiap pelanggan.
* **Clustering:** Penentuan jumlah cluster optimal dengan *Elbow Method* dan evaluasi menggunakan *Silhouette Score*.
* **Dimensionality Reduction:** Visualisasi persebaran cluster menggunakan **PCA (Principal Component Analysis)**.
* **Business Insight:** Analisis karakteristik tiap segmen untuk rekomendasi strategi bisnis.

---

## Petunjuk Setup Environment

Ikuti langkah-langkah berikut untuk menyiapkan lingkungan kerja lokal Anda agar dapat menjalankan proyek ini.

### Prasyarat
Pastikan Anda telah menginstal **Python 3.8+** dan **pip**.

### Langkah Instalasi

1.  **Clone Repositori ini**
    Buka terminal atau command prompt, lalu jalankan:
    ```bash
    git clone [https://github.com/username-anda/nama-repo-anda.git](https://github.com/username-anda/nama-repo-anda.git)
    cd nama-repo-anda
    ```

2.  **Buat Virtual Environment (Disarankan)**
    Agar *library* tidak bentrok dengan sistem utama:
    ```bash
    # Untuk Windows
    python -m venv venv
    venv\Scripts\activate

    # Untuk macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Instal Library yang Dibutuhkan**
    Jalankan perintah berikut untuk menginstal semua dependensi (pandas, numpy, matplotlib, seaborn, scikit-learn):
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn jupyter
    ```
    *(Atau jika ada file requirements.txt)*: `pip install -r requirements.txt`

---

## Cara Menjalankan Aplikasi

Aplikasi ini berbentuk **Jupyter Notebook**. Berikut cara menjalankannya:

1.  Pastikan virtual environment sudah aktif.
2.  Jalankan Jupyter Notebook dengan perintah:
    ```bash
    jupyter notebook
    ```
3.  Browser akan terbuka secara otomatis. Cari dan buka file notebook proyek, misalnya: `Customer_Segmentation_RFM.ipynb`.
4.  Jalankan setiap sel (*cell*) kode secara berurutan dari atas ke bawah dengan menekan tombol **Run** atau `Shift + Enter`.
5.  Hasil analisis, grafik Elbow, visualisasi PCA, dan Heatmap segmen akan muncul langsung di dalam notebook.

---

## Tautan Model Machine Learning

Jika Anda ingin menggunakan model K-Means yang sudah dilatih tanpa menjalankan ulang proses training, Anda dapat mengunduhnya di bawah ini:

* **Download Model (Pickle):** https://drive.google.com/file/d/1uohp_MVHmlh2vpOtCbIVJLKDD1IUxiw-/view?usp=sharing
### Cara Memuat (Load) Model di Python:

```python
import joblib
import pandas as pd

# 1. Load Model dan Scaler
kmeans_model = joblib.load('kmeans_model.pkl')
scaler = joblib.load('scaler.pkl')

# 2. Siapkan Data Baru (Contoh)
# Pastikan urutan kolom sesuai: Recency, Frequency, Monetary
new_data = pd.DataFrame({
    'Recency': [10, 300],
    'Frequency': [50, 1],
    'Monetary': [10000, 50]
})

# 3. Lakukan Scaling (Transformasi Log jika diperlukan sebelumnya)
new_data_scaled = scaler.transform(new_data)

# 4. Prediksi Cluster
prediction = kmeans_model.predict(new_data_scaled)
print(f"Prediksi Cluster: {prediction}")
