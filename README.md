# Laporan Proyek Machine Learning - Raka Satria Efendi

## Domain Proyek
Penyakit yang tidak terdeteksi secara dini dapat memperburuk kondisi kesehatan pasien, terutama di daerah dengan akses terbatas ke tenaga medis. Menurut WHO (2023), keterlambatan diagnosis menyebabkan peningkatan angka kematian preventable hingga 35% di negara berkembang. Oleh karena itu, diperlukan sistem cerdas yang mampu memprediksi penyakit berdasarkan gejala dalam bentuk teks untuk mendukung diagnosis awal.

Proyek ini mengembangkan sistem klasifikasi penyakit berbasis Natural Language Processing (NLP) yang menganalisis deskripsi gejala untuk memprediksi penyakit secara akurat. Sistem ini memanfaatkan algoritma machine learning seperti Support Vector Machine (SVM), Logistic Regression, Naive Bayes, dan XGBoost untuk mengenali pola gejala dan menghubungkannya dengan label penyakit. Sistem ini berpotensi dikembangkan menjadi chatbot medis berbasis AI untuk mendukung self-diagnosis awal dan penyuluhan kesehatan.

**Referensi Pendukung:**
- WHO (2023) menyoroti pentingnya diagnosis dini.
- Jurnal seperti Dessi et al. (2021) dan Kalra et al. (2019) mendukung penggunaan TF-IDF dan SVM untuk klasifikasi teks medis.

## Business Understanding

### Problem Statements
1. Bagaimana mengembangkan sistem klasifikasi otomatis untuk memprediksi penyakit berdasarkan input teks gejala?
2. Algoritma machine learning atau NLP mana yang paling efektif untuk klasifikasi penyakit berbasis gejala tertulis?
3. Seberapa akurat sistem prediksi ini dalam membantu diagnosis awal berdasarkan gejala?

### Goals
1. Membangun sistem klasifikasi penyakit berbasis gejala teks.
2. Membandingkan performa model NLP dan memilih model terbaik berdasarkan metrik evaluasi.
3. Menyediakan solusi otomatis untuk membantu masyarakat dan tenaga kesehatan dalam diagnosis dini.

### Solution Statements
1. Menggunakan TF-IDF untuk merepresentasikan teks gejala dan menganalisis kontribusi kata penting terhadap klasifikasi.
2. Menerapkan dan membandingkan algoritma Logistic Regression, Naive Bayes, SVM, dan XGBoost, dengan hyperparameter tuning menggunakan Optuna untuk model terbaik.
3. Menganalisis hubungan semantik antar gejala untuk meningkatkan akurasi klasifikasi.

### Metodologi
Proyek ini bertujuan memprediksi penyakit berdasarkan deskripsi gejala menggunakan model klasifikasi NLP, dengan label penyakit sebagai target utama.

### Metrik
Metrik evaluasi meliputi:
- **Confusion Matrix**: Menampilkan True Positive (TP), False Positive (FP), True Negative (TN), dan False Negative (FN).
- **Akurasi**: Persentase prediksi benar dari total data.
- **Precision**: Ketepatan prediksi kelas positif.
- **Recall**: Kemampuan menangkap data positif.
- **F1-Score**: Rata-rata harmonik antara precision dan recall.

Metrik ini penting untuk mengevaluasi performa model, terutama pada dataset dengan distribusi kelas tidak seimbang.

## Data Understanding
Dataset diambil dari Kaggle, dipublikasikan oleh niyarrbarman ([link dataset](https://www.kaggle.com/datasets/niyarrbarman/symptom2disease)). Dataset ini berisi 1200 entri dengan dua kolom utama:
- **label**: Nama penyakit (24 kelas, seperti Diabetes, Psoriasis, dll.).
- **text**: Deskripsi gejala dalam teks.

### Exploratory Data Analysis (EDA)
- **Distribusi Kelas**: [gambar distribusi label penyakit] menunjukkan distribusi kelas yang cukup seimbang.
- **Analisis Gejala**: [gambar word cloud gejala] menunjukkan kata-kata kunci seperti "pain", "fever", dan "itching" yang sering muncul.
- **Jumlah Kelas**: 24 penyakit unik diidentifikasi, dengan beberapa penyakit memiliki gejala tumpang tindih (misalnya, GERD dan Peptic Ulcer).

### Data Quality Verification
- **Duplikat**: Tidak ada data duplikat ditemukan.
- **Missing Value**: Tidak ada nilai hilang pada kolom teks dan label.
- **Outlier**: Tidak ada outlier signifikan pada panjang teks atau jumlah kata.

## Data Preparation
### Data Cleaning
- **Text Cleaning**: Menghapus tanda baca, angka, dan kata-kata tidak relevan (stopwords) menggunakan NLTK.
- **Stemming**: Menggunakan PorterStemmer untuk mengubah kata ke bentuk dasar (misalnya, "running" menjadi "run").
- **TF-IDF Vectorization**: Mengubah teks gejala menjadi vektor numerik untuk diproses model.
- **Label Encoding**: Mengubah label penyakit menjadi format numerik.

### Data Splitting
- Data dibagi menjadi 80% data latih dan 20% data uji menggunakan `train_test_split`.

### Normalisasi
- Fitur numerik (panjang teks dan jumlah kata) dinormalisasi menggunakan `MinMaxScaler` untuk menyeragamkan skala.

[gambar pipeline data preparation]

## Modeling
### 1. Algoritma Support Vector Machine (SVM)
- **Kelebihan**: Efektif untuk data berdimensi tinggi seperti TF-IDF, robust terhadap noise.
- **Kekurangan**: Sensitif terhadap pemilihan kernel dan risiko overfitting pada data kompleks.
- **Parameter**: Menggunakan kernel linear dengan tuning Optuna (C=1.0).
- **Akurasi**: 98.5% (terbaik setelah tuning, dipilih untuk menghindari overfitting dibandingkan skor 100%).

### 2. Algoritma Logistic Regression
- **Kelebihan**: Sederhana, interpretable, cocok untuk data linear.
- **Kekurangan**: Kurang efektif pada hubungan non-linear.
- **Parameter**: Tuning dengan Optuna untuk parameter C.
- **Akurasi**: 98.1%.

### 3. Algoritma Multinomial Naive Bayes
- **Kelebihan**: Cepat, cocok untuk data teks.
- **Kekurangan**: Mengasumsikan independensi fitur, kurang akurat pada gejala tumpang tindih.
- **Akurasi**: 93%.

### 4. Algoritma XGBoost
- **Kelebihan**: Kuat menangani data kompleks, mendukung regularisasi.
- **Kekurangan**: Rentan terhadap kesalahan silang antar kelas.
- **Akurasi**: 91%.

### Pemilihan Model Terbaik
SVM dengan tuning Optuna dipilih karena:
- Akurasi tertinggi (98.5%).
- Konsistensi tinggi di semua kelas.
- Minim kesalahan pada penyakit penting seperti diabetes dan psoriasis.

[gambar perbandingan akurasi model]

## Evaluation
### Metrik Evaluasi
- **Confusion Matrix**: [gambar confusion matrix SVM] menunjukkan klasifikasi sempurna pada 21 dari 24 penyakit.
- **Classification Report**: [gambar tabel classification report] menunjukkan precision, recall, dan F1-score rata-rata 98.32% untuk SVM.
- **Cross-Validation**: Akurasi rata-rata SVM sebesar 98.23% pada 5-fold cross-validation.

### Studi Kasus
- **Input Gejala**: "increased thirst and hunger, frequent urination, unexplained weight loss, blurred vision, slow healing wounds".
- **Prediksi**: Diabetes (sesuai label aktual).
- **Interpretasi**: Sistem berhasil mengenali pola gejala diabetes dengan akurat, menunjukkan potensi sebagai alat bantu diagnosis awal.

### Kelemahan
- Model seperti Naive Bayes dan XGBoost kesulitan membedakan penyakit dengan gejala mirip (misalnya, drug reaction vs. GERD).
- Hasil prediksi harus divalidasi oleh profesional medis.

## Kesimpulan
1. **Sistem Klasifikasi**: Sistem NLP berbasis SVM mampu memprediksi penyakit dengan akurasi 98.5% berdasarkan gejala teks.
2. **Algoritma Terbaik**: SVM dengan tuning Optuna memberikan keseimbangan terbaik antara akurasi dan generalisasi.
3. **Dampak Klinis**: Sistem ini sangat potensial untuk diagnosis awal di lingkungan dengan sumber daya terbatas, tetapi tidak menggantikan dokter.

## Referensi
1. Dessi, D., Helaoui, R., Kumar, V., Recupero, D. R., & Riboni, D. (2021). TF-IDF vs word embeddings for morbidity identification in clinical notes: An initial study. *arXiv*. https://arxiv.org/abs/2105.09632
2. Kalra, S., Li, L., & Tizhoosh, H. R. (2019). Automatic classification of pathology reports using TF-IDF features. *arXiv*. https://arxiv.org/abs/1903.07406
3. Lai, L.-H., Lin, Y.-L., Liu, Y.-H., Lai, J.-P., Yang, W.-C., Hou, H.-P., & Pai, P.-F. (2024). The use of machine learning models with Optuna in disease prediction. *Electronics, 13*(23), 4775. https://doi.org/10.3390/electronics13234775
4. Rudd, J. M. (2017). Application of support vector machine modeling and graph theory metrics for disease classification. *arXiv*. https://arxiv.org/abs/1708.00122
