# Perbandingan Naive Bayes vs BERT untuk Analisis Sentimen Ulasan Wisata Taman Dolan

Proyek Tugas Akhir ini membandingkan performa model **Naive Bayes (ComplementNB)** dan **BERT (IndoBERT)** dalam melakukan klasifikasi sentimen terhadap ulasan wisata **Taman Dolan, Kota Batu**. Ulasan dikategorikan ke dalam tiga kelas: **positif**, **netral**, dan **negatif**, berdasarkan rating yang diberikan pengunjung.

Kedua model dirancang untuk menangani masalah **data tidak seimbang (imbalanced)** menggunakan pendekatan class weight, dengan tambahan GridSearchCV untuk tuning hyperparameter Naive Bayes dan frozen layer + early stopping untuk BERT.

## Dataset

- **Sumber**: Ulasan pengunjung Taman Dolan, Kota Batu
- **Jumlah data**: 2.069 ulasan
- **Kolom**: `User`, `rating`, `Tanggal`, `ulasan`


- **Aturan labeling sentimen**:
  - Rating ≥ 4 → **positif**
  - Rating = 3 → **netral**
  - Rating < 3 → **negatif**

File dataset: `2069_ulasan_TamanDolan.csv`

## Metodologi

| Aspek | Naive Bayes | BERT |
|-------|-------------|------|
| Model | ComplementNB | IndoBERT (`indobenchmark/indobert-base-p1`) |
| Fitur | TF-IDF (unigram + bigram, max_features=10000) | Frozen embedding + 6 encoder layer bawah |
| Imbalanced handling | Class weight (sample_weight) | Class weight pada CrossEntropyLoss |
| Tuning | GridSearchCV 5-fold (alpha: 0.01–5.0) | Weight decay 0.01, warmup ratio 0.1, early stopping (patience=3) |
| Preprocessing | Cleaning + slang normalization + stopword removal + stemming | Cleaning + slang normalization (ringan, konteks dipertahankan) |

**Split data**: 70% train / 15% validation / 15% test (stratified)

## Cara Menjalankan

1. **Clone repository**
   ```bash
   git clone https://github.com/vani2522/Analisis-Sentimen-Ulasan-Wisata.git
   cd nama-repo
   ```

2. **Buka notebook di Google Colab**
   - Upload file `.ipynb` ke Google Colab, atau buka langsung dari GitHub melalui menu `File > Open notebook > GitHub` di Colab.

3. **Siapkan dataset**
   - Upload file `2069_ulasan_TamanDolan.csv` ke Google Drive kamu.
   - Sesuaikan path pada bagian `FILE_PATH` di notebook (Cell "LOAD DATASET DARI GOOGLE DRIVE") agar mengarah ke lokasi file di Drive kamu.

4. **Jalankan notebook secara berurutan**
   - Cell 1: Install dependency (`transformers`, `torch`, `scikit-learn`, `PySastrawi`, dll.)
   - Cell 2–3: Import library & load dataset
   - Cell 4–5: Labeling & text preprocessing
   - Cell 6: Split data (train/val/test)
   - Cell 7: TF-IDF & class weight
   - Cell 8: Training Naive Bayes
   - Cell 9–11: Setup & training BERT
   - Cell 12: Evaluasi BERT
   - Cell 13–14: Visualisasi & ringkasan hasil
   - Cell 15: Simpan model ke Google Drive (opsional)
   - Cell 16 dst: Demo prediksi & analisis error

   > Direkomendasikan menggunakan **GPU runtime** di Colab (`Runtime > Change runtime type > GPU`) agar training BERT lebih cepat.

## Hasil Singkat

| Metrik | Naive Bayes | BERT (IndoBERT) |
|--------|-------------|------------------|
| Accuracy | 0.6694 | **0.8024** |
| Precision | 0.5125 | **0.6796** |
| Recall | 0.6062 | **0.6624** |
| F1-Macro | 0.5322 | **0.6694** |

**Kesimpulan**: BERT (IndoBERT) mengungguli Naive Bayes pada seluruh metrik evaluasi, dengan selisih F1-Macro sebesar **0.1372**. BERT lebih unggul terutama dalam mengenali kelas minoritas (netral & negatif) dibandingkan Naive Bayes, berkat kemampuannya memahami konteks kalimat secara lebih baik — misalnya pada kasus sarkasme atau kalimat dengan sentimen kontradiktif terhadap ratingnya.

## Struktur Repository
```
├── README.md
├── notebook/
│   └── NB_vs_BERT_Analisis_Sentimen_TamanDolan.ipynb
└── dataset/
    └── 2069_ulasan_TamanDolan.csv
```

## Catatan
- Model dan tokenizer BERT hasil training dapat disimpan ke Google Drive melalui cell "SIMPAN MODEL" pada notebook, sehingga bisa dimuat kembali tanpa perlu training ulang.
- Notebook juga menyediakan fitur demo prediksi interaktif serta analisis kesalahan prediksi (error analysis) untuk keperluan pembahasan lebih lanjut dalam laporan TA.