# 📜 OCR Aksara Jawa — Streamlit App

Aplikasi web berbasis **Streamlit** untuk mengenali aksara Jawa (*Javanese script*) dalam gambar naskah kuno menggunakan pipeline deep learning **YOLOv8 + CRNN + CTC Decoding**.

---

## 📋 Daftar Isi

- [Gambaran Umum](#gambaran-umum)
- [Arsitektur Pipeline](#arsitektur-pipeline)
- [Detail Model](#detail-model)
- [Struktur Folder](#struktur-folder)
- [Instalasi](#instalasi)
- [Cara Penggunaan](#cara-penggunaan)
- [Penjelasan Teknis](#penjelasan-teknis)
- [Konfigurasi](#konfigurasi)

---

## Gambaran Umum

Aplikasi ini mengimplementasikan pipeline OCR (*Optical Character Recognition*) end-to-end untuk mengenali aksara Jawa dalam gambar dokumen/naskah kuno. Sistem ini menghasilkan **transliterasi Latin** dari aksara Jawa yang terdeteksi.

### Fitur Utama

| Fitur | Deskripsi |
|-------|-----------|
| 🔍 **Deteksi Kata** | YOLOv8 mendeteksi lokasi setiap kata dalam gambar |
| 🧠 **Pengenalan Karakter** | CRNN Hybrid mengenali karakter dari setiap kata |
| 📊 **Visualisasi Hasil** | Bounding box, tabel confidence, dan grid crop kata |
| ⏱️ **Metrik Waktu** | Menampilkan waktu deteksi, pengenalan, dan total |
| 🎯 **Confidence Threshold** | Slider untuk mengatur sensitivitas deteksi |

---

## Arsitektur Pipeline

```
📷 Input Image (Gambar Naskah Kuno)
       │
       ▼
┌──────────────────────┐
│  YOLOv8 Detection    │  ← Model: yolov8_best.pt
│  (Deteksi Kata)      │  ← Input: gambar (640×640)
│  Output: Bounding    │  ← Conf threshold: 0.25
│  Boxes + Confidence  │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Crop & Preprocess   │  ← Crop setiap bounding box
│  Resize: 32 × 128   │  ← Normalisasi ImageNet
│  Normalize: μ,σ      │     mean=[0.485, 0.456, 0.406]
└──────────┬───────────┘     std=[0.229, 0.224, 0.225]
           │
           ▼
┌──────────────────────┐
│  CRNN Recognition    │  ← Model: crnn_best.pt
│  VGG16 → Adapt →     │  ← Input: (3, 32, 128)
│  BiLSTM → FC         │  ← Output: (16, 23)
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  CTC Greedy Decode   │  ← Hapus blank (idx=0)
│  Collapse repeated   │  ← Hapus karakter berulang
└──────────┬───────────┘
           │
           ▼
📝 Transliterasi Latin
   (contoh: "kangjêng surakarta")
```

---

## Detail Model

### 1. YOLOv8 — Deteksi Kata

| Parameter | Nilai |
|-----------|-------|
| **Model** | YOLOv8 (custom trained) |
| **Task** | Object Detection |
| **Input** | Gambar (auto-resize ke 640×640) |
| **Output** | Bounding boxes + confidence scores |
| **Confidence Threshold** | 0.25 (adjustable) |
| **File** | `models/yolov8_best.pt` (6.2 MB) |

### 2. CRNN Hybrid — Pengenalan Karakter

Arsitektur CRNN (*Convolutional Recurrent Neural Network*) hybrid:

```
Input (3, 32, 128)
       │
       ▼
┌─────────────────────────────┐
│ VGG16 Feature Extractor     │  Block 1-3 (layer 0-16)
│ (Frozen, pretrained)        │  Output: (256, 4, 16)
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│ Adaptation Layers           │  2×Conv2d + BN + ReLU + Dropout
│ (Trainable)                 │  + AdaptiveMaxPool2d
│                             │  Output: (16, 512)
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│ Bidirectional LSTM          │  2 layers, hidden=256
│ (Trainable)                 │  Output: (16, 512)
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│ Fully Connected Layer       │  Linear(512, 23)
│ + Softmax + Argmax          │  Output: (16, 23)
└─────────────────────────────┘
```

| Parameter | Nilai |
|-----------|-------|
| **Backbone** | VGG16 Block 1-3 (frozen) |
| **Sequence Model** | BiLSTM 2-layer, hidden=256 |
| **Loss Function** | CTC Loss |
| **Input Size** | 32 × 128 pixels (H × W) |
| **Sequence Length** | 16 time steps |
| **Num Classes** | 23 (22 karakter + 1 CTC blank) |
| **File** | `models/crnn_best.pt` (66.2 MB) |

### 3. Charset

22 karakter Latin yang digunakan untuk transliterasi:

```
a b d e g h i j k l m n o p r s t u w y è ê
```

File: `models/charset.txt`

---

## Struktur Folder

```
app/
├── app.py              # Streamlit main application
├── models.py           # Definisi arsitektur CRNN (PyTorch)
├── pipeline.py         # OCR Pipeline (YOLOv8 + CRNN + CTC)
├── requirements.txt    # Daftar dependensi Python
├── README.md           # Dokumentasi (file ini)
└── models/
    ├── yolov8_best.pt  # Model YOLOv8 (deteksi kata)
    ├── crnn_best.pt    # Model CRNN (pengenalan karakter)
    └── charset.txt     # Daftar karakter (22 huruf Latin)
```

### Penjelasan File

| File | Deskripsi |
|------|-----------|
| **app.py** | Antarmuka Streamlit dengan tema dark premium. Menangani upload gambar, menampilkan hasil deteksi, tabel confidence, grid crop, dan metrik waktu. |
| **models.py** | Definisi arsitektur CRNN Hybrid: `VGG16FeatureExtractor`, `AdaptationLayers`, `BidirectionalLSTM`, `CRNNHybrid`. |
| **pipeline.py** | Class `OCRPipeline` yang menggabungkan YOLOv8 + CRNN + `CTCLabelConverter` menjadi satu pipeline inference. |
| **requirements.txt** | Daftar library Python yang diperlukan beserta versi minimumnya. |

---

## Instalasi

### Prasyarat

- **Python** 3.9 atau lebih baru
- **pip** (Python package manager)

### Langkah Instalasi

1. **Masuk ke direktori app:**

   ```bash
   cd "reverse engineer/app"
   ```

2. **Install dependensi:**

   ```bash
   pip install -r requirements.txt
   ```

   Dependensi utama:

   | Library | Fungsi |
   |---------|--------|
   | `streamlit` | Framework web app |
   | `torch` + `torchvision` | PyTorch deep learning framework |
   | `ultralytics` | YOLOv8 inference engine |
   | `opencv-python-headless` | Pemrosesan gambar |
   | `Pillow` | Manipulasi gambar Python |
   | `numpy` | Komputasi numerik |

3. **Verifikasi model tersedia:**

   ```bash
   ls models/
   # Harus ada: yolov8_best.pt  crnn_best.pt  charset.txt
   ```

---

## Cara Penggunaan

### Menjalankan Aplikasi

```bash
cd "reverse engineer/app"
streamlit run app.py
```

Aplikasi akan terbuka di browser pada `http://localhost:8501`.

### Alur Penggunaan

1. **Upload gambar** — Klik area upload atau drag-and-drop gambar naskah aksara Jawa (JPG/PNG)
2. **Atur threshold** — Gunakan slider di sidebar untuk mengatur confidence threshold YOLOv8
3. **Klik "Proses OCR"** — Tunggu model memproses gambar
4. **Lihat hasil:**
   - **Metrik:** Jumlah kata terdeteksi, waktu proses
   - **Transliterasi:** Teks Latin hasil pengenalan
   - **Gambar annotated:** Bounding box dengan label pada gambar asli
   - **Tabel detail:** Confidence deteksi dan pengenalan per kata
   - **Grid crop:** Gambar potongan setiap kata yang terdeteksi
   - **Raw output:** Detail teknis CTC decoding (expandable)

---

## Penjelasan Teknis

### Alur Inferensi Step-by-Step

#### Step 1: Deteksi dengan YOLOv8

```python
results = yolo_model(image, conf=0.25)
# Output: list bounding boxes (x1, y1, x2, y2) + confidence
```

Model YOLOv8 menerima gambar dan mendeteksi lokasi setiap kata dalam naskah. Setiap deteksi menghasilkan bounding box dan skor confidence.

#### Step 2: Crop dan Preprocessing

```python
crop = image[y1:y2, x1:x2]            # Crop bounding box
crop_rgb = cv2.cvtColor(crop, BGR2RGB) # Convert ke RGB
pil_img = Image.fromarray(crop_rgb)    # ke PIL Image

# Transform: resize → tensor → normalize
tensor = transforms.Compose([
    Resize((32, 128)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406],
              std=[0.229, 0.224, 0.225])
])(pil_img)
```

Setiap crop di-resize ke 32×128 piksel dan dinormalisasi menggunakan statistik ImageNet.

#### Step 3: Pengenalan dengan CRNN

```python
output = crnn_model(tensor)  # Shape: (1, 16, 23)
probs = softmax(output, dim=2)
max_probs, indices = probs.max(dim=2)  # Greedy selection
```

CRNN menghasilkan probabilitas 23 kelas untuk setiap 16 time step. Greedy decoding memilih kelas dengan probabilitas tertinggi.

#### Step 4: CTC Decoding

```python
# indices contoh: [0, 0, 11, 11, 0, 2, 2, 0, ...]
# CTC collapse:
#   1. Hapus blank (idx=0)     → [11, 11, 2, 2]
#   2. Hapus karakter berulang → [11, 2]
#   3. Map ke charset          → "ka"
```

CTC (*Connectionist Temporal Classification*) decoding menghapus token blank dan karakter berulang berturutan untuk menghasilkan teks akhir.

---

## Konfigurasi

### Parameter Default

| Parameter | Default | Deskripsi |
|-----------|---------|-----------|
| `yolo_conf` | 0.25 | Minimum confidence untuk deteksi |
| `crnn_img_size` | (32, 128) | Ukuran input CRNN (H × W) |
| `device` | cpu | Device untuk inferensi |

### Mengubah Confidence Threshold

Gunakan slider **"Confidence Threshold"** di sidebar aplikasi (range: 0.05 – 0.95).

- **Nilai rendah (0.05–0.15):** Lebih banyak deteksi, termasuk yang kurang yakin
- **Nilai default (0.25):** Keseimbangan antara presisi dan recall
- **Nilai tinggi (0.50+):** Hanya menampilkan deteksi dengan confidence tinggi

---

## Teknologi yang Digunakan

| Teknologi | Versi | Fungsi |
|-----------|-------|--------|
| **Python** | ≥3.9 | Bahasa pemrograman utama |
| **Streamlit** | ≥1.30 | Framework web app interaktif |
| **PyTorch** | ≥2.0 | Deep learning framework |
| **Torchvision** | ≥0.15 | VGG16 pretrained model |
| **Ultralytics** | ≥8.0 | YOLOv8 inference engine |
| **OpenCV** | ≥4.8 | Pemrosesan dan anotasi gambar |
| **Pillow** | ≥10.0 | Preprocessing gambar |
| **NumPy** | ≥1.24 | Operasi array dan numerik |

---

*Dibuat untuk mengenali dan melestarikan aksara Jawa dalam naskah kuno melalui teknologi AI modern.*
