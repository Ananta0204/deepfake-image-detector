# 🛡️ Forensic AI & Deepfake Verification Lab

[![Streamlit](https://img.shields.io/badge/Streamlit-1.44+-FF4B4B.svg?style=flat&logo=Streamlit)](https://streamlit.io)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg?style=flat&logo=huggingface)](https://huggingface.co)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8.svg?style=flat&logo=opencv)](https://opencv.org)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?style=flat&logo=python)](https://python.org)

Sistem antarmuka web interaktif berbasis **Streamlit** dan **Deep Learning** untuk memverifikasi autentisitas citra digital, mendeteksi manipulasi wajah (*Deepfake / Face-swap*), serta mengenali gambar buatan generator AI modern (*Midjourney, DALL-E, Stable Diffusion*). 

Proyek ini dikembangkan sebagai pemenuhan Tugas Kelompok Mata Kuliah **Kecerdasan Buatan / Computer Vision** — **Universitas Lampung**.

---

## 🌟 Fitur Utama

- **🧠 Dual-Engine AI Architecture:**
  - *Model Modern AI (`umm-maybe/AI-image-detector`):* Spesialis mendeteksi citra hasil sintesis generator AI modern (Midjourney, DALL-E 3, Flux) vs rekaman sensor kamera fisik.
  - *Model FaceSwap & GAN (`dima806/deepfake_vs_real_image_detection`):* Berbasis **Vision Transformer (ViT)** khusus mendeteksi manipulasi biometrik wajah dan rekayasa deepfake klasik.
- **🔥 Error Level Analysis (ELA) Inferno Heatmap:**
  - Memvisualisasikan perbedaan rasio kompresi digital secara langsung.
  - Area editan, manipulasi lokal, atau tempelan stiker/wajah akan menyala terang (warna oranye/kuning) pada peta spektral.
- **🎯 Deteksi & Isolasi Wajah Otomatis (Face ROI):**
  - Mengisolasi area wajah utama menggunakan algoritma Haar Cascade agar model AI fokus pada biometrik wajah tanpa terganggu teks meme atau latar belakang.
- **📸 Fleksibilitas Sumber Citra:**
  - Mendukung unggah berkas lokal (`JPG`, `JPEG`, `PNG`) serta pengujian langsung menggunakan **Webcam**.
- **⚖️ Threshold Sensitivity Tuning:**
  - Slider kalibrasi toleransi vonis (50% – 90%) untuk mencegah *false alarm* pada foto kamera ponsel berkondisi alami.

---

## 🛠️ Arsitektur Alur Sistem

```text
[ Input Citra (Upload / Webcam) ]
                │
       ┌────────┴────────┐
       ▼                 ▼
[ Haar Cascade ]   [ ELA Heatmap Engine ]
(Ekstraksi Wajah)   (Peta Anomali Kompresi)
       │
       ▼
[ Vision Transformer (ViT) Pipeline ]
  ├─ Ekstraksi Token Spasial Citra
  └─ Inferensi Probabilitas (Real vs Fake)
       │
       ▼
[ Decision Engine & Threshold Calibration ]
       │
       ▼
[ Dashboard Visualisasi Streamlit ]

🚀 Panduan Instalasi & Penggunaan
1. Kloning Repository
git clone https://github.com/Ananta0204/deepfake-image-detector.git
cd deepfake-image-detector

2. Buat & Aktifkan Virtual Environment
# Windows
python -m venv env
env\Scripts\activate

# macOS / Linux
python3 -m venv env
source env/bin/activate

3. Pasang Dependensi
pip install -r requirements.txt

4. Jalankan Aplikasi
streamlit run app.py
Aplikasi akan otomatis berjalan pada peramban web di alamat: http://localhost:8501.

📁 Struktur Direktori Proyek
deepfake-image-detector/
│
├── app.py              # Logika utama aplikasi, pipeline AI, dan antarmuka Streamlit
├── requirements.txt    # Daftar dependensi dan pustaka Python
├── .gitignore          # Konfigurasi pengecualian file sistem/environment
└── README.md           # Dokumentasi teknis proyek



