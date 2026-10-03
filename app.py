import io
import cv2
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
import streamlit as st
from transformers import pipeline

# 1. KONFIGURASI HALAMAN & CUSTOM MODERN CSS
st.set_page_config(
    page_title="Forensic AI & Deepfake Verification Lab",
    page_icon="🔬",
    layout="wide",
)

# Custom CSS
st.markdown(
    """
    <style>
    /* Styling Header Card */
    .header-box {
        background: linear-gradient(135deg, #1e2638 0%, #0f1422 100%);
        padding: 24px;
        border-radius: 14px;
        border: 1px solid #2d3748;
        margin-bottom: 25px;
    }
    .header-title {
        font-size: 28px;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 6px;
    }
    .header-desc {
        color: #a0aec0;
        font-size: 14px;
    }
    /* Metric Card Custom */
    .metric-card {
        background-color: #171c28;
        border: 1px solid #283149;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }
    .metric-val {
        font-size: 26px;
        font-weight: 700;
    }
    .metric-label {
        font-size: 12px;
        color: #a0aec0;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 18px;
        border-radius: 8px;
    }
    </style>
""",
    unsafe_allow_html=True,
)


# SIDEBAR
with st.sidebar:
    st.markdown("### ⚙️ Control Panel")

    model_choice = st.selectbox(
        "Mesin AI Klasifikasi:",
        [
            "Model Modern AI (Midjourney/DALL-E/Kamera)",
            "Model FaceSwap & GAN Deepfake (Klasik)",
        ],
        help="Pilih Modern AI untuk foto kamera/AI generatif baru. Pilih FaceSwap untuk meme/tukar wajah.",
    )

    threshold = (
        st.slider(
            "Sensitivitas Vonis Fake (%):",
            min_value=50,
            max_value=90,
            value=50,
            help="Ambang batas probabilitas untuk menetapkan vonis mutlak.",
        )
        / 100.0
    )

    ela_quality = st.slider("Sensitivitas Kompresi ELA:", 75, 95, 90)

    st.markdown("---")
    st.caption("🔬 **Digital Forensics Lab** v2.5")
    st.caption("Mendukung Analisis Spektral ViT & Respon Kompresi Piksel.")


# ENGINE MODEL AI
@st.cache_resource
def get_classifier(model_name: str):
    return pipeline("image-classification", model=model_name)


if "Modern" in model_choice:
    model_id = "umm-maybe/AI-image-detector"
else:
    model_id = "dima806/deepfake_vs_real_image_detection"

pipe = get_classifier(model_id)


# FUNGSI FORENSIK DIGITAL
def compute_ela_heatmap(image: Image.Image, quality: int = 90) -> Image.Image:
    """Error Level Analysis yang diubah menjadi Color Heatmap (Inferno)

    agar area modifikasi menyala kontras.
    """
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=quality)
    buffer.seek(0)
    resaved_image = Image.open(buffer)

    # Hitung selisih
    diff = ImageChops.difference(image.convert("RGB"), resaved_image)
    diff_np = np.array(diff)

    # Ubah selisih menjadi grayscale
    gray_diff = cv2.cvtColor(diff_np, cv2.COLOR_RGB2GRAY)

    # Normalisasi kontras agar lebih terang
    cv2.normalize(gray_diff, gray_diff, 0, 255, cv2.NORM_MINMAX)

    # Konversi menjadi heatmap berwarna (COLORMAP_INFERNO: Hitam -> Ungu -> Oranye -> Kuning Menyala)
    heatmap = cv2.applyColorMap(gray_diff, cv2.COLORMAP_INFERNO)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    return Image.fromarray(heatmap_rgb)


def detect_and_crop_face(pil_image: Image.Image):
    """Mendeteksi wajah utama secara otomatis."""
    try:
        cv_img = np.array(pil_image.convert("RGB"))
        gray = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)
        if hasattr(cv2, "CascadeClassifier"):
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
            )
            if len(faces) > 0:
                x, y, w, h = max(faces, key=lambda item: item[2] * item[3])
                pad = int(0.2 * w)
                x_pad = max(0, x - pad)
                y_pad = max(0, y - pad)
                w_pad = min(cv_img.shape[1] - x_pad, w + 2 * pad)
                h_pad = min(cv_img.shape[0] - y_pad, h + 2 * pad)
                return (
                    pil_image.crop(
                        (x_pad, y_pad, x_pad + w_pad, y_pad + h_pad)
                    ),
                    True,
                )
    except Exception:
        pass
    return pil_image, False


# HEADER UTAMA APLIKASI
st.markdown(
    f"""
    <div class="header-box">
        <div class="header-title">🛡️ Forensic Deepfake & Image Tamper Lab</div>
        <div class="header-desc">
            Sistem Terpadu Deteksi Autentisitas Citra Digital &bull; 
            Mesin Aktif: <code style="color:#63b3ed;">{model_id}</code> &bull; 
            Ambang Vonis: <b style="color:#feb2b2;">{threshold*100:.0f}%</b>
        </div>
    </div>
""",
    unsafe_allow_html=True,
)

# Input Tabs
tab_input1, tab_input2 = st.tabs(
    ["📁 Unggah File Gambar", "📸 Ambil Foto via Webcam"]
)
input_image = None

with tab_input1:
    file_uploader = st.file_uploader(
        "Pilih file foto (JPG, PNG, JPEG):",
        type=["jpg", "png", "jpeg"],
        label_visibility="collapsed",
    )
    if file_uploader:
        input_image = Image.open(file_uploader).convert("RGB")

with tab_input2:
    camera_file = st.camera_input(
        "Ambil foto langsung:", label_visibility="collapsed"
    )
    if camera_file:
        input_image = Image.open(camera_file).convert("RGB")

# DASHBOARD HASIL DUA KOLOM
if input_image is not None:
    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

    # Ekstraksi Wajah & ELA Heatmap
    focused_image, face_detected = detect_and_crop_face(input_image)
    ela_heatmap = compute_ela_heatmap(input_image, quality=ela_quality)

    # Inferensi Model AI
    with st.spinner("Menjalankan audit forensik digital..."):
        target_img = input_image if "Modern" in model_choice else focused_image
        raw_preds = pipe(target_img)

        scores = {}
        for item in raw_preds:
            label = item["label"].lower()
            if label in ["artificial", "fake"]:
                scores["FAKE"] = item["score"]
            elif label in ["human", "real"]:
                scores["REAL"] = item["score"]

        fake_score = scores.get("FAKE", 0.0)
        real_score = scores.get("REAL", 0.0)

      # Logika Keputusan
        if fake_score > real_score:
            if fake_score >= threshold:
                badge_type = "error"
                verdict_title = "🚨 TERINDIKASI: MANIPULASI / FAKE"
                verdict_desc = f"Keyakinan AI: **{fake_score*100:.2f}%**. Citra memiliki karakteristik artefak sintetis yang kuat."
            else:
                badge_type = "warning"
                verdict_title = "⚠️ STATUS: MERAGUKAN (Cenderung Palsu)"
                verdict_desc = f"Skor Fake (**{fake_score*100:.2f}%**) melampaui Real, namun belum melewati batas vonis ({threshold*100:.0f}%)."
        else:
            if real_score >= threshold:
                badge_type = "success"
                verdict_title = "✅ TERINDIKASI: FOTO ASLI / AUTENTIK"
                verdict_desc = f"Keyakinan AI: **{real_score*100:.2f}%**. Pola pencahayaan dan noise konsisten dengan kamera alami."
            else:
                badge_type = "warning"
                verdict_title = "⚠️ STATUS: MERAGUKAN (Cenderung Asli)"
                verdict_desc = f"Skor Real (**{real_score*100:.2f}%**) melampaui Fake, namun mendekati batas toleransi ({threshold*100:.0f}%)."

    # Layout Utama: 2 Kolom Berimbang
    col_visual, col_analytics = st.columns([1, 1], gap="large")

    # KOLOM KIRI: Galeri Visual Citra
    with col_visual:
        st.markdown("#### 🖼️ Analisis Visual Multi-Spektral")
        view_tab1, view_tab2, view_tab3 = st.tabs(
            [
                "📷 Citra Asli",
                "🎯 Fokus Biometrik",
                "🔥 Peta Forensik (ELA Heatmap)",
            ]
        )

        with view_tab1:
            st.image(
                input_image,
                use_container_width=True,
                caption="Gambar sumber yang diuji",
            )
        with view_tab2:
            if face_detected:
                st.image(
                    focused_image,
                    use_container_width=True,
                    caption="Region of Interest (ROI) Wajah Berhasil Diisolasi",
                )
            else:
                st.info(
                    "Wajah spesifik tidak terkunci. Memindai seluruh bidang citra."
                )
                st.image(input_image, use_container_width=True)
        with view_tab3:
            st.image(
                ela_heatmap,
                use_container_width=True,
                caption="Warna menyala (Kuning/Oranye) menunjukkan batas tempelan atau manipulasi kompresi.",
            )

    # KOLOM KANAN: Hasil Evaluasi & Metrik
    with col_analytics:
        st.markdown("#### 📊 Hasil Verifikasi Sistem")

        # Kartu Status Utama
        if badge_type == "error":
            st.error(f"### {verdict_title}\n{verdict_desc}")
        elif badge_type == "success":
            st.success(f"### {verdict_title}\n{verdict_desc}")
        else:
            st.warning(f"### {verdict_title}\n{verdict_desc}")

        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

        # Dua Kartu Metrik Angka
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Probabilitas Fake (Sintetis)</div>
                    <div class="metric-val" style="color: #fc8181;">{fake_score * 100:.2f}%</div>
                </div>
            """,
                unsafe_allow_html=True,
            )
        with m_col2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Probabilitas Real (Alami)</div>
                    <div class="metric-val" style="color: #68d391;">{real_score * 100:.2f}%</div>
                </div>
            """,
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

        # Progress bar
        st.write("**Distribusi Keyakinan AI:**")
        st.progress(float(fake_score))
        st.caption(
            f"Fake: {fake_score*100:.1f}% &bull; Real: {real_score*100:.1f}%"
        )

        st.markdown("---")

        # Catatan Diagnostik
        with st.expander("🔍 Lihat Catatan Analisis Forensik Lengkap", expanded=True):
            st.markdown(
                f"""
            - **Integritas Kompresi (Tab 3):** Peta ELA menggunakan skema *Inferno Heatmap*. Perubahan gradasi warna tajam pada batas wajah atau teks mengonfirmasi adanya operasi *editing* digital lokal.
            - **Model ViT:** Ekstraksi fitur visual mengamati pola frekuensi spasial piksel untuk membedakan antara lensa kamera fisik vs generator laten.
            - **Rekomendasi:** {'Gunakan bukti ELA sebagai verifikasi silang terhadap dugaan foto manipulasi.' if badge_type != 'success' else 'Citra menunjukkan konsistensi sensorik yang valid sebagai foto autentik.'}
            """
            )
else:
    st.info(
        "💡 Silakan pilih tab di atas: Unggah gambar atau nyalakan webcam Anda untuk memulai analisis forensik."
    )