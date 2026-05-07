import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ============================================================
# KONFIGURASI HALAMAN
# ============================================================
st.set_page_config(
    page_title="AvocaRipe — Prediksi Kematangan Alpukat",
    page_icon="🥑",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Space+Mono:wght@400;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Background utama */
    .stApp {
        background: #f4faf0;
    }

    /* ===== SIDEBAR ===== */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a3a1a 0%, #2d5a27 60%, #3d7a35 100%);
        border-right: none;
    }
    [data-testid="stSidebar"] * {
        color: #e8f5e2 !important;
    }
    [data-testid="stSidebar"] .stMarkdown h2 {
        color: #a8e063 !important;
        font-size: 1rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }

    /* ===== TABS ===== */
    .stTabs [data-baseweb="tab-list"] {
        background: white;
        border-radius: 16px;
        padding: 5px;
        gap: 4px;
        box-shadow: 0 2px 12px rgba(30, 80, 20, 0.08);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.88rem;
        color: #4a7a3a;
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #2d5a27, #4a9e3f) !important;
        color: white !important;
    }

    /* ===== BUTTON ===== */
    .stButton > button {
        background: linear-gradient(135deg, #2d5a27 0%, #4a9e3f 100%);
        color: white;
        border: none;
        border-radius: 14px;
        font-weight: 700;
        font-size: 1rem;
        padding: 0.75rem 2rem;
        transition: all 0.25s;
        box-shadow: 0 4px 15px rgba(45, 90, 39, 0.3);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(45, 90, 39, 0.4);
    }

    /* ===== METRIC ===== */
    [data-testid="stMetric"] {
        background: white;
        border-radius: 16px;
        padding: 1.2rem 1.4rem;
        box-shadow: 0 2px 10px rgba(30,80,20,0.07);
        border: 1px solid rgba(30,80,20,0.06);
    }
    [data-testid="stMetricLabel"] p { color: #4a7a3a !important; font-weight: 600 !important; }
    [data-testid="stMetricValue"]   { color: #1a3a1a !important; font-weight: 800 !important; }

    /* ===== SELECTBOX & SLIDER label ===== */
    [data-testid="stSlider"] label,
    [data-testid="stSelectbox"] label {
        color: #1a3a1a !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
    }

    /* ===== ALL ### HEADINGS ===== */
    .stMarkdown h3 { color: #1a3a1a !important; font-weight: 800 !important; }
    .stMarkdown h2 { color: #1a3a1a !important; font-weight: 800 !important; }

    /* ===== ALERT ===== */
    [data-testid="stAlert"] {
        border-radius: 14px;
        border: none;
    }

    /* ===== CUSTOM CARDS ===== */
    .hero-card {
        background: linear-gradient(135deg, #1a3a1a 0%, #2d5a27 50%, #4a9e3f 100%);
        border-radius: 24px;
        padding: 2.2rem 2rem;
        color: white;
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
        text-align: center;
    }
    .hero-card::before {
        content: "🥑";
        position: absolute;
        left: 1.5rem;
        top: 50%;
        transform: translateY(-50%);
        font-size: 4rem;
        opacity: 0.2;
    }
    .hero-card::after {
        content: "🥑";
        position: absolute;
        right: 1.5rem;
        top: 50%;
        transform: translateY(-50%);
        font-size: 4rem;
        opacity: 0.2;
    }
    .hero-card h1 {
        font-size: 1.8rem;
        font-weight: 800;
        margin: 0 0 0.3rem 0;
    }
    .hero-card p {
        font-size: 0.9rem;
        opacity: 0.82;
        margin: 0;
    }

    .info-card {
        background: white;
        border-radius: 16px;
        padding: 1.3rem 1.5rem;
        margin-bottom: 0.8rem;
        box-shadow: 0 2px 10px rgba(30,80,20,0.06);
        border-left: 4px solid #4a9e3f;
    }
    .info-card h3 { font-size:0.95rem; font-weight:700; color:#1a3a1a; margin:0 0 0.4rem 0; }
    .info-card p, .info-card li { font-size:0.88rem; color:#4a6a40; line-height:1.7; margin:0; }

    .result-ripe {
        background: linear-gradient(135deg, #d4f7c5, #a8e063);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        border: 2px solid #4a9e3f;
    }
    .result-halfripe {
        background: linear-gradient(135deg, #fff8d4, #ffe066);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        border: 2px solid #f0b429;
    }
    .result-unripe {
        background: linear-gradient(135deg, #e8f5e2, #c8e6c9);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        border: 2px solid #66bb6a;
    }

    .tip-box {
        background: #f1faea;
        border-radius: 12px;
        padding: 0.9rem 1.1rem;
        margin-top: 0.7rem;
        border: 1px solid #b5e0a0;
    }
    .tip-box p { font-size:0.85rem; color:#2d5a27; margin:0; font-weight:500; line-height:1.6; }

    .code-block {
        background: #1e1e2e;
        border-radius: 14px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 0.8rem;
        font-family: 'Space Mono', monospace;
        font-size: 0.82rem;
        color: #cdd6f4;
        line-height: 1.8;
        overflow-x: auto;
    }
    .code-label {
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #4a9e3f;
        margin-bottom: 0.4rem;
    }

    .dev-card {
        background: linear-gradient(135deg, #1a3a1a, #2d5a27);
        border-radius: 20px;
        padding: 2rem;
        color: white;
        text-align: center;
    }
    .dev-card h2 { font-size:1.3rem; font-weight:800; color:white; margin:0.8rem 0 0.2rem 0; }
    .dev-card p  { font-size:0.85rem; opacity:0.75; margin:0; }
    .badge {
        background: rgba(255,255,255,0.15);
        border-radius: 100px;
        padding: 0.25rem 0.9rem;
        font-size:0.75rem;
        font-weight:600;
        display:inline-block;
        margin:0.25rem;
    }

    .section-heading {
        color: #1a3a1a;
        font-size: 1.1rem;
        font-weight: 800;
        margin: 1.2rem 0 0.6rem 0;
        padding-left: 0.5rem;
        border-left: 4px solid #4a9e3f;
    }

    .fact-box {
        background: white;
        border-radius: 14px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(30,80,20,0.07);
        border: 1px solid #d4edce;
    }
    .fact-num { font-family:'Space Mono',monospace; font-size:1.6rem; font-weight:700; color:#2d5a27; }
    .fact-desc { font-size:0.75rem; color:#5a8a50; margin-top:0.2rem; }

    .input-panel {
        background: white;
        border-radius: 18px;
        padding: 1.5rem;
        box-shadow: 0 2px 12px rgba(30,80,20,0.07);
        border: 1px solid rgba(30,80,20,0.06);
        margin-bottom: 1rem;
    }
    .panel-label {
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #4a9e3f;
        margin-bottom: 0.8rem;
    }

    hr { border-color: #d4edce; margin: 1.2rem 0; }

    /* Contact links */
    .contact-link {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        color: #2d5a27;
        text-decoration: none;
        font-weight: 600;
        font-size: 0.88rem;
    }
    .contact-link:hover { color: #4a9e3f; text-decoration: underline; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODEL & DATA
# ============================================================
@st.cache_resource
def load_model():
    return joblib.load("model_tree.joblib")

@st.cache_data
def load_data():
    return pd.read_csv("avocado_ripeness_dataset.csv")

model = load_model()
df    = load_data()


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:

    # ---- LOGO SEKOLAH ----
    st.image("Logo_SMK_Negeri_1_Purbalingga.png", use_container_width=True)

    st.markdown("""
    <div style="text-align:center; padding: 0.5rem 0 1rem 0;">
        <div style="font-size:1.25rem; font-weight:800; color:#a8e063; letter-spacing:0.03em;">AvocaRipe</div>
        <div style="font-size:0.7rem; color:#8ab878; text-transform:uppercase; letter-spacing:0.1em; margin-top:2px;">Avocado Ripeness Predictor</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:rgba(168,224,99,0.2); margin:0 0 1rem 0;'>", unsafe_allow_html=True)

    st.markdown("## ℹ️ Tentang Aplikasi")
    st.markdown("""
    <div style="font-size:0.83rem; color:#c5e0b8; line-height:1.7;">
    Aplikasi berbasis <b style="color:#a8e063;">Machine Learning</b> untuk memprediksi tingkat kematangan alpukat menggunakan data fisik dan sensor.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:rgba(168,224,99,0.2); margin:1rem 0;'>", unsafe_allow_html=True)

    st.markdown("## 🤖 Model")
    st.markdown("""
    <div style="font-size:0.82rem; color:#c5e0b8; line-height:1.9;">
    ✅ &nbsp;<b style="color:#a8e063;">Decision Tree</b><br>
    ✅ &nbsp;Random Forest<br>
    ✅ &nbsp;Logistic Regression
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:rgba(168,224,99,0.2); margin:1rem 0;'>", unsafe_allow_html=True)

    st.markdown("## 📊 Dataset")
    st.markdown(f"""
    <div style="font-size:0.82rem; color:#c5e0b8; line-height:1.9;">
    📁 &nbsp;Avocado Ripeness Dataset<br>
    📋 &nbsp;<b style="color:#fbbf24;">{len(df):,}</b> baris data<br>
    🔖 &nbsp;<b style="color:#fbbf24;">5</b> kelas kematangan<br>
    📌 &nbsp;<b style="color:#fbbf24;">8</b> fitur input
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:rgba(168,224,99,0.2); margin:1rem 0;'>", unsafe_allow_html=True)

    st.markdown("## 🎯 Kelas Target")
    st.markdown("""
    <div style="font-size:0.82rem; color:#c5e0b8; line-height:1.9;">
    🟢 &nbsp;<b style="color:#a8e063;">Ripe</b> — Matang sempurna<br>
    🟡 &nbsp;<b style="color:#fbbf24;">Half-ripe</b> — Setengah matang<br>
    🟤 &nbsp;<b style="color:#d4a574;">Unripe</b> — Belum matang
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:rgba(168,224,99,0.2); margin:1rem 0;'>", unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align:center; font-size:0.7rem; color:#5a7a50;">
        © 2025 AvocaRipe · SMK ML Project
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# JUDUL UTAMA DI TENGAH (di atas tab)
# ============================================================
st.markdown("""
<div style="text-align:center; padding: 1.5rem 0 1rem 0;">
    <div style="font-size:2.2rem; font-weight:900; color:#1a3a1a; letter-spacing:-0.01em; line-height:1.1;">
        🥑 AvocaRipe
    </div>
    <div style="font-size:1rem; color:#4a7a3a; font-weight:500; margin-top:0.3rem;">
        Sistem Prediksi Kematangan Alpukat Berbasis Machine Learning
    </div>
    <div style="display:inline-block; background:#e8f5e2; border-radius:100px; padding:0.3rem 1.2rem; font-size:0.78rem; color:#2d5a27; font-weight:600; margin-top:0.6rem; border:1px solid #b5e0a0;">
        SMK · Project Machine Learning · 2025
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "  🥑  Prediksi  ",
    "  📚  Informasi  ",
    "  💻  Kode  ",
    "  👨‍💻  Developer  "
])


# ============================================================
# TAB 1 — PREDIKSI
# ============================================================
with tab1:

    st.markdown("""
    <div class="hero-card">
        <h1>🥑 Prediksi Kematangan Alpukat</h1>
        <p>Masukkan data karakteristik alpukat di bawah ini, lalu tekan tombol Prediksi.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown('<div class="panel-label">🧪 Karakteristik Fisik & Warna</div>', unsafe_allow_html=True)

        firmness = st.slider(
            "💪 Firmness (Kekerasan)", 0.0, 100.0, 50.0, step=0.5,
            help="Tingkat kekerasan alpukat. Semakin tinggi = semakin keras (belum matang)"
        )
        hue = st.slider(
            "🎨 Hue (Rona Warna)", 0, 360, 100,
            help="Nilai rona warna pada skala 0–360. Alpukat matang cenderung ke arah ungu/hitam."
        )
        saturation = st.slider(
            "🌈 Saturation (Saturasi Warna)", 0, 100, 50,
            help="Tingkat kejernihan/intensitas warna kulit alpukat"
        )
        brightness = st.slider(
            "☀️ Brightness (Kecerahan)", 0, 100, 50,
            help="Tingkat kecerahan permukaan kulit alpukat"
        )

    with col2:
        st.markdown('<div class="panel-label">🔬 Sensor & Dimensi Fisik</div>', unsafe_allow_html=True)

        color_category = st.selectbox(
            "🎨 Kategori Warna Kulit",
            ["green", "dark green", "purple", "black"],
            help="Pilih kategori warna kulit alpukat yang paling mendekati kondisi nyata"
        )
        sound_db = st.slider(
            "🔊 Sound (dB) — Suara Ketukan", 20, 100, 50,
            help="Nilai desibel saat alpukat diketuk. Suara berat/dalam = matang"
        )
        weight_g = st.slider(
            "⚖️ Berat (gram)", 100, 300, 200,
            help="Berat alpukat dalam gram"
        )
        size_cm3 = st.slider(
            "📦 Ukuran (cm³)", 100, 300, 200,
            help="Volume/ukuran fisik alpukat dalam sentimeter kubik"
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Tombol prediksi tengah
    _, col_btn, _ = st.columns([1, 1.5, 1])
    with col_btn:
        prediksi_btn = st.button("🔮 Prediksi Kematangan", use_container_width=True)

    if prediksi_btn:
        data_input = pd.DataFrame([[
            firmness, hue, saturation, brightness,
            color_category, sound_db, weight_g, size_cm3
        ]], columns=[
            "firmness", "hue", "saturation", "brightness",
            "color_category", "sound_db", "weight_g", "size_cm3"
        ])

        try:
            hasil = model.predict(data_input)[0]

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("---")
            st.markdown('<h3 style="color:#1a3a1a; font-weight:800; margin-bottom:0.8rem;">🎯 Hasil Prediksi</h3>', unsafe_allow_html=True)

            col_res1, col_res2 = st.columns([1.2, 1])

            with col_res1:
                if hasil.lower() == "ripe":
                    st.markdown("""
                    <div class="result-ripe">
                        <div style="font-size:3rem;">🥑</div>
                        <div style="font-size:1.6rem; font-weight:800; color:#1a3a1a; margin:0.4rem 0;">Matang Sempurna!</div>
                        <div style="font-size:0.9rem; color:#2d5a27; opacity:0.85;">Alpukat siap dikonsumsi sekarang.</div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown("""
                    <div class="tip-box">
                        <p>✅ <b>Siap disantap!</b> Simpan di kulkas jika belum ingin dimakan hari ini. Konsumsi dalam 1–2 hari untuk kualitas terbaik.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.progress(100)

                elif hasil.lower() == "half-ripe":
                    st.markdown("""
                    <div class="result-halfripe">
                        <div style="font-size:3rem;">⏳</div>
                        <div style="font-size:1.6rem; font-weight:800; color:#7a5000; margin:0.4rem 0;">Setengah Matang</div>
                        <div style="font-size:0.9rem; color:#8a6000; opacity:0.85;">Alpukat hampir siap, butuh waktu sedikit lagi.</div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown("""
                    <div class="tip-box" style="background:#fffbea; border-color:#ffe066;">
                        <p style="color:#7a5000;">⏳ <b>Tunggu 1–2 hari lagi.</b> Simpan di suhu ruang agar proses pematangan berlanjut. Jangan simpan di kulkas dulu.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.progress(55)

                else:
                    st.markdown(f"""
                    <div class="result-unripe">
                        <div style="font-size:3rem;">🌱</div>
                        <div style="font-size:1.6rem; font-weight:800; color:#1a3a1a; margin:0.4rem 0;">Belum Matang ({hasil})</div>
                        <div style="font-size:0.9rem; color:#2d5a27; opacity:0.85;">Alpukat masih perlu waktu beberapa hari.</div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown("""
                    <div class="tip-box">
                        <p>🌱 <b>Simpan di suhu ruang</b> selama 3–5 hari. Hindari kulkas. Letakkan dekat buah pisang untuk mempercepat pematangan.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.progress(20)

            with col_res2:
                st.markdown('<p style="color:#1a3a1a; font-weight:700; font-size:0.95rem; margin-bottom:0.5rem;">📋 Ringkasan Input:</p>', unsafe_allow_html=True)
                ringkasan = {
                    "Fitur": ["Firmness", "Hue", "Saturation", "Brightness", "Warna", "Sound (dB)", "Berat (g)", "Ukuran (cm³)"],
                    "Nilai": [firmness, hue, saturation, brightness, color_category, sound_db, weight_g, size_cm3]
                }
                st.dataframe(pd.DataFrame(ringkasan), hide_index=True, use_container_width=True)

        except Exception as e:
            st.error("❌ Terjadi error saat prediksi. Pastikan file model tersedia.")
            st.exception(e)

    # Chart distribusi dataset
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<h3 style="color:#1a3a1a; font-weight:800; margin-bottom:0.8rem;">📊 Distribusi Kematangan dalam Dataset</h3>', unsafe_allow_html=True)

    col_c1, col_c2 = st.columns(2)

    with col_c1:
        fig1, ax1 = plt.subplots(figsize=(6, 3.5))
        fig1.patch.set_facecolor('#f4faf0')
        ax1.set_facecolor('#f4faf0')
        counts = df["ripeness"].value_counts()
        colors_bar = ['#4a9e3f', '#f0b429', '#66bb6a', '#2d5a27', '#a8e063']
        bars = ax1.bar(counts.index, counts.values, color=colors_bar[:len(counts)],
                       edgecolor='white', linewidth=1.5, width=0.5)
        for bar in bars:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     f'{int(bar.get_height()):,}', ha='center', fontsize=9,
                     fontweight='bold', color='#1a3a1a')
        ax1.set_xlabel("Kelas Kematangan", fontsize=9, color='#4a6a40')
        ax1.set_ylabel("Jumlah Data", fontsize=9, color='#4a6a40')
        ax1.set_title("Jumlah Data per Kelas", fontsize=10, fontweight='bold', color='#1a3a1a', pad=10)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.tick_params(labelsize=8, colors='#4a6a40')
        for sp in ['left','bottom']: ax1.spines[sp].set_color('#d4edce')
        plt.tight_layout()
        st.pyplot(fig1)

    with col_c2:
        fig2, ax2 = plt.subplots(figsize=(6, 3.5))
        fig2.patch.set_facecolor('#f4faf0')
        ax2.set_facecolor('#f4faf0')
        colors_pie = ['#4a9e3f', '#f0b429', '#66bb6a', '#2d5a27', '#a8e063']
        wedges, texts, autotexts = ax2.pie(
            counts.values, labels=counts.index,
            autopct='%1.1f%%', colors=colors_pie[:len(counts)],
            startangle=90, wedgeprops=dict(edgecolor='white', linewidth=2)
        )
        for t in texts:      t.set_fontsize(9); t.set_color('#1a3a1a')
        for at in autotexts: at.set_fontsize(9); at.set_fontweight('bold'); at.set_color('white')
        ax2.set_title("Proporsi Kelas", fontsize=10, fontweight='bold', color='#1a3a1a', pad=10)
        plt.tight_layout()
        st.pyplot(fig2)


# ============================================================
# TAB 2 — INFORMASI
# ============================================================
with tab2:

    st.markdown("""
    <div class="hero-card">
        <h1>📚 Panduan Lengkap Alpukat</h1>
        <p>Pelajari cara mengenali kematangan alpukat dan manfaatnya bagi kesehatan.</p>
    </div>
    """, unsafe_allow_html=True)

    # Fakta cepat
    st.markdown('<h3 style="color:#1a3a1a; font-weight:800; margin-bottom:0.8rem;">⚡ Fakta Singkat Alpukat</h3>', unsafe_allow_html=True)

    col_f1, col_f2, col_f3, col_f4 = st.columns(4)
    for col, num, desc in zip(
        [col_f1, col_f2, col_f3, col_f4],
        ["8–12", "3–5", "6–8","20g"],
        ["Hari pematangan di suhu ruang", "Hari bertahan di kulkas (sudah matang)", "Gram serat per buah alpukat","Lemak sehat per 100g alpukat",]
    ):
        with col:
            st.markdown(f"""
            <div class="fact-box">
                <div class="fact-num">{num}</div>
                <div class="fact-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Tentang alpukat
    st.markdown('<h3 style="color:#1a3a1a; font-weight:800; margin-bottom:0.6rem;">🥑 Apa itu Alpukat?</h3>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-card">
        <h3>🌿 Deskripsi Umum</h3>
        <p>Alpukat (<i>Persea americana</i>) adalah buah tropis kaya nutrisi yang berasal dari Meksiko dan Amerika Tengah. Uniknya, alpukat <b>tidak matang di pohon</b> — proses pematangan baru terjadi setelah dipetik. Ini membuat prediksi kematangan alpukat menjadi tantangan di projek yang akan saya buat ini.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Tingkatan kematangan
    st.markdown('<h3 style="color:#1a3a1a; font-weight:800; margin-bottom:0.8rem;">🔢 Tingkatan Kematangan Alpukat</h3>', unsafe_allow_html=True)

    col_k1, col_k2, col_k3 = st.columns(3)
    with col_k1:
        st.markdown("""
        <div class="info-card" style="border-left-color:#66bb6a;">
            <h3>🌱 Unripe (Belum Matang)</h3>
            <p>
            🎨 Warna kulit: <b>hijau cerah</b><br>
            💪 Tekstur: <b>sangat keras</b><br>
            🔊 Suara ketukan: <b>nyaring/keras</b><br>
            ⏳ Waktu tersisa: <b>3–7 hari</b><br><br>
            ❌ Tidak disarankan dikonsumsi langsung karena rasa masih pahit.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col_k2:
        st.markdown("""
        <div class="info-card" style="border-left-color:#f0b429;">
            <h3>⏳ Half-ripe (Setengah Matang)</h3>
            <p>
            🎨 Warna kulit: <b>hijau tua / ungu muda</b><br>
            💪 Tekstur: <b>sedikit lunak</b><br>
            🔊 Suara ketukan: <b>agak berat</b><br>
            ⏳ Waktu tersisa: <b>1–2 hari</b><br><br>
            ⚠️ Bisa dikonsumsi tapi rasa belum optimal. Lebih baik tunggu sehari lagi.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col_k3:
        st.markdown("""
        <div class="info-card" style="border-left-color:#4a9e3f;">
            <h3>✅ Ripe (Matang Sempurna)</h3>
            <p>
            🎨 Warna kulit: <b>ungu gelap / hitam</b><br>
            💪 Tekstur: <b>lunak saat ditekan</b><br>
            🔊 Suara ketukan: <b>berat/dalam</b><br>
            ⏳ Waktu konsumsi: <b>segera / 1–2 hari</b><br><br>
            ✅ Siap dikonsumsi! Simpan di kulkas untuk memperlambat pematangan.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Cara cek manual
    st.markdown('<h3 style="color:#1a3a1a; font-weight:800; margin-bottom:0.8rem;">👆 Cara Cek Kematangan Secara Manual</h3>', unsafe_allow_html=True)

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.markdown("""
        <div class="info-card" style="border-left-color:#2d5a27;">
            <h3>👀 1. Cek Warna Kulit</h3>
            <p>
            • <b>Hijau cerah</b> → belum matang<br>
            • <b>Hijau tua</b> → hampir matang<br>
            • <b>Ungu / Hitam</b> → matang sempurna<br><br>
            ⚠️ Warna saja tidak cukup — gunakan dikombinasikan cara lain.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-card" style="border-left-color:#4a9e3f;">
            <h3>✋ 2. Tekan Perlahan</h3>
            <p>
            • <b>Sangat keras</b> → belum matang<br>
            • <b>Sedikit lunak</b> → setengah matang<br>
            • <b>Lunak merata</b> → matang sempurna<br>
            • <b>Terlalu lunak/lembek</b> → sudah terlalu matang
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col_m2:
        st.markdown("""
        <div class="info-card" style="border-left-color:#66bb6a;">
            <h3>🔊 3. Ketuk dan Dengarkan</h3>
            <p>
            • <b>Suara nyaring/keras</b> → isi masih padat (belum matang)<br>
            • <b>Suara berat/dalam</b> → isi sudah lunak (matang)<br><br>
            Teknik ini mirip cara cek semangka matang — bunyinya lebih dalam.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-card" style="border-left-color:#f0b429;">
            <h3>🌿 4. Cek Tangkai</h3>
            <p>
            • Cabut tangkai kecil di atas alpukat<br>
            • <b>Coklat gelap di bawah tangkai</b> → matang<br>
            • <b>Hijau di bawah tangkai</b> → belum matang<br>
            • <b>Hitam / busuk</b> → terlalu matang
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Tips penyimpanan
    st.markdown('<h3 style="color:#1a3a1a; font-weight:800; margin-bottom:0.8rem;">💡 Tips Menyimpan Alpukat</h3>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-card" style="background:linear-gradient(135deg,#f4faf0,#e8f5e2); border-left-color:#2d5a27;">
        <h3>📦 Panduan Penyimpanan</h3>
        <p>
        🌡️ &nbsp;<b>Belum matang</b> → Simpan di suhu ruang (20–25°C), jangan masukkan kulkas<br>
        🍌 &nbsp;<b>Mempercepat pematangan</b> → Letakkan bersama pisang atau apel dalam kantong kertas<br>
        ❄️ &nbsp;<b>Sudah matang</b> → Pindahkan ke kulkas, tahan 3–5 hari<br>
        🥄 &nbsp;<b>Sudah dipotong</b> → Olesi permukaan dengan air jeruk nipis, bungkus rapat, simpan di kulkas<br>
        🧊 &nbsp;<b>Beku</b> → Haluskan dulu lalu bekukan — tahan hingga 3 bulan
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Manfaat kesehatan
    st.markdown('<h3 style="color:#1a3a1a; font-weight:800; margin-bottom:0.8rem;">💊 Manfaat Alpukat untuk Kesehatan</h3>', unsafe_allow_html=True)

    col_b1, col_b2, col_b3 = st.columns(3)
    with col_b1:
        st.markdown("""
        <div class="info-card" style="border-left-color:#4a9e3f;">
            <h3>❤️ Jantung Sehat</h3>
            <p>Kandungan lemak tak jenuh tunggal (oleic acid) membantu menurunkan kolesterol LDL dan menjaga kesehatan jantung.</p>
        </div>
        """, unsafe_allow_html=True)

    with col_b2:
        st.markdown("""
        <div class="info-card" style="border-left-color:#2d5a27;">
            <h3>🧠 Otak & Saraf</h3>
            <p>Kaya vitamin K, folat, dan lemak omega-3 yang mendukung fungsi otak, memori, dan sistem saraf pusat.</p>
        </div>
        """, unsafe_allow_html=True)

    with col_b3:
        st.markdown("""
        <div class="info-card" style="border-left-color:#66bb6a;">
            <h3>⚡ Energi & Serat</h3>
            <p>Tinggi kalium, magnesium, dan serat — membantu pencernaan, menstabilkan gula darah, dan meningkatkan energi.</p>
        </div>
        """, unsafe_allow_html=True)

    # Penjelasan fitur ML
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<h3 style="color:#1a3a1a; font-weight:800; margin-bottom:0.8rem;">🤖 Fitur yang Digunakan Model ML</h3>', unsafe_allow_html=True)

    fitur_info = [
        ("💪", "Firmness", "Kekerasan kulit alpukat (0–100). Nilai tinggi = keras = belum matang"),
        ("🎨", "Hue", "Rona warna (0–360°). Merah/ungu = matang, hijau = belum matang"),
        ("🌈", "Saturation", "Intensitas/kejernihan warna kulit (0–100)"),
        ("☀️", "Brightness", "Kecerahan permukaan kulit (0–100)"),
        ("🎨", "Color Category", "Kategori warna: green → dark green → purple → black"),
        ("🔊", "Sound (dB)", "Suara saat diketuk. Suara berat = matang, nyaring = belum matang"),
        ("⚖️", "Weight (g)", "Berat buah dalam gram. Alpukat matang cenderung lebih berat"),
        ("📦", "Size (cm³)", "Volume fisik alpukat dalam sentimeter kubik"),
    ]
    for icon, name, desc in fitur_info:
        st.markdown(f"""
        <div style="display:flex; align-items:flex-start; gap:0.8rem; background:white; border-radius:12px; padding:0.7rem 1rem; margin-bottom:0.4rem; box-shadow:0 1px 6px rgba(30,80,20,0.05); border:1px solid #d4edce;">
            <span style="font-size:1.2rem; min-width:1.5rem;">{icon}</span>
            <div>
                <span style="font-weight:700; font-size:0.88rem; color:#1a3a1a;">{name}</span>
                <span style="font-size:0.82rem; color:#4a7a3a; margin-left:0.5rem;">— {desc}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ============================================================
# TAB 3 — KODE
# ============================================================
with tab3:

    st.markdown("""
    <div class="hero-card">
        <h1>💻 Source Code</h1>
        <p>Kode lengkap untuk membangun model prediksi kematangan alpukat dari awal hingga akhir.</p>
    </div>
    """, unsafe_allow_html=True)

    # Step 1
    st.markdown('<h3 style="color:#1a3a1a; font-weight:800; margin-bottom:0.6rem;">📦 Step 1 — Import Library</h3>', unsafe_allow_html=True)
    st.code("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

print("✅ Semua library berhasil diimport!")""", language="python")

    # Step 2
    st.markdown('<h3 style="color:#1a3a1a; font-weight:800; margin:1rem 0 0.6rem 0;">📂 Step 2 — Load Dataset</h3>', unsafe_allow_html=True)
    st.code("""df = pd.read_csv("avocado_ripeness_dataset.csv")

print(f"Shape  : {df.shape}")
print(f"Kolom  : {list(df.columns)}")
print(f"Target : {df['ripeness'].unique()}")

df.head()""", language="python")

    # Step 3
    st.markdown('<h3 style="color:#1a3a1a; font-weight:800; margin:1rem 0 0.6rem 0;">🔍 Step 3 — Preprocessing</h3>', unsafe_allow_html=True)
    st.code("""# Pisahkan fitur dan target
X = df[["firmness","hue","saturation","brightness",
        "color_category","sound_db","weight_g","size_cm3"]]
y = df["ripeness"]

# Train-test split (80:20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Definisi kolom
numeric_columns = ["firmness","hue","saturation","brightness",
                   "sound_db","weight_g","size_cm3"]
ordinal_columns = ["color_category"]
warna_order     = ["green","dark green","purple","black"]

# ColumnTransformer
preprocessing = ColumnTransformer(
    transformers=[
        ("scaler", StandardScaler(), numeric_columns),
        ("oe", OrdinalEncoder(categories=[warna_order]), ordinal_columns)
    ]
)

print("✅ Preprocessing selesai!")""", language="python")

    # Step 4
    st.markdown('<h3 style="color:#1a3a1a; font-weight:800; margin:1rem 0 0.6rem 0;">🤖 Step 4 — Training Model (Decision Tree)</h3>', unsafe_allow_html=True)
    st.code("""model_tree = Pipeline(
    steps=[
        ("preprocessing", preprocessing),
        ("model", DecisionTreeClassifier(
            max_depth=8,
            min_samples_split=20,
            random_state=42
        ))
    ]
)

model_tree.fit(X_train, y_train)
y_pred_dt = model_tree.predict(X_test)

print("Accuracy :", accuracy_score(y_test, y_pred_dt))
print(classification_report(y_test, y_pred_dt))

# Cross Validation
scores = cross_val_score(model_tree, X_train, y_train, cv=5, scoring="accuracy")
print(f"CV Mean : {scores.mean():.4f} ± {scores.std():.4f}")""", language="python")

    # Step 5
    st.markdown('<h3 style="color:#1a3a1a; font-weight:800; margin:1rem 0 0.6rem 0;">💾 Step 5 — Simpan Model</h3>', unsafe_allow_html=True)
    st.code("""import joblib

joblib.dump(model_tree, "model_tree.joblib")
print("✅ Model berhasil disimpan!")

# Load & prediksi data baru
model_loaded = joblib.load("model_tree.joblib")

data_baru = pd.DataFrame([[93.8, 105, 87, 41, "dark green", 75, 299, 140]],
    columns=["firmness","hue","saturation","brightness",
             "color_category","sound_db","weight_g","size_cm3"])

prediksi = model_loaded.predict(data_baru)[0]
print(f"Prediksi Kematangan: {prediksi}")""", language="python")

    # Info pipeline
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<h3 style="color:#1a3a1a; font-weight:800; margin-bottom:0.6rem;">🔄 Alur Pipeline ML</h3>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-card">
        <h3>📌 Kenapa Menggunakan Pipeline?</h3>
        <p>
        Pipeline menggabungkan langkah preprocessing dan model dalam satu objek sehingga:<br><br>
        ✅ &nbsp;Tidak perlu encode ulang saat prediksi data baru<br>
        ✅ &nbsp;Menghindari data leakage (scaler hanya fit di data training)<br>
        ✅ &nbsp;Kode lebih bersih dan mudah dimaintain<br>
        ✅ &nbsp;Model yang disimpan sudah include preprocessing-nya
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background:#1e1e2e; border-radius:14px; padding:1.2rem 1.5rem; font-family:'Space Mono',monospace; font-size:0.82rem; color:#cdd6f4; line-height:2;">
    <span style="color:#a6e3a1;">Input Raw Data</span><br>
    &nbsp;&nbsp;&nbsp;&nbsp;↓<br>
    <span style="color:#89b4fa;">ColumnTransformer</span><br>
    &nbsp;&nbsp;&nbsp;&nbsp;├── StandardScaler → [firmness, hue, saturation, brightness, sound_db, weight_g, size_cm3]<br>
    &nbsp;&nbsp;&nbsp;&nbsp;└── OrdinalEncoder → [color_category: green(0), dark green(1), purple(2), black(3)]<br>
    &nbsp;&nbsp;&nbsp;&nbsp;↓<br>
    <span style="color:#f38ba8;">DecisionTreeClassifier</span><br>
    &nbsp;&nbsp;&nbsp;&nbsp;↓<br>
    <span style="color:#f9e2af;">Output: unripe / half-ripe / ripe / firm-ripe / breaking</span>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# TAB 4 — DEVELOPER
# ============================================================
with tab4:

    st.markdown("""
    <div class="hero-card">
        <h1>👨‍💻 Tentang Developer</h1>
        <p>Informasi pembuat aplikasi dan teknologi yang digunakan.</p>
    </div>
    """, unsafe_allow_html=True)

    col_dev1, col_dev2 = st.columns([1, 1.6], gap="large")

    with col_dev1:

        # ---- FOTO DEVELOPER ----
        st.image("Ade.jpg", use_container_width=True,
                 caption="Aden Bagus Susilo — Developer")

        st.markdown("""
        <div class="dev-card">
            <h2>Aden Bagus Susilo</h2>
            <p>Machine Learning Student · SMK</p>
            <br>
            <span class="badge">🎓 SMK</span>
            <span class="badge">🤖 Machine Learning</span>
            <span class="badge">🐍 Python</span>
            <span class="badge">🥑 Agriculture Tech</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-card" style="border-left-color:#4a9e3f; margin-top:0.8rem;">
            <h3>📬 Kontak</h3>
            <p>
            📧 &nbsp;<a href="mailto:susilobagusaden@gmail.com" style="color:#2d5a27; font-weight:600; text-decoration:none;">susilobagusaden@gmail.com</a><br>
            📸 &nbsp;<a href="https://www.instagram.com/Radenss" target="_blank" style="color:#2d5a27; font-weight:600; text-decoration:none;">instagram.com/Radenss</a><br>
            💼 &nbsp;<span style="color:#4a6a40; font-size:0.85rem;">Terbuka untuk kolaborasi project ML</span>
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col_dev2:

        # ---- LOGO SEKOLAH ----
        st.image("Logo_SMK_Negeri_1_Purbalingga.png", use_container_width=True)

        st.markdown('<h3 style="color:#1a3a1a; font-weight:800; margin-bottom:0.6rem;">🛠️ Teknologi yang Digunakan</h3>', unsafe_allow_html=True)

        tech_items = [
            ("🐍", "Python 3.x",          "Bahasa pemrograman utama"),
            ("📊", "Pandas",               "Manipulasi dan analisis data"),
            ("🤖", "Scikit-learn",         "Pipeline & algoritma Machine Learning"),
            ("💾", "Joblib",               "Menyimpan dan memuat model (.joblib)"),
            ("📈", "Matplotlib / Seaborn", "Visualisasi data dan grafik"),
            ("🌐", "Streamlit",            "Framework web app interaktif"),
        ]
        for icon, name, desc in tech_items:
            st.markdown(f"""
            <div style="display:flex; align-items:center; gap:0.9rem; background:white; border-radius:12px; padding:0.8rem 1rem; margin-bottom:0.45rem; box-shadow:0 1px 6px rgba(30,80,20,0.06); border:1px solid #d4edce;">
                <div style="font-size:1.4rem; min-width:1.8rem; text-align:center;">{icon}</div>
                <div>
                    <div style="font-weight:700; font-size:0.88rem; color:#1a3a1a;">{name}</div>
                    <div style="font-size:0.78rem; color:#4a7a3a;">{desc}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<h3 style="color:#1a3a1a; font-weight:800; margin-bottom:0.6rem;">🤖 Performa Model</h3>', unsafe_allow_html=True)

        # Accuracy values from notebook output (all 100% on this dataset)
        models_info = [
            ("🔵", "Logistic Regression", "100%", "Model linear untuk klasifikasi"),
            ("🌲", "Random Forest",        "100%", "Ensemble dari banyak decision tree"),
            ("🌳", "Decision Tree",        "100%", "Pohon keputusan — model aktif"),
        ]
        for icon, name, acc, desc in models_info:
            st.markdown(f"""
            <div style="display:flex; align-items:center; justify-content:space-between; background:white; border-radius:12px; padding:0.8rem 1rem; margin-bottom:0.45rem; box-shadow:0 1px 6px rgba(30,80,20,0.06); border:1px solid #d4edce;">
                <div style="display:flex; align-items:center; gap:0.7rem;">
                    <span style="font-size:1.2rem;">{icon}</span>
                    <div>
                        <div style="font-weight:700; font-size:0.88rem; color:#1a3a1a;">{name}</div>
                        <div style="font-size:0.75rem; color:#4a7a3a;">{desc}</div>
                    </div>
                </div>
                <span style="background:linear-gradient(135deg,#2d5a27,#4a9e3f); color:white; border-radius:100px; padding:0.2rem 0.7rem; font-size:0.75rem; font-weight:700; font-family:'Space Mono',monospace;">{acc}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div class="tip-box" style="margin-top:0.5rem;">
            <p>✅ Semua model mencapai akurasi <b>100%</b> pada dataset ini (250 data, test size 20% = 50 data).</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
    <div style="text-align:center; padding:1.5rem; background:white; border-radius:16px; box-shadow:0 2px 10px rgba(30,80,20,0.06); border:1px solid #d4edce;">
        <div style="font-size:0.8rem; color:#6a9a60; font-weight:500;">
            🥑 AvocaRipe &nbsp;·&nbsp; SMK Machine Learning Project &nbsp;·&nbsp; 2025<br>
            <span style="font-size:0.72rem; font-family:'Space Mono',monospace; margin-top:4px; display:block; color:#8aba78;">Built with Python · Scikit-learn · Streamlit</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
