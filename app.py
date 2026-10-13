"""
Dashboard Kualitas Udara Kota Kendari
Tab 1 : Analisis Data (Pertanyaan 1-3) dengan filter tanggal
Tab 2 : Model Machine Learning (Pertanyaan 4 - Random Forest Regressor)

Jalankan dengan:  streamlit run app.py
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# ============================================================
# KONFIGURASI HALAMAN
# ============================================================
st.set_page_config(
    page_title="Kualitas Udara Kendari",
    page_icon="🌸",
    layout="wide",
)

BASE_DIR = Path(__file__).parent

# Palet warna pastel
PINK = "#FF8FB1"
LAVENDER = "#B39DFF"
MINT = "#6FD8B0"
PEACH = "#FFB38A"
SKY = "#8CC8FF"
YELLOW = "#FFD97D"
TEXT = "#4A3F6B"
SOFT = "#8A7FA8"

WARNA_PARAM = {
    "pm2_5": PINK,
    "pm10": PEACH,
    "ozone": MINT,
    "carbon_monoxide": LAVENDER,
}
LABEL_PARAM = {
    "pm2_5": "PM2.5",
    "pm10": "PM10",
    "ozone": "O₃ (Ozon)",
    "carbon_monoxide": "CO",
}
SATUAN_PARAM = {
    "pm2_5": "µg/m³",
    "pm10": "µg/m³",
    "ozone": "µg/m³",
    "carbon_monoxide": "µg/m³",
}

LABEL_FITUR = {
    "pm10": "PM10 (µg/m³)",
    "carbon_monoxide": "CO (µg/m³)",
    "nitrogen_dioxide": "NO₂ (µg/m³)",
    "sulphur_dioxide": "SO₂ (µg/m³)",
    "ozone": "O₃ (µg/m³)",
    "visibility": "Visibilitas (m)",
    "shortwave_radiation": "Radiasi Shortwave (W/m²)",
    "direct_radiation": "Radiasi Langsung (W/m²)",
    "diffuse_radiation": "Radiasi Difus (W/m²)",
    "uv_index": "Indeks UV",
}

# ============================================================
# CSS (tampilan cute)
# ============================================================
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800&display=swap');

    html, body, [class*="css"], .stMarkdown, .stText, button, label {{
        font-family: 'Nunito', sans-serif !important;
    }}
    .stApp {{
        background: linear-gradient(160deg, #FFF3F8 0%, #F4EFFF 50%, #EAF8F3 100%);
        color: {TEXT};
    }}
    header[data-testid="stHeader"] {{ background: transparent; }}
    .block-container {{ padding-top: 2rem; max-width: 1200px; }}

    /* Hero */
    .hero {{
        background: linear-gradient(120deg, {PINK} 0%, {LAVENDER} 100%);
        border-radius: 28px;
        padding: 28px 34px;
        color: white;
        box-shadow: 0 10px 30px rgba(179,157,255,.35);
        margin-bottom: 18px;
    }}
    .hero h1 {{ margin: 0; font-size: 2.1rem; font-weight: 800; color: white; }}
    .hero p  {{ margin: 6px 0 0 0; font-size: 1.02rem; opacity: .95; }}

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 10px; background: transparent; border-bottom: none;
    }}
    .stTabs [data-baseweb="tab"] {{
        background: white; border-radius: 999px; padding: 10px 22px;
        font-weight: 700; color: {SOFT};
        box-shadow: 0 4px 12px rgba(179,157,255,.15);
        border: 2px solid transparent;
    }}
    .stTabs [aria-selected="true"] {{
        background: linear-gradient(120deg, {PINK}, {LAVENDER});
        color: white !important;
    }}
    .stTabs [data-baseweb="tab-highlight"],
    .stTabs [data-baseweb="tab-border"] {{ display: none; }}

    /* Kartu */
    .card {{
        background: rgba(255,255,255,.85);
        border-radius: 22px; padding: 18px 22px;
        box-shadow: 0 8px 24px rgba(179,157,255,.18);
        border: 1.5px solid rgba(255,255,255,.9);
        margin-bottom: 14px;
    }}
    .metric {{
        border-radius: 22px; padding: 16px 20px; color: white;
        box-shadow: 0 8px 20px rgba(0,0,0,.08);
    }}
    .metric .lbl {{ font-size: .85rem; font-weight: 700; opacity: .95; }}
    .metric .val {{ font-size: 1.9rem; font-weight: 800; line-height: 1.2; }}
    .metric .sub {{ font-size: .78rem; opacity: .95; }}

    .q-title {{
        font-size: 1.2rem; font-weight: 800; color: {TEXT};
        margin: 6px 0 2px 0;
    }}
    .q-desc {{ color: {SOFT}; font-size: .92rem; margin-bottom: 10px; }}
    .insight {{
        background: #FFF9E6; border-left: 6px solid {YELLOW};
        border-radius: 14px; padding: 12px 16px; color: {TEXT};
        font-size: .95rem; margin-top: 6px;
    }}
    .badge {{
        display: inline-block; padding: 6px 16px; border-radius: 999px;
        color: white; font-weight: 800; font-size: 1rem;
    }}
    div[data-testid="stSlider"] label p,
    div[data-testid="stDateInput"] label p {{ font-weight: 700; color: {TEXT}; }}
    .stButton > button {{
        background: linear-gradient(120deg, {PINK}, {LAVENDER});
        color: white; border: none; border-radius: 999px;
        padding: 10px 28px; font-weight: 800;
        box-shadow: 0 6px 16px rgba(255,143,177,.4);
    }}
    .stButton > button:hover {{ color: white; transform: translateY(-1px); }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# FUNGSI BANTU
# ============================================================
@st.cache_data
def muat_data() -> pd.DataFrame:
    df = pd.read_csv(BASE_DIR / "dataset_analisis.csv", parse_dates=["datetime"])
    return df.sort_values("datetime").reset_index(drop=True)


@st.cache_resource
def muat_model():
    model = joblib.load(BASE_DIR / "model_random_forest_pm25.pkl")
    fitur = joblib.load(BASE_DIR / "fitur_model.pkl")
    return model, fitur


@st.cache_data
def evaluasi_model(_model, fitur: list, df: pd.DataFrame):
    """Pembagian data sama persis dengan di notebook (test_size=0.2, random_state=42)."""
    X, y = df[fitur], df["pm2_5"]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_pred = _model.predict(X_test)
    return {
        "y_test": y_test.values,
        "y_pred": y_pred,
        "mae": mean_absolute_error(y_test, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "r2": r2_score(y_test, y_pred),
    }


def hex_ke_rgba(hex_warna: str, alpha: float = 0.15) -> str:
    """Ubah '#RRGGBB' menjadi 'rgba(r,g,b,a)' (kompatibel dengan semua versi plotly)."""
    h = hex_warna.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def kartu_metrik(label, nilai, sub, warna):
    return f"""
    <div class="metric" style="background:linear-gradient(135deg,{warna},{warna}CC);">
        <div class="lbl">{label}</div>
        <div class="val">{nilai}</div>
        <div class="sub">{sub}</div>
    </div>
    """


def interpretasi_korelasi(r: float) -> str:
    a = abs(r)
    if a < 0.2:
        kekuatan = "sangat lemah"
    elif a < 0.4:
        kekuatan = "lemah"
    elif a < 0.6:
        kekuatan = "sedang"
    elif a < 0.8:
        kekuatan = "kuat"
    else:
        kekuatan = "sangat kuat"
    arah = "positif" if r > 0 else "negatif"
    return f"{kekuatan} dan {arah}"


def tata_letak(fig, tinggi=380, **kw):
    fig.update_layout(
        height=tinggi,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.6)",
        font=dict(family="Nunito, sans-serif", color=TEXT),
        margin=dict(l=20, r=20, t=50, b=20),
        **kw,
    )
    fig.update_xaxes(gridcolor="#EFE9FB", zerolinecolor="#EFE9FB")
    fig.update_yaxes(gridcolor="#EFE9FB", zerolinecolor="#EFE9FB")
    return fig


def scatter_dengan_tren(df, x, y, warna, judul, xlabel, ylabel):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df[x], y=df[y], mode="markers", name="Data per jam",
            marker=dict(color=warna, size=8, opacity=0.6,
                        line=dict(color="white", width=1)),
        )
    )
    if len(df) >= 3 and df[x].nunique() > 1:
        koef = np.polyfit(df[x], df[y], 1)
        xs = np.linspace(df[x].min(), df[x].max(), 50)
        fig.add_trace(
            go.Scatter(x=xs, y=np.polyval(koef, xs), mode="lines",
                       name="Garis tren",
                       line=dict(color=TEXT, width=3, dash="dash"))
        )
    fig.update_layout(title=judul, xaxis_title=xlabel, yaxis_title=ylabel,
                      legend=dict(orientation="h", y=-0.2))
    return tata_letak(fig, 400)


def kategori_pm25(nilai: float):
    """Acuan: pedoman WHO 2021 (15 µg/m³, 24 jam) & baku mutu PP 22/2021 (55 µg/m³, 24 jam)."""
    if nilai <= 15:
        return "Baik 😊", MINT, "Di bawah pedoman WHO (15 µg/m³)."
    if nilai <= 55:
        return "Sedang 🙂", YELLOW, "Di atas pedoman WHO, tetapi masih di bawah baku mutu Indonesia (55 µg/m³)."
    return "Tidak Sehat 😷", PINK, "Melewati baku mutu Indonesia (55 µg/m³). Kurangi aktivitas di luar ruangan."


# ============================================================
# LOAD DATA
# ============================================================
df = muat_data()
tgl_min, tgl_max = df["datetime"].dt.date.min(), df["datetime"].dt.date.max()

st.markdown(
    """
    <div class="hero">
        <h1>🌸 Dashboard Kualitas Udara Kota Kendari</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

tab1, tab2 = st.tabs(["🌤️  Analisis Data (Pertanyaan 1–3)", "🤖  Model Machine Learning (Pertanyaan 4)"])

# ============================================================
# TAB 1 : ANALISIS DATA
# ============================================================
with tab1:
    # ---------- Filter tanggal ----------
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### 📅 Pilih Rentang Tanggal")
    rentang = st.date_input(
        "Tanggal mulai – tanggal akhir",
        value=(tgl_min, tgl_max),
        min_value=tgl_min,
        max_value=tgl_max,
        format="DD/MM/YYYY",
        help="Semua grafik dan angka di tab ini mengikuti rentang tanggal yang dipilih.",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # date_input bisa mengembalikan 1 tanggal saja saat user baru klik tanggal pertama
    if isinstance(rentang, (tuple, list)):
        if len(rentang) == 2:
            mulai, akhir = rentang
        else:
            mulai = akhir = rentang[0]
    else:
        mulai = akhir = rentang

    mask = (df["datetime"].dt.date >= mulai) & (df["datetime"].dt.date <= akhir)
    dff = df.loc[mask].copy()

    if dff.empty:
        st.warning("Tidak ada data pada rentang tanggal tersebut 🥺")
        st.stop()

    st.caption(
        f"Menampilkan **{len(dff):,} data per jam** dari "
        f"**{mulai.strftime('%d %b %Y')}** sampai **{akhir.strftime('%d %b %Y')}**."
    )

    # ---------- Pertanyaan 1 ----------
    st.markdown('<div class="q-title">1️⃣ Kondisi Kualitas Udara</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="q-desc">Bagaimana rata-rata konsentrasi PM2.5, PM10, O₃, dan CO '
        "pada rentang tanggal yang dipilih?</div>",
        unsafe_allow_html=True,
    )

    kolom_udara = ["pm2_5", "pm10", "ozone", "carbon_monoxide"]
    rata = dff[kolom_udara].mean()

    cols = st.columns(4)
    for c, k in zip(cols, kolom_udara):
        c.markdown(
            kartu_metrik(LABEL_PARAM[k], f"{rata[k]:.2f}", f"rata-rata • {SATUAN_PARAM[k]}", WARNA_PARAM[k]),
            unsafe_allow_html=True,
        )
    st.write("")

    col_a, col_b = st.columns([1, 1.6])

    with col_a:
        fig_bar = go.Figure(
            go.Bar(
                x=[LABEL_PARAM[k] for k in kolom_udara],
                y=[rata[k] for k in kolom_udara],
                marker=dict(color=[WARNA_PARAM[k] for k in kolom_udara]),
                text=[f"{rata[k]:.1f}" for k in kolom_udara],
                textposition="outside",
            )
        )
        fig_bar.update_layout(title="Rata-rata Parameter Kualitas Udara",
                              yaxis_title="Nilai rata-rata", showlegend=False)
        st.plotly_chart(tata_letak(fig_bar, 400), width="stretch")

    with col_b:
        harian = dff.set_index("datetime")[kolom_udara].resample("D").mean().round(2)
        fig_line = make_subplots(
            rows=2, cols=2,
            subplot_titles=[LABEL_PARAM[k] for k in kolom_udara],
            vertical_spacing=0.18, horizontal_spacing=0.09,
        )
        for i, k in enumerate(kolom_udara):
            fig_line.add_trace(
                go.Scatter(
                    x=harian.index, y=harian[k], mode="lines+markers",
                    line=dict(color=WARNA_PARAM[k], width=3, shape="spline"),
                    marker=dict(size=6, color="white",
                                line=dict(color=WARNA_PARAM[k], width=2)),
                    fill="tozeroy",
                    fillcolor=hex_ke_rgba(WARNA_PARAM[k], 0.15),
                    name=LABEL_PARAM[k],
                    hovertemplate="%{x|%d %b}<br>%{y:.2f}<extra></extra>",
                ),
                row=i // 2 + 1, col=i % 2 + 1,
            )
        fig_line.update_layout(title="Tren Rata-rata Harian", showlegend=False)
        st.plotly_chart(tata_letak(fig_line, 400), width="stretch")

    # tanggal tertinggi
    st.markdown("**🏆 Tanggal dengan rata-rata harian tertinggi**")
    cols = st.columns(4)
    for c, k in zip(cols, kolom_udara):
        tgl = harian[k].idxmax()
        c.markdown(
            f"""<div class="card" style="text-align:center;padding:12px;">
            <div style="color:{WARNA_PARAM[k]};font-weight:800;">{LABEL_PARAM[k]}</div>
            <div style="font-size:1.05rem;font-weight:800;">{tgl.strftime('%d %b %Y')}</div>
            <div style="color:{SOFT};font-size:.85rem;">{harian[k].max():.2f} {SATUAN_PARAM[k]}</div>
            </div>""",
            unsafe_allow_html=True,
        )

    st.divider()

    # ---------- Pertanyaan 2 ----------
    st.markdown('<div class="q-title">2️⃣ Radiasi Matahari & Ozon (O₃)</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="q-desc">Apakah radiasi matahari berhubungan dengan perubahan kadar ozon?</div>',
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns([2.2, 1])
    with c1:
        fig2 = scatter_dengan_tren(
            dff, "shortwave_radiation", "ozone", MINT,
            "Radiasi Matahari vs Ozon", "Shortwave Radiation (W/m²)", "Ozon O₃ (µg/m³)",
        )
        st.plotly_chart(fig2, width="stretch")
    with c2:
        if len(dff) >= 3 and dff["shortwave_radiation"].nunique() > 1:
            r2_ = dff["shortwave_radiation"].corr(dff["ozone"])
            st.markdown(
                kartu_metrik("Korelasi Pearson (r)", f"{r2_:.3f}", interpretasi_korelasi(r2_), MINT),
                unsafe_allow_html=True,
            )
            st.markdown(
                f"""<div class="insight">💡 Hubungan radiasi matahari dan ozon
                <b>{interpretasi_korelasi(r2_)}</b>. Semakin tinggi radiasi matahari, kadar ozon
                {'cenderung ikut naik' if r2_ > 0 else 'cenderung turun'}.</div>""",
                unsafe_allow_html=True,
            )
        else:
            st.info("Pilih rentang tanggal yang lebih panjang untuk menghitung korelasi.")

    st.divider()

    # ---------- Pertanyaan 3 ----------
    st.markdown('<div class="q-title">3️⃣ Visibilitas & PM2.5</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="q-desc">Apakah visibilitas (jarak pandang) memengaruhi konsentrasi PM2.5?</div>',
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns([2.2, 1])
    with c1:
        fig3 = scatter_dengan_tren(
            dff, "visibility", "pm2_5", LAVENDER,
            "Visibilitas vs PM2.5", "Visibilitas (m)", "PM2.5 (µg/m³)",
        )
        st.plotly_chart(fig3, width="stretch")
    with c2:
        if len(dff) >= 3 and dff["visibility"].nunique() > 1:
            r3_ = dff["visibility"].corr(dff["pm2_5"])
            st.markdown(
                kartu_metrik("Korelasi Pearson (r)", f"{r3_:.3f}", interpretasi_korelasi(r3_), LAVENDER),
                unsafe_allow_html=True,
            )
            st.markdown(
                f"""<div class="insight">💡 Hubungan visibilitas dan PM2.5
                <b>{interpretasi_korelasi(r3_)}</b>. Saat visibilitas
                {'menurun, PM2.5 cenderung naik' if r3_ < 0 else 'naik, PM2.5 cenderung ikut naik'}.</div>""",
                unsafe_allow_html=True,
            )
        else:
            st.info("Pilih rentang tanggal yang lebih panjang untuk menghitung korelasi.")

# ============================================================
# TAB 2 : MODEL MACHINE LEARNING
# ============================================================
with tab2:
    model, fitur = muat_model()
    hasil = evaluasi_model(model, fitur, df)

    st.markdown('<div class="q-title">4️⃣ Prediksi PM2.5 dengan Random Forest Regressor</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="q-desc">Model dilatih dengan 10 fitur (kualitas udara, radiasi matahari, dan visibilitas) '
        "untuk memprediksi konsentrasi PM2.5. Data uji = 20% (random_state 42), sama seperti di notebook.</div>",
        unsafe_allow_html=True,
    )

    # ---------- Metrik evaluasi ----------
    cols = st.columns(3)
    cols[0].markdown(kartu_metrik("MAE", f"{hasil['mae']:.3f}", "rata-rata selisih absolut", PINK), unsafe_allow_html=True)
    cols[1].markdown(kartu_metrik("RMSE", f"{hasil['rmse']:.3f}", "akar rata-rata galat kuadrat", LAVENDER), unsafe_allow_html=True)
    cols[2].markdown(kartu_metrik("R² Score", f"{hasil['r2']:.3f}", "kemampuan model menjelaskan variasi", MINT), unsafe_allow_html=True)
    st.write("")

    col_a, col_b = st.columns(2)

    with col_a:
        y_t, y_p = hasil["y_test"], hasil["y_pred"]
        lo, hi = float(min(y_t.min(), y_p.min())), float(max(y_t.max(), y_p.max()))
        fig_ap = go.Figure()
        fig_ap.add_trace(go.Scatter(
            x=y_t, y=y_p, mode="markers", name="Data uji",
            marker=dict(color=PINK, size=9, opacity=0.7, line=dict(color="white", width=1)),
        ))
        fig_ap.add_trace(go.Scatter(
            x=[lo, hi], y=[lo, hi], mode="lines", name="Prediksi ideal",
            line=dict(color=TEXT, dash="dash", width=2),
        ))
        fig_ap.update_layout(title="PM2.5 Aktual vs Prediksi",
                             xaxis_title="PM2.5 Aktual", yaxis_title="PM2.5 Prediksi",
                             legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(tata_letak(fig_ap, 420), width="stretch")

    with col_b:
        imp = pd.Series(model.feature_importances_, index=fitur).sort_values()
        fig_imp = go.Figure(go.Bar(
            x=imp.values, y=[LABEL_FITUR[k] for k in imp.index], orientation="h",
            marker=dict(color=imp.values, colorscale=[[0, "#FFE0EC"], [1, LAVENDER]]),
            text=[f"{v:.1%}" for v in imp.values], textposition="outside",
        ))
        fig_imp.update_layout(title="Tingkat Kepentingan Fitur", xaxis_title="Feature importance")
        st.plotly_chart(tata_letak(fig_imp, 420), width="stretch")

    st.markdown(
        f"""<div class="insight">💡 Model sangat akurat (R² = {hasil['r2']:.3f}).
        Fitur paling berpengaruh adalah <b>{LABEL_FITUR[imp.index[-1]].split(' (')[0]}</b>
        ({imp.iloc[-1]:.1%}), karena PM2.5 adalah bagian dari partikel PM10 sehingga keduanya
        sangat berkorelasi.</div>""",
        unsafe_allow_html=True,
    )

    st.divider()

    # ---------- Prediksi interaktif ----------
    st.markdown('<div class="q-title">🎀 Coba Prediksi PM2.5</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="q-desc">Isi nilai kondisi udara & cuaca pada kolom di bawah, lalu klik tombol prediksi.</div>',
        unsafe_allow_html=True,
    )

    nilai_input = {}
    kiri, kanan = st.columns(2)
    for i, f in enumerate(fitur):
        target_col = kiri if i % 2 == 0 else kanan
        lo, hi = float(df[f].min()), float(df[f].max())
        default = float(df[f].median())
        langkah = 1.0 if hi - lo > 50 else 0.1
        with target_col:
            nilai_input[f] = st.number_input(
                LABEL_FITUR[f],
                min_value=0.0,
                value=default,
                step=langkah,
                key=f"in_{f}",
                help=f"Rentang data: {lo:g} – {hi:g} (rata-rata {df[f].mean():.2f})",
            )

    if st.button("✨ Prediksi PM2.5"):
        X_baru = pd.DataFrame([nilai_input], columns=fitur)
        pred = float(model.predict(X_baru)[0])
        nama, warna, ket = kategori_pm25(pred)

        st.write("")
        g1, g2 = st.columns([1.2, 1])
        with g1:
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number",
                value=pred,
                number=dict(suffix=" µg/m³", font=dict(color=TEXT)),
                title=dict(text="Prediksi PM2.5", font=dict(color=TEXT)),
                gauge=dict(
                    axis=dict(range=[0, 80], tickcolor=SOFT),
                    bar=dict(color=warna, thickness=0.35),
                    bgcolor="white",
                    steps=[
                        dict(range=[0, 15], color="#D9F5EA"),
                        dict(range=[15, 55], color="#FFF1C9"),
                        dict(range=[55, 80], color="#FFD9E6"),
                    ],
                ),
            ))
            st.plotly_chart(tata_letak(fig_g, 320), width="stretch")
        with g2:
            st.markdown(
                f"""<div class="card" style="text-align:center;margin-top:30px;">
                <div style="color:{SOFT};font-weight:700;">Kategori</div>
                <span class="badge" style="background:{warna};">{nama}</span>
                <p style="margin-top:14px;">{ket}</p>
                </div>""",
                unsafe_allow_html=True,
            )

    st.caption("Catatan: model dilatih dari data Kendari 1 Sep – 6 Okt 2026, jadi hasil prediksi paling andal "
               "untuk nilai input dalam rentang data tersebut.")