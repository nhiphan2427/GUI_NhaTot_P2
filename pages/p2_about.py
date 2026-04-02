import os
import base64
import streamlit as st
from pages.base import BasePage

_IMG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "images")


class P2AboutPage(BasePage):
    def render(self) -> None:
        self._render_banner()
        st.title("ℹ️ Về dự án — Project 2")

        col1, col2 = st.columns(2)
        with col1:
            self._render_project_info()
        with col2:
            self._render_run_guide()

        st.markdown("---")
        self._render_pipeline()
        st.markdown("---")
       
    def _render_banner(self):
        banner_path = os.path.join(_IMG_DIR, "banner_nhatot.png")
        if os.path.exists(banner_path):
            with open(banner_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            st.markdown(
                f'<img src="data:image/png;base64,{b64}" '
                f'style="width:100%;border-radius:12px;margin-bottom:16px;">',
                unsafe_allow_html=True,
            )

    def _render_project_info(self):
        st.subheader("📌 Thông tin dự án")
        st.markdown("""
        | Mục | Thông tin |
        |-----|-----------|
        | **Lớp** | DL07_DATN_k311_T27 |
        | **Chủ đề** | Hệ thống Gợi ý & Phân cụm Thị trường |
        | **Dữ liệu** | NhaTot.vn · 3 quận TP.HCM |
        | **Framework** | Scikit-learn + PySpark MLlib |
        | **NLP** | SBERT · TF-IDF · BM25 · FastText |
        | **GUI** | Streamlit |
        | **Records** | ~7.939 tin đăng |
        """)

        st.subheader("📁 Cấu trúc file")
        st.code("""
📁 NhaTot/
├── nhatot_p2_app.py        ← Entry point Project 2
├── data_loader_p2.py       ← DataLoaderP2 class
├── models_p2.py            ← RecommenderModel + ClusterModel
├── 📁 pages/
│   ├── p2_business.py      ← Business Problem
│   ├── p2_recommender.py   ← Gợi ý nhà
│   ├── p2_clustering.py    ← Phân cụm thị trường
│   ├── p2_evaluation.py    ← Đánh giá & Report
│   └── p2_about.py         ← Về dự án
├── house_samples.csv       ← Dữ liệu mẫu cho Recommender
├── nha_cosine_sim_23032026.pkl ← Ma trận Cosine Sim
└── nhatot_clustered_final.csv  ← Dữ liệu phân cụm + PCA
        """, language="text")

    def _render_run_guide(self):
        st.subheader("📋 Hướng dẫn chạy app")
        st.code("""
# 1. Cài thư viện
pip install streamlit pandas numpy matplotlib
pip install seaborn scikit-learn

# 2. Chạy Project 2 app
streamlit run nhatot_p2_app.py

# 3. Mở trình duyệt
# http://localhost:8501
        """, language="bash")

        st.subheader("📂 Nguồn notebook")
        notebooks = [
            ("Final_01_Recommender_System_(Hybrid_Model).ipynb",
             "So sánh 4 model (TF-IDF, BM25, FastText, SBERT) → Hybrid: SBERT 40% + Giá 30% + Vị trí 30%"),
            ("Final_02_Market_Segmentation_Scikit-learn.ipynb",
             "K-Means + GMM + Agglomerative (K=2) · 10 features · IQR + LabelEncoder + StandardScaler"),
            ("Final_03_Market_Segmentation_Big Data.ipynb",
             "PySpark MLlib: K-Means + Bisecting K-Means + GMM (K=3) · 3 phân khúc · CH=2770, DB=0.906"),
        ]
        for nb, desc in notebooks:
            st.markdown(f"""
            <div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:10px;
                        padding:10px 14px;margin-bottom:8px">
                <div style="font-size:12px;font-weight:700;color:#E8512A">{nb}</div>
                <div style="font-size:12px;color:#475569;margin-top:3px">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    def _render_pipeline(self):
        st.subheader("📚 Pipeline Project 2")
        st.markdown("""
        ```
        [Dữ liệu thô NhaTot.vn — 3 quận]
               ↓
        [Final_01_Recommender_System_(Hybrid_Model).ipynb]
        NLP: 5 bộ từ điển tiếng Việt, tách từ, chuẩn hóa, bỏ stopword
        So sánh 4 model: TF-IDF, BM25, FastText, SBERT → Chọn SBERT
        Hybrid: SBERT(40%) + Price(30%) + Location(30%)
               ↓
        [nhatot_cleaned_final.csv + nha_cosine_sim.pkl]
               ↓                    ↓
        [Final_02_Segmentation   [Final_03_Segmentation
           Scikit-learn.ipynb]      Big Data (PySpark).ipynb]
        10 features               6 features (Spark Pipeline)
        IQR + LabelEncoder        VectorAssembler + StandardScaler
        K-Means + GMM +           K-Means + Bisecting K-Means +
        Agglomerative (k=2)       GMM (k=3)
        PCA 2D viz                CH=2770, DB=0.906
               ↓
        [nhatot_clustered_final.csv]
        cluster_km · cluster_gmm · PCA1 · PCA2 · segment_name
               ↓
        [GUI Streamlit — nhatot_p2_app.py]
        Gợi ý nhà · Phân cụm · Đánh giá
        ```
        """)

   