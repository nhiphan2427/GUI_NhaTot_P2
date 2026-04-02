import sys, os, base64
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
from config import apply_config
from data_loader_p2 import DataLoaderP2
from pages.p2_business    import P2BusinessPage
from pages.p2_recommender import P2RecommenderPage
from pages.p2_clustering  import P2ClusteringPage
from pages.p2_evaluation  import P2EvaluationPage
from pages.p2_about       import P2AboutPage

apply_config()

# ── Header ─────────────────────────────────────────────────────────────────────
df_cl     = DataLoaderP2.load_clustered()
n_total   = len(df_cl)
n_segments = df_cl["segment_name"].nunique() if "segment_name" in df_cl.columns else 2
sil_score  = 0.339  # PySpark K=3 Silhouette (Notebook 03)

_logo_path = os.path.join(os.path.dirname(__file__), "images", "nhatot.jpg")
if os.path.exists(_logo_path):
    with open(_logo_path, "rb") as _f:
        _logo_b64 = base64.b64encode(_f.read()).decode()
    _logo_html = (
        f'<img src="data:image/jpeg;base64,{_logo_b64}" '
        f'style="height:58px;width:auto;border-radius:14px;'
        f'image-rendering:crisp-edges;image-rendering:-webkit-optimize-contrast;">'
    )
else:
    _logo_html = '<div class="app-logo-sq">NT</div>'

st.markdown(f"""
<div class="app-header">
    <div style="display:flex;align-items:center;gap:12px">
        {_logo_html}
        <div>
            <div class="app-brand-name">NhaTot Analytics — Project 2</div>
            <div class="app-brand-sub">Recommender System &amp; Market Segmentation · DL07_DATN_k311_T27</div>
        </div>
    </div>
    <div style="display:flex;gap:10px;align-items:center">
        <div class="hd-stat">
            <div class="hd-stat-num">{n_total:,}</div>
            <div class="hd-stat-label">Tin đăng</div>
        </div>
        <div class="hd-stat">
            <div class="hd-stat-num">{n_segments}</div>
            <div class="hd-stat-label">Phân khúc</div>
        </div>
        <div class="hd-stat">
            <div class="hd-stat-num">{sil_score}</div>
            <div class="hd-stat-label">Silhouette</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Tab navigation ─────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋  Business Problem",
    "🏠  Gợi ý nhà",
    "📊  Phân cụm TT",
    "🔍  Evaluation & Report",
    "ℹ️  Về dự án",
])

with tab1: P2BusinessPage().render()
with tab2: P2RecommenderPage().render()
with tab3: P2ClusteringPage().render()
with tab4: P2EvaluationPage().render()
with tab5: P2AboutPage().render()

# ── Team Footer ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="team-footer">
    <div class="team-footer-title">Thành viên nhóm</div>
    <div class="team-footer-grid">
        <div class="team-member">
            <div class="team-member-name">Nguyễn Lý Kim Ngân</div>
            <div class="team-member-mail">ngan.nguyen.287@gmail.com</div>
        </div>
        <div class="team-member">
            <div class="team-member-name">Phan Yến Nhi</div>
            <div class="team-member-mail">nhiphan2427@gmail.com</div>
        </div>
        <div class="team-member">
            <div class="team-member-name">Trương Đình Huy Du</div>
            <div class="team-member-mail">truongdinhhuydu123@gmail.com</div>
        </div>
    </div>
    <div class="team-footer-instructor">
        Giảng viên hướng dẫn: <span>Khuất Thùy Phương</span>
    </div>
</div>
""", unsafe_allow_html=True)
