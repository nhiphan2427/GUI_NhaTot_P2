import streamlit as st

PAGE_CONFIG = dict(
    page_title="NhaTot Analytics",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

COLORS = {
    "Bình Thạnh": "#7C3AED",
    "Gò Vấp":    "#F43F5E",
    "Phú Nhận":  "#10B981",
    "primary":   "#E8512A",
    "navy":      "#1A2744",
    "success":   "#10B981",
    "warning":   "#F59E0B",
    "danger":    "#F43F5E",
    "blue":      "#2563EB",
}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@300;400;500;600;700;800&display=swap');
*, html, body, [class*="css"] { font-family: 'Be Vietnam Pro', sans-serif; box-sizing: border-box; }

/* ── Ẩn chrome mặc định ── */
[data-testid="stSidebar"],
[data-testid="collapsedControl"],
header[data-testid="stHeader"],
.stDeployButton, footer { display: none !important; }

/* ── Nền tổng thể ── */
.stApp { background: #F1F5F9; }

/* ── Xoá TOÀN BỘ khoảng trắng trên cùng ── */
.stApp,
.stApp > div,
.stApp > div > div,
section[data-testid="stAppViewContainer"],
section[data-testid="stAppViewContainer"] > div,
div[data-testid="stAppViewBlockContainer"],
.main,
.main > div,
.block-container {
    padding-top: 0 !important;
    margin-top: 0 !important;
}
.main .block-container {
    padding-top: 0 !important;
    padding-left: 0 !important;
    padding-right: 0 !important;
    max-width: 100% !important;
}

/* ════════════════════════════════════════
   HEADER  (logo · brand · nav · stats)
════════════════════════════════════════ */
.app-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: #1A2744;
    padding: 0 40px;
    height: 96px;
    position: sticky;
    top: 0;
    z-index: 999;
    box-shadow: 0 4px 16px rgba(0,0,0,0.3);
    margin-top: 0 !important;
}
.app-logo-sq {
    background: #E8512A;
    color: white;
    font-weight: 800;
    font-size: 20px;
    width: 58px; height: 58px;
    border-radius: 14px;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    letter-spacing: -0.5px;
    box-shadow: 0 4px 14px rgba(232,81,42,0.45);
}
.app-brand-name { font-size: 20px; font-weight: 800; color: white; line-height: 1.2; }
.app-brand-sub  { font-size: 12px; color: rgba(255,255,255,0.45); letter-spacing: 0.3px; margin-top: 3px; }
.hd-stat {
    background: rgba(255,255,255,0.1);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 14px;
    padding: 10px 22px;
    text-align: center;
    min-width: 84px;
}
.hd-stat-num   { font-size: 22px; font-weight: 800; color: white; line-height: 1; }
.hd-stat-label { font-size: 10px; color: rgba(255,255,255,0.5); letter-spacing: 0.8px; text-transform: uppercase; margin-top: 4px; }

/* ════════════════════════════════════════
   TAB NAV  (dark bar · orange active)
════════════════════════════════════════ */
.stTabs [data-baseweb="tab-list"] {
    background: #1A2744 !important;
    border-bottom: none !important;
    padding: 8px 28px !important;
    gap: 4px !important;
    margin: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    color: rgba(255,255,255,0.65) !important;
    background: transparent !important;
    border-radius: 8px !important;
    border: none !important;
    border-bottom: none !important;
    padding: 8px 16px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    transition: all 0.2s !important;
}
.stTabs [data-baseweb="tab"]:hover {
    background: rgba(255,255,255,0.1) !important;
    color: white !important;
}
.stTabs [aria-selected="true"] {
    background: #E8512A !important;
    color: white !important;
    font-weight: 700 !important;
    box-shadow: 0 2px 8px rgba(232,81,42,0.4) !important;
}
.stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"] { display: none !important; }
.stTabs [data-baseweb="tab-panel"] { padding: 24px 28px !important; }

/* ════════════════════════════════════════
   HERO BANNER
════════════════════════════════════════ */
.hero-card {
    background: linear-gradient(135deg, #1A2744 0%, #1E3560 55%, #2D1B69 100%);
    border-radius: 20px;
    padding: 48px 52px 44px 52px;
    color: white;
    position: relative;
    overflow: hidden;
    margin-bottom: 24px;
}
.hero-card::before {
    content: '';
    position: absolute;
    top: -80px; right: -80px;
    width: 320px; height: 320px;
    border-radius: 50%;
    background: rgba(139,90,200,0.25);
}
.hero-card::after {
    content: '';
    position: absolute;
    top: 100px; right: 60px;
    width: 180px; height: 180px;
    border-radius: 50%;
    background: rgba(90,40,140,0.2);
}
.hero-tag {
    color: #E8512A;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    margin-bottom: 18px;
}
.hero-title {
    font-size: 42px;
    font-weight: 800;
    line-height: 1.2;
    color: white;
    margin-bottom: 18px;
}
.hero-accent { color: #E8512A; }
.hero-desc {
    font-size: 15px;
    color: rgba(255,255,255,0.65);
    max-width: 620px;
    line-height: 1.7;
    margin-bottom: 36px;
    position: relative; z-index: 1;
}
.hero-stats {
    display: flex;
    gap: 0;
    border-top: 1px solid rgba(255,255,255,0.12);
    padding-top: 28px;
    position: relative; z-index: 1;
}
.hero-stat { flex: 1; padding-right: 24px; }
.hero-stat + .hero-stat {
    padding-left: 24px;
    border-left: 1px solid rgba(255,255,255,0.12);
}
.hero-stat-num   { font-size: 30px; font-weight: 800; color: white; line-height: 1; }
.hero-stat-label { font-size: 12px; color: rgba(255,255,255,0.55); margin-top: 6px; }

/* ════════════════════════════════════════
   WHITE KPI CARDS
════════════════════════════════════════ */
.kpi-w {
    background: white;
    border: 1px solid #E8EEF6;
    border-radius: 16px;
    padding: 20px 22px 16px 22px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.05);
    height: 100%;
}
.kpi-w-label {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1.2px;
    color: #94A3B8;
    text-transform: uppercase;
    margin-bottom: 10px;
}
.kpi-w-value {
    font-size: 30px;
    font-weight: 800;
    color: #1E293B;
    line-height: 1;
    margin-bottom: 6px;
}
.kpi-w-sub   { font-size: 13px; color: #64748B; margin-bottom: 14px; }
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
}
.badge-green  { background: #ECFDF5; color: #059669; }
.badge-blue   { background: #EFF6FF; color: #2563EB; }
.badge-yellow { background: #FFFBEB; color: #D97706; }
.badge-red    { background: #FFF1F2; color: #E11D48; }

/* ════════════════════════════════════════
   CHART CARD
════════════════════════════════════════ */
.chart-card {
    background: white;
    border: 1px solid #E8EEF6;
    border-radius: 16px;
    padding: 20px 22px 16px 22px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.05);
    margin-bottom: 20px;
}
.chart-card-title {
    font-size: 14px;
    font-weight: 700;
    color: #1E293B;
    margin-bottom: 14px;
    padding-bottom: 10px;
    border-bottom: 1px solid #F1F5F9;
}

/* ════════════════════════════════════════
   SECTION TITLE
════════════════════════════════════════ */
.section-title {
    font-size: 17px;
    font-weight: 700;
    color: #1E293B;
    margin: 28px 0 16px 0;
}
.section-label {
    font-size: 10px; font-weight: 700; letter-spacing: 1px;
    color: #94A3B8; text-transform: uppercase; margin: 18px 0 10px 0;
}

/* ════════════════════════════════════════
   PRICE CARD  (gradient)
════════════════════════════════════════ */
.price-card {
    background: linear-gradient(135deg, #1A2744, #2D3A8C);
    border-radius: 16px;
    padding: 24px;
    color: white;
    margin-bottom: 14px;
    box-shadow: 0 8px 24px rgba(26,39,68,0.35);
    position: relative; overflow: hidden;
}
.price-card::before {
    content:''; position:absolute; top:-40px; right:-40px;
    width:140px; height:140px; border-radius:50%;
    background: rgba(255,255,255,0.07);
}
.price-big  { font-size: 38px; font-weight: 800; color: white; line-height:1.1; margin: 6px 0 8px 0; }
.price-sub  { font-size: 14px; color: rgba(255,255,255,0.7); }
.price-range{ font-size: 12px; color: rgba(255,255,255,0.55); margin-top:8px; padding-top:8px; border-top:1px solid rgba(255,255,255,0.15); }

/* ════════════════════════════════════════
   ANOMALY CARDS
════════════════════════════════════════ */
.anomaly-ok {
    background:#ECFDF5; border:1.5px solid #6EE7B7; border-radius:14px;
    padding:16px 20px; margin-bottom:14px; display:flex; align-items:flex-start; gap:14px;
}
.anomaly-warn {
    background:#FFFBEB; border:1.5px solid #FCD34D; border-radius:14px;
    padding:16px 20px; margin-bottom:14px; display:flex; align-items:flex-start; gap:14px;
}
.anomaly-danger {
    background:#FFF1F2; border:1.5px solid #FCA5A5; border-radius:14px;
    padding:16px 20px; margin-bottom:14px; display:flex; align-items:flex-start; gap:14px;
}

/* ════════════════════════════════════════
   FEATURE BARS
════════════════════════════════════════ */
.feat-row       { margin-bottom: 12px; }
.feat-label-row { display:flex; justify-content:space-between; margin-bottom:5px; font-size:13px; color:#475569; font-weight:500; }
.feat-bar-bg    { background:#E2E8F0; border-radius:6px; height:8px; overflow:hidden; }
.feat-bar-fill  { background: linear-gradient(90deg, #E8512A, #FF8C69); height:8px; border-radius:6px; }

/* ════════════════════════════════════════
   RESULT HEADER / BADGE
════════════════════════════════════════ */
.result-page-header { display:flex; justify-content:space-between; align-items:center; margin-bottom:16px; }
.model-badge {
    background:#FFF0EB; color:#C03A1A; padding:5px 14px;
    border-radius:20px; font-size:12px; font-weight:700; border:1px solid #FCCBB8;
}

/* ════════════════════════════════════════
   FORM LABELS
════════════════════════════════════════ */
[data-testid="stSelectbox"] label,
[data-testid="stSlider"] label,
[data-testid="stNumberInput"] label {
    text-transform: uppercase; font-size:10px !important;
    font-weight:700 !important; letter-spacing:0.8px; color:#94A3B8 !important;
}

/* ════════════════════════════════════════
   BUTTONS
════════════════════════════════════════ */
.stButton > button {
    background: #E8512A; color: white; border: none;
    border-radius: 10px; font-weight: 700; font-size: 14px;
    transition: all 0.25s; width: 100%;
    box-shadow: 0 4px 14px rgba(232,81,42,0.35);
}
.stButton > button:hover { background:#c94020; transform:translateY(-2px); box-shadow:0 6px 18px rgba(232,81,42,0.45); }

/* ════════════════════════════════════════
   ANOMALY PAGE – METHOD CARDS & SCORE
════════════════════════════════════════ */
.method-card {
    background: white;
    border: 1px solid #E8EEF6;
    border-radius: 16px;
    padding: 20px 22px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    border-top: 4px solid;
    height: 100%;
}
.method-card-red    { border-top-color: #F43F5E; }
.method-card-yellow { border-top-color: #F59E0B; }
.method-card-blue   { border-top-color: #2563EB; }
.method-tag {
    font-size: 10px; font-weight: 700; letter-spacing: 1.2px;
    text-transform: uppercase; color: #94A3B8; margin-bottom: 8px;
}
.method-title { font-size: 17px; font-weight: 700; color: #1E293B; margin-bottom: 8px; }
.method-desc  { font-size: 13px; color: #64748B; line-height: 1.6; margin-bottom: 14px; }
.method-badge {
    display: inline-block; padding: 4px 12px; border-radius: 20px;
    font-size: 12px; font-weight: 600;
}
.mb-red    { background: #FFF1F2; color: #E11D48; }
.mb-yellow { background: #FFFBEB; color: #D97706; }
.mb-blue   { background: #EFF6FF; color: #2563EB; }

.score-card {
    background: white; border: 1px solid #E8EEF6;
    border-radius: 16px; padding: 24px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
}
.score-label { font-size: 13px; color: #94A3B8; margin-bottom: 4px; }
.score-num   { font-size: 68px; font-weight: 800; color: #E8512A; line-height: 1; text-align: center; margin: 8px 0 16px 0; }
.score-num-ok      { color: #10B981; }
.score-num-warning { color: #F59E0B; }
.score-num-danger  { color: #F43F5E; }

.verdict-ok {
    background: #ECFDF5; border: 1.5px solid #6EE7B7; border-radius: 12px;
    padding: 14px 18px; display: flex; align-items: center; gap: 12px; margin-bottom: 16px;
}
.verdict-warn {
    background: #FFFBEB; border: 1.5px solid #FCD34D; border-radius: 12px;
    padding: 14px 18px; display: flex; align-items: center; gap: 12px; margin-bottom: 16px;
}
.verdict-danger {
    background: #FFF1F2; border: 1.5px solid #FCA5A5; border-radius: 12px;
    padding: 14px 18px; display: flex; align-items: center; gap: 12px; margin-bottom: 16px;
}
.verdict-icon {
    width: 34px; height: 34px; border-radius: 8px; display: flex;
    align-items: center; justify-content: center; font-size: 16px;
    font-weight: 700; flex-shrink: 0; color: white;
}
.vi-ok     { background: #10B981; }
.vi-warn   { background: #F59E0B; }
.vi-danger { background: #F43F5E; }

.detail-bar-row { margin-bottom: 14px; }
.detail-bar-label {
    display: flex; justify-content: space-between;
    font-size: 13px; color: #475569; font-weight: 500; margin-bottom: 5px;
}
.detail-bar-bg   { background: #F1F5F9; border-radius: 6px; height: 10px; overflow: hidden; }
.detail-bar-fill { height: 10px; border-radius: 6px; }
.dbf-red    { background: linear-gradient(90deg, #F43F5E, #FB7185); }
.dbf-yellow { background: linear-gradient(90deg, #F59E0B, #FCD34D); }
.dbf-blue   { background: linear-gradient(90deg, #2563EB, #60A5FA); }

/* anomaly table */
.ano-table { width:100%; border-collapse:collapse; font-size:13px; }
.ano-table th {
    text-transform: uppercase; font-size: 10px; font-weight: 700;
    letter-spacing: 0.8px; color: #94A3B8; padding: 8px 12px;
    border-bottom: 2px solid #F1F5F9; text-align: left;
}
.ano-table td { padding: 10px 12px; border-bottom: 1px solid #F8FAFC; color: #334155; }
.ano-table tr:hover td { background: #F8FAFC; }
.score-pill {
    display: inline-block; padding: 3px 10px; border-radius: 20px;
    font-weight: 700; font-size: 12px;
}
.sp-red    { background: #FFF1F2; color: #E11D48; }
.sp-yellow { background: #FFFBEB; color: #D97706; }
.sp-green  { background: #ECFDF5; color: #059669; }
.check-yes { color: #10B981; font-weight: 700; }
.check-no  { color: #CBD5E1; font-weight: 700; }

/* misc */
[data-testid="stDataFrame"]  { border-radius:12px; overflow:hidden; box-shadow:0 2px 8px rgba(0,0,0,0.06); }
.stAlert                      { border-radius:12px !important; }
h1  { color:#1E293B !important; font-size:24px !important; font-weight:800 !important; padding-top:0; }
h2,h3 { color:#1E293B !important; font-weight:700 !important; }
.info-box {
    background: #EEF2FF; border-left:4px solid #1A2744;
    padding:12px 16px; border-radius:10px; margin:0 0 20px 0;
    font-size:13px; color:#1E293B;
}
[data-testid="stMetricValue"] {
    font-size: 16px !important;
    white-space: nowrap;
    overflow: visible !important;
}

/* ════════════════════════════════════════
   TEAM FOOTER
════════════════════════════════════════ */
.team-footer {
    background: #1A2744;
    border-radius: 16px;
    padding: 32px 40px;
    margin: 40px 28px 20px 28px;
    color: white;
}
.team-footer-title {
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #E8512A;
    margin-bottom: 20px;
    padding-bottom: 12px;
    border-bottom: 1px solid rgba(255,255,255,0.1);
}
.team-footer-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 20px;
    margin-bottom: 20px;
}
.team-member {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px;
    padding: 16px 20px;
}
.team-member-name {
    font-size: 14px;
    font-weight: 700;
    color: white;
    margin-bottom: 6px;
}
.team-member-mail {
    font-size: 12px;
    color: rgba(255,255,255,0.5);
}
.team-footer-instructor {
    text-align: center;
    font-size: 13px;
    color: rgba(255,255,255,0.55);
    padding-top: 16px;
    border-top: 1px solid rgba(255,255,255,0.1);
}
.team-footer-instructor span {
    color: white;
    font-weight: 600;
}
</style>
"""

TYPE_MAP  = {"Hem": "Hẻm", "MatTien": "Mặt tiền", "BietThu": "Biệt thự"}
LEGAL_MAP = {
    "DaCoSo": "Đã có sổ", "DangChoSo": "Đang chờ sổ",
    "GiayToKhac": "Giấy tờ khác", "Chưa xác định": "Chưa xác định",
}

def apply_config():
    st.set_page_config(**PAGE_CONFIG)
    st.markdown(CSS, unsafe_allow_html=True)
    # Ngăn Google Translate tự động dịch trang (phá DOM của Streamlit/React)
    st.markdown('<meta name="google" content="notranslate">', unsafe_allow_html=True)
