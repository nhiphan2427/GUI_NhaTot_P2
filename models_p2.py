import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from data_loader_p2 import DataLoaderP2


# ── Hàm gợi ý dựa trên Cosine Similarity (SBERT pre-computed) ───────────────
def get_recommendations(df: pd.DataFrame, house_id: int, cosine_sim, nums: int = 5) -> pd.DataFrame:
    """Trả về `nums` căn nhà tương tự nhất với house_id (dùng SBERT cosine sim pre-computed)."""
    matching = df.index[df["id"] == house_id].tolist()
    if not matching:
        return pd.DataFrame()
    idx = matching[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1: nums + 1]
    house_indices = [i[0] for i in sim_scores]
    sim_values    = [round(float(i[1]) * 100, 1) for i in sim_scores]
    result = df.iloc[house_indices].copy()
    result["do_tuong_dong"] = sim_values
    return result


# ── Gợi ý theo văn bản nhập tay (TF-IDF fallback cho real-time search) ───────
@st.cache_resource
def _build_tfidf(texts: tuple):
    """Fit TF-IDF vectorizer trên toàn bộ tập dữ liệu (cached)."""
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=8000, sublinear_tf=True)
    matrix = vectorizer.fit_transform(list(texts))
    return vectorizer, matrix


def _detect_district(query: str) -> str | None:
    """Phát hiện quận từ query nhập tay."""
    q = query.lower()
    district_aliases = {
        "Bình Thạnh": ["bình thạnh", "binh thanh", "bt"],
        "Gò Vấp":     ["gò vấp", "go vap", "gv"],
        "Phú Nhuận":  ["phú nhuận", "phu nhuan", "pn"],
    }
    for district, aliases in district_aliases.items():
        for alias in aliases:
            if alias in q:
                return district
    return None


def _detect_house_traits(query: str) -> list[str]:
    """Phát hiện tính chất nhà từ query: mặt tiền, hẻm, tầng, gần chợ, v.v."""
    q = query.lower()
    trait_keywords = [
        "mặt tiền", "mat tien", "mt",
        "hẻm", "hem", "hẻm xe hơi", "hẻm xe tải", "hxt",
        "gần chợ", "gan cho", "chợ",
        "gần trường", "gan truong", "trường học", "truong hoc",
        "tầng", "tang", "lầu", "lau",
        "sổ hồng", "so hong", "pháp lý",
        "kinh doanh", "buôn bán",
        "nội thất", "noi that",
    ]
    return [kw for kw in trait_keywords if kw in q]


def get_recommendations_by_text(df: pd.DataFrame, query: str, nums: int = 5) -> pd.DataFrame:
    """Tìm `nums` căn nhà tương đồng nhất với đoạn văn bản `query` do người dùng nhập.
    Thứ tự ưu tiên: Quận (40%) → Tính chất nhà (30%) → TF-IDF ngữ nghĩa (30%)."""
    title_col = next((c for c in ["tieu_de", "title", "name"] if c in df.columns), df.columns[1])
    addr_col  = next((c for c in ["dia_chi", "address"] if c in df.columns), None)
    quan_col  = "Quan" if "Quan" in df.columns else None
    combined  = (df[title_col].fillna("") + " " + df.get("mo_ta", pd.Series([""] * len(df))).fillna("")).tolist()

    n = len(df)

    # --- Phần 1: Quận matching (ưu tiên cao nhất — 40%) ---
    district_scores = np.zeros(n)
    detected_district = _detect_district(query)
    if detected_district:
        for i in range(n):
            match = False
            if quan_col and str(df.iloc[i].get(quan_col, "")).strip() == detected_district:
                match = True
            elif addr_col and detected_district.lower() in str(df.iloc[i].get(addr_col, "")).lower():
                match = True
            elif detected_district.lower() in combined[i].lower():
                match = True
            district_scores[i] = 1.0 if match else 0.0

    # --- Phần 2: Tính chất nhà matching (30%) ---
    trait_scores = np.zeros(n)
    detected_traits = _detect_house_traits(query)
    if detected_traits:
        for i in range(n):
            text_lower = combined[i].lower()
            # Thêm địa chỉ vào text để match
            if addr_col:
                text_lower += " " + str(df.iloc[i].get(addr_col, "")).lower()
            matched = sum(1 for t in detected_traits if t in text_lower)
            trait_scores[i] = matched / len(detected_traits)

    # --- Phần 3: TF-IDF cosine similarity (ngữ nghĩa — 30%) ---
    vectorizer, tfidf_matrix = _build_tfidf(tuple(combined))
    query_vec = vectorizer.transform([query])
    tfidf_scores = cosine_similarity(query_vec, tfidf_matrix)[0]

    # --- Kết hợp: Quận (40%) + Tính chất (30%) + TF-IDF (30%) ---
    final_scores = 0.4 * district_scores + 0.3 * trait_scores + 0.3 * tfidf_scores

    top_idx = final_scores.argsort()[::-1][:nums]
    result = df.iloc[top_idx].copy()
    result["do_tuong_dong"] = [round(float(final_scores[i]) * 100, 1) for i in top_idx]
    return result


# ── Mô hình phân cụm (giống y Notebook 02: 10 features, IQR, 3 model) ───────
class ClusterModelManager:
    # 10 features giống Notebook 02
    FEATURES = [
        "gia_ban_trieu", "dien_tich_m2", "price_per_m2", "width",
        "n_bedroom", "n_toilet", "house_type", "legal_status", "quan", "phuong",
    ]

    @staticmethod
    @st.cache_resource
    def load_model():
        """Train KMeans + GMM + Agglomerative (k=2) từ nhatot_clustered_final.csv.
        Giống pipeline Notebook 02: 10 features, StandardScaler, K=2."""
        df = DataLoaderP2.load_clustered()
        feat_cols = [c for c in ClusterModelManager.FEATURES if c in df.columns]
        X = df[feat_cols].dropna()

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 3 model giống Notebook 02
        km  = KMeans(n_clusters=2, random_state=42, n_init=10).fit(X_scaled)
        gmm = GaussianMixture(n_components=2, random_state=42).fit(X_scaled)
        agg = AgglomerativeClustering(n_clusters=2).fit(X_scaled)

        pca = PCA(n_components=2, random_state=42).fit(X_scaled)

        return km, gmm, agg, scaler, pca, feat_cols

    @staticmethod
    def predict_segment(km, scaler, feat_cols: list, input_dict: dict) -> dict:
        """Dự đoán phân khúc cho 1 căn nhà mới."""
        X = pd.DataFrame([{c: input_dict.get(c, 0) for c in feat_cols}])
        X_scaled = scaler.transform(X)
        cluster = int(km.predict(X_scaled)[0])
        # Cluster 0 = cheaper (phổ thông), Cluster 1 = expensive (cao cấp) based on centroids
        c0_price = km.cluster_centers_[0][0]
        c1_price = km.cluster_centers_[1][0]
        if c0_price < c1_price:
            segment = "Phổ thông" if cluster == 0 else "Cao cấp"
        else:
            segment = "Cao cấp" if cluster == 0 else "Phổ thông"
        distances = km.transform(X_scaled)[0]
        confidence = round((1 - distances.min() / (distances.sum() + 1e-9)) * 100, 1)
        return {"cluster": cluster, "segment": segment, "confidence": confidence}
