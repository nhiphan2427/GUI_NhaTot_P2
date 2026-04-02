import os
import pickle
import streamlit as st
import pandas as pd
import numpy as np


class DataLoaderP2:
    """Quản lý tải và cache dữ liệu cho Project 2."""

    @staticmethod
    @st.cache_data
    def load_houses() -> pd.DataFrame:
        """Load house_samples.csv dùng cho recommender."""
        paths = [
            "house_samples.csv",
            os.path.join(os.path.dirname(__file__), "house_samples.csv"),
        ]
        for p in paths:
            if os.path.exists(p):
                return pd.read_csv(p, encoding="utf-8-sig", on_bad_lines="skip")
        return DataLoaderP2._demo_houses()

    @staticmethod
    @st.cache_data
    def load_clustered() -> pd.DataFrame:
        """Load nhatot_clustered_final.csv dùng cho phân cụm."""
        paths = [
            "nhatot_clustered_final.csv",
            os.path.join(os.path.dirname(__file__), "nhatot_clustered_final.csv"),
        ]
        for p in paths:
            if os.path.exists(p):
                df = pd.read_csv(p, encoding="utf-8-sig", on_bad_lines="skip")
                # Chuẩn hóa segment_name theo đúng Notebook 02
                if "segment_name" in df.columns:
                    seg_map = {}
                    for s in df["segment_name"].unique():
                        sl = str(s).lower()
                        if "cao" in sl:
                            seg_map[s] = "Phân khúc Cao cấp (Diện tích lớn/Mặt tiền)"
                        else:
                            seg_map[s] = "Phân khúc Phổ thông (Nhà hẻm/Giá rẻ)"
                    df["segment_name"] = df["segment_name"].map(seg_map)
                return df
        return DataLoaderP2._demo_clustered()

    @staticmethod
    @st.cache_resource
    def load_cosine_sim():
        """Load precomputed cosine similarity matrix (SBERT-based)."""
        paths = [
            "nha_cosine_sim_23032026.pkl",
            "nha_cosine_sim.pkl",
            os.path.join(os.path.dirname(__file__), "nha_cosine_sim_23032026.pkl"),
            os.path.join(os.path.dirname(__file__), "nha_cosine_sim.pkl"),
        ]
        for p in paths:
            if os.path.exists(p):
                with open(p, "rb") as f:
                    return pickle.load(f)
        return None

    # ── Demo fallback ─────────────────────────────────────────────────────────
    @staticmethod
    def _demo_houses() -> pd.DataFrame:
        np.random.seed(0)
        n = 10
        return pd.DataFrame({
            "id": range(1, n + 1),
            "tieu_de": [f"Nhà mẫu {i} - Bình Thạnh" for i in range(1, n + 1)],
            "gia_ban": [f"{np.random.randint(3, 15)},{np.random.randint(0, 99):02d} tỷ" for _ in range(n)],
            "dien_tich": [f"{np.random.randint(20, 100)} m²" for _ in range(n)],
            "mo_ta": [f"Nhà mẫu tại Bình Thạnh, thuận tiện đi lại." for _ in range(n)],
            "dia_chi": ["Bình Thạnh, TP.HCM"] * n,
        })

    @staticmethod
    def _demo_clustered() -> pd.DataFrame:
        np.random.seed(1)
        n = 200
        cluster = np.random.choice([0, 1], n)
        return pd.DataFrame({
            "gia_ban_trieu": np.where(cluster == 0,
                                      np.random.normal(4000, 1000, n).clip(500, 8000),
                                      np.random.normal(15000, 4000, n).clip(8000, 50000)),
            "dien_tich_m2": np.random.lognormal(3.8, 0.5, n).clip(10, 400).round(1),
            "price_per_m2": np.random.normal(100, 40, n).clip(20, 400).round(2),
            "width": np.random.choice([3.5, 4.0, 4.5, 5.0, 6.0], n),
            "n_bedroom": np.random.randint(1, 7, n),
            "n_toilet": np.random.randint(1, 5, n),
            "house_type": np.random.choice([0, 1, 2], n),
            "legal_status": np.random.choice([0, 1, 2, 3], n),
            "quan": np.random.choice([0, 1, 2], n),
            "phuong": np.random.choice(range(25), n),
            "cluster_km": cluster,
            "cluster_gmm": cluster,
            "PCA1": np.random.normal(0, 1, n),
            "PCA2": np.random.normal(0, 1, n),
            "segment_name": np.where(cluster == 0,
                                     "Phân khúc Phổ thông (Nhà hẻm/Giá rẻ)",
                                     "Phân khúc Cao cấp (Diện tích lớn/Mặt tiền)"),
        })
