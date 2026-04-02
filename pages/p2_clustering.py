import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from pages.base import BasePage, fmt_price
from data_loader_p2 import DataLoaderP2
from models_p2 import ClusterModelManager


SEGMENT_COLORS = {
    "Phổ thông": "#2563EB",
    "Cao cấp":   "#E8512A",
    "Trung cấp": "#10B981",
    "Phân khúc Phổ thông (Nhà hẻm/Giá rẻ)": "#2563EB",
    "Phân khúc Cao cấp (Diện tích lớn/Mặt tiền)": "#E8512A",
    "Phân khúc Phổ thông (Nhà hẻm nhỏ)": "#2563EB",
    "Phân khúc Trung cấp (Nhà phố/Hẻm lớn)": "#10B981",
    "Phân khúc Trung cấp (Nhà phố/Hẻm xe tải)": "#10B981",
    "Phân khúc Cao cấp (Mặt tiền/Kinh doanh)": "#E8512A",
    0: "#2563EB",
    1: "#E8512A",
    2: "#10B981",
}


class P2ClusteringPage(BasePage):
    def render(self) -> None:
        st.title("📊 Phân cụm Thị trường Bất động sản")
        st.markdown(
            '<div class="info-box">🔬 <b>Notebook 02</b>: K-Means + GMM + Agglomerative (Scikit-learn, k=2) · '
            "<b>Notebook 03</b>: K-Means + Bisecting K-Means + GMM (PySpark MLlib, k=3) · "
            "Đánh giá bằng Silhouette · CH Score · DB Index</div>",
            unsafe_allow_html=True,
        )

        with st.spinner("⏳ Đang tải dữ liệu phân cụm..."):
            df = DataLoaderP2.load_clustered()
            km, gmm, agg, scaler, pca, feat_cols = ClusterModelManager.load_model()

        tab_vis, tab_stats, tab_spark, tab_predict = st.tabs([
            "📈 Trực quan hóa cụm (Sklearn)",
            "📋 Kết quả 3 Model (Notebook 02)",
            "⚡ PySpark Big Data (Notebook 03)",
            "🔍 Dự đoán phân khúc",
        ])

        with tab_vis:
            self._render_visualization(df, km, gmm, agg, scaler, pca, feat_cols)

        with tab_stats:
            self._render_notebook02_results(df)

        with tab_spark:
            self._render_notebook03_results()

        with tab_predict:
            self._render_prediction(km, scaler, feat_cols)

    # ── Tab 1: Trực quan hóa PCA — 3 model (giống Notebook 02) ──────────────
    def _render_visualization(self, df, km, gmm, agg, scaler, pca, feat_cols):
        st.subheader("Biểu đồ phân cụm (PCA 2D) — 3 thuật toán Scikit-learn")

        # Scatter Plot: Diện tích vs Giá tiền (giống Notebook 02 cell 28)
        st.markdown('<div class="section-title">Phân cụm K-Means: Diện tích vs Giá tiền</div>',
                    unsafe_allow_html=True)
        if "dien_tich_m2" in df.columns and "gia_ban_trieu" in df.columns:
            use_col = "cluster_km" if "cluster_km" in df.columns else None
            seg_col = "segment_name" if "segment_name" in df.columns else None

            fig, ax = plt.subplots(figsize=(8, 5))
            if use_col:
                for c in sorted(df[use_col].dropna().unique()):
                    mask = df[use_col] == c
                    label = df.loc[mask, seg_col].iloc[0] if seg_col else f"Cụm {int(c)}"
                    color = SEGMENT_COLORS.get(label, SEGMENT_COLORS.get(c, "#94A3B8"))
                    ax.scatter(
                        df.loc[mask, "dien_tich_m2"], df.loc[mask, "gia_ban_trieu"],
                        c=color, alpha=0.55, s=18, label=label, edgecolors="none"
                    )
            ax.set_xlabel("Diện tích (m²)", fontsize=10)
            ax.set_ylabel("Giá bán (Triệu VNĐ)", fontsize=10)
            ax.set_title("PHÂN CỤM K-MEANS: DIỆN TÍCH VS GIÁ TIỀN", fontsize=12, fontweight="bold")
            ax.legend(fontsize=8)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        # PCA 2D — 3 models side by side
        st.markdown('<div class="section-title">Trực quan hóa PCA 2D — 3 thuật toán</div>',
                    unsafe_allow_html=True)

        seg_col = "segment_name" if "segment_name" in df.columns else None

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        for ax, (col_label, title) in zip(
            axes,
            [("cluster_km", "K-Means (k=2) ⭐"),
             ("cluster_gmm", "GMM (k=2)"),
             ("cluster_km", "Agglomerative (k=2)")]
        ):
            if col_label not in df.columns or "PCA1" not in df.columns:
                ax.text(0.5, 0.5, "Không có dữ liệu PCA", ha="center", va="center")
                continue

            clusters = sorted(df[col_label].dropna().unique())
            for c in clusters:
                mask = df[col_label] == c
                seg = None
                if seg_col:
                    segs = df.loc[mask, seg_col].unique()
                    seg = segs[0] if len(segs) > 0 else None
                color = SEGMENT_COLORS.get(seg, SEGMENT_COLORS.get(c, "#94A3B8"))
                label = seg if seg else f"Cụm {int(c)}"
                ax.scatter(
                    df.loc[mask, "PCA1"], df.loc[mask, "PCA2"],
                    c=color, alpha=0.55, s=18, label=label, edgecolors="none"
                )

            ax.set_title(title, fontsize=11, fontweight="bold")
            ax.set_xlabel("PCA 1", fontsize=9)
            ax.set_ylabel("PCA 2", fontsize=9)
            ax.legend(fontsize=7, markerscale=1.2)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(labelsize=8)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown("""
        <div class="info-box">
            📌 <b>Nhận xét (Notebook 02):</b> Phân khúc Phổ thông (xanh) và Cao cấp (đỏ)
            tách nhau dứt khoát theo trục PCA 1. K-Means và Agglomerative cho kết quả
            gần tương đồng (70/30 vs 73/27), khẳng định cấu trúc phân hóa rõ ràng.
        </div>
        """, unsafe_allow_html=True)

    # ── Tab 2: Kết quả 3 Model Notebook 02 ──────────────────────────────────
    def _render_notebook02_results(self, df):
        st.subheader("Kết quả phân cụm — Notebook 02 (Scikit-learn, K=2)")

        # Bảng tổng kết 3 model (số liệu từ Notebook 02)
        st.markdown("#### K-Means (K=2) ⭐")
        km_df = pd.DataFrame({
            "Tên Phân Khúc": [
                "Phân khúc Phổ thông (Nhà hẻm/Giá rẻ)",
                "Phân khúc Cao cấp (Diện tích lớn/Mặt tiền)",
            ],
            "Số lượng": [5324, 2252],
            "Giá TB (Tỷ)": [5.54, 10.91],
            "Giá Min (Tỷ)": [0.5, 4.8],
            "Giá Max (Tỷ)": [14.5, 21.0],
            "Diện tích TB (m²)": [43.1, 69.2],
            "Đơn giá TB (tr/m²)": [139.3, 169.5],
        })
        st.dataframe(km_df, hide_index=True, use_container_width=True)

        st.markdown("#### GMM (K=2)")
        gmm_df = pd.DataFrame({
            "Tên Phân Khúc": [
                "Phân khúc Phổ thông (Nhà hẻm/Giá rẻ)",
                "Phân khúc Cao cấp (Diện tích lớn/Mặt tiền)",
            ],
            "Số lượng": [6470, 1106],
            "Giá TB (Tỷ)": [6.66, 9.89],
            "Giá Min (Tỷ)": [0.50, 1.45],
            "Giá Max (Tỷ)": [21.0, 21.0],
            "Diện tích TB (m²)": [49.2, 60.6],
            "Đơn giá TB (tr/m²)": [143.5, 175.9],
        })
        st.dataframe(gmm_df, hide_index=True, use_container_width=True)

        st.markdown("#### Agglomerative (K=2)")
        agg_df = pd.DataFrame({
            "Tên Phân Khúc": [
                "Phân khúc Phổ thông (Nhà hẻm/Giá rẻ)",
                "Phân khúc Cao cấp (Diện tích lớn/Mặt tiền)",
            ],
            "Số lượng": [5531, 2045],
            "Giá TB (Tỷ)": [5.93, 10.38],
            "Giá Min (Tỷ)": [0.50, 1.45],
            "Giá Max (Tỷ)": [20.0, 21.0],
            "Diện tích TB (m²)": [44.2, 69.0],
            "Đơn giá TB (tr/m²)": [143.8, 160.2],
        })
        st.dataframe(agg_df, hide_index=True, use_container_width=True)

        # Bảng so sánh 3 model (giống Notebook 02 cell 24-25)
        st.markdown("#### So sánh 3 thuật toán")
        compare_df = pd.DataFrame({
            "Tiêu chí": [
                "Tỷ lệ Phân bổ",
                "Độ vọt về Giá",
                "Độ vọt Diện tích",
                "Đánh giá chung",
            ],
            "K-Means ⭐": [
                "70% / 30%",
                "Tăng ~2.0 lần (5.5 tỷ → 10.9 tỷ)",
                "43m² → 69m²",
                "Nhà vô địch về độ tách biệt",
            ],
            "GMM": [
                "85% / 15%",
                "Tăng ~1.5 lần (6.6 tỷ → 9.9 tỷ)",
                "49m² → 61m²",
                "An toàn quá mức, cụm Cao cấp bị loãng",
            ],
            "Agglomerative": [
                "73% / 27%",
                "Tăng ~1.75 lần (5.9 tỷ → 10.3 tỷ)",
                "44m² → 69m²",
                "Bám sát K-Means, validation tốt",
            ],
        })
        st.dataframe(compare_df, hide_index=True, use_container_width=True)

        st.markdown("""
        <div class="info-box">
            🏆 <b>Kết luận Notebook 02:</b> K-Means cho ranh giới "sắc" nhất (giá gấp 2 lần),
            Agglomerative bám sát là validation tốt. GMM gom 85% vào Phổ thông → cụm Cao cấp bị loãng.
            <b>Chọn K-Means (K=2)</b> cho Scikit-learn.
        </div>
        """, unsafe_allow_html=True)

        # KPI cards per segment
        st.markdown("#### Thống kê theo phân khúc (từ dữ liệu)")
        seg_col = "segment_name" if "segment_name" in df.columns else "cluster_km"
        numeric = ["gia_ban_trieu", "dien_tich_m2", "price_per_m2", "width", "n_bedroom", "n_toilet"]
        avail   = [c for c in numeric if c in df.columns]

        for seg in df[seg_col].unique():
            seg_data = df[df[seg_col] == seg]
            color = SEGMENT_COLORS.get(str(seg), "#2563EB")
            n = len(seg_data)

            price_m = seg_data["gia_ban_trieu"].mean() / 1000 if "gia_ban_trieu" in seg_data else 0
            area_m  = seg_data["dien_tich_m2"].mean() if "dien_tich_m2" in seg_data else 0
            pm2_m   = seg_data["price_per_m2"].mean() if "price_per_m2" in seg_data else 0
            bed_m   = seg_data["n_bedroom"].mean() if "n_bedroom" in seg_data else 0

            pct = round(n / len(df) * 100, 1)
            st.markdown(f"""
            <div style="background:white;border:1px solid #E8EEF6;border-radius:16px;
                        padding:20px 24px;margin-bottom:16px;
                        border-left:5px solid {color};
                        box-shadow:0 2px 10px rgba(0,0,0,0.05)">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px">
                    <div style="font-size:18px;font-weight:800;color:#1E293B">
                        <span style="color:{color}">{seg}</span>
                    </div>
                    <div style="background:#F1F5F9;padding:4px 14px;border-radius:20px;
                                font-size:13px;font-weight:600;color:#475569">
                        {n:,} căn ({pct}%)
                    </div>
                </div>
                <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px">
                    <div style="text-align:center">
                        <div style="font-size:11px;color:#94A3B8;text-transform:uppercase;
                                    letter-spacing:.5px">Giá TB</div>
                        <div style="font-size:22px;font-weight:800;color:{color}">{price_m:.2f} tỷ</div>
                    </div>
                    <div style="text-align:center">
                        <div style="font-size:11px;color:#94A3B8;text-transform:uppercase;
                                    letter-spacing:.5px">Diện tích TB</div>
                        <div style="font-size:22px;font-weight:800;color:#1E293B">{area_m:.0f} m²</div>
                    </div>
                    <div style="text-align:center">
                        <div style="font-size:11px;color:#94A3B8;text-transform:uppercase;
                                    letter-spacing:.5px">Giá/m² TB</div>
                        <div style="font-size:22px;font-weight:800;color:#1E293B">{pm2_m:.0f} tr</div>
                    </div>
                    <div style="text-align:center">
                        <div style="font-size:11px;color:#94A3B8;text-transform:uppercase;
                                    letter-spacing:.5px">Phòng ngủ TB</div>
                        <div style="font-size:22px;font-weight:800;color:#1E293B">{bed_m:.1f}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Price distribution chart
        st.markdown('<div class="section-title">Phân phối giá theo phân khúc</div>', unsafe_allow_html=True)
        if "gia_ban_trieu" in df.columns and seg_col in df.columns:
            fig, ax = plt.subplots(figsize=(8, 4))
            for seg in df[seg_col].unique():
                seg_data = df[df[seg_col] == seg]["gia_ban_trieu"].clip(upper=30000)
                color = SEGMENT_COLORS.get(str(seg), "#94A3B8")
                ax.hist(seg_data / 1000, bins=40, alpha=0.65, color=color,
                        label=str(seg), edgecolor="none")
            ax.set_xlabel("Giá bán (tỷ đồng)", fontsize=10)
            ax.set_ylabel("Số lượng tin đăng", fontsize=10)
            ax.set_title("Phân phối giá bán theo phân khúc", fontsize=11, fontweight="bold")
            ax.legend(fontsize=8)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    # ── Tab 3: PySpark Big Data (Notebook 03) ────────────────────────────────
    def _render_notebook03_results(self):
        st.subheader("Kết quả phân cụm — Notebook 03 (PySpark MLlib, K=3)")

        st.markdown("""
        Notebook 03 triển khai trên môi trường **PySpark** (Spark MLlib).
        Với khả năng xử lý sâu hơn, PySpark nhận diện được **3 phân khúc** rõ nét
        (Phổ thông, Trung cấp, Cao cấp) mà môi trường local bỏ sót.
        """)

        # Bảng đánh giá K tối ưu (từ Notebook 03 cell 13)
        st.markdown("#### Đánh giá chọn K tối ưu (PySpark)")
        k_eval_df = pd.DataFrame({
            "K": [2, 3, 4, 5],
            "WCSS (Inertia) ↓": [33667.38, 26247.10, 21752.26, 18603.60],
            "Silhouette ↑": [0.341, 0.339, 0.227, 0.236],
            "CH Score ↑": [2650.68, 2770.29, 2749.77, 2731.39],
            "DB Index ↓": [1.358, 0.906, 1.029, 1.027],
        })
        st.dataframe(
            k_eval_df.style
                .highlight_max(subset=["Silhouette ↑", "CH Score ↑"], color="#EBF7F0")
                .highlight_min(subset=["DB Index ↓", "WCSS (Inertia) ↓"], color="#EBF7F0"),
            hide_index=True, use_container_width=True,
        )

        st.markdown("""
        <div class="info-box">
            📌 <b>Chọn K=3:</b> CH Score đạt đỉnh (~2770), DB Index đạt đáy thấp nhất (0.906).
            Silhouette chênh lệch rất nhỏ giữa K=2 (0.341) và K=3 (0.339).
        </div>
        """, unsafe_allow_html=True)

        # Kết quả 3 model PySpark (từ Notebook 03 cell 20)
        st.markdown("#### K-Means (PySpark, K=3) ⭐")
        km_spark_df = pd.DataFrame({
            "Tên Phân Khúc": [
                "Phân khúc Phổ thông (Nhà hẻm nhỏ)",
                "Phân khúc Trung cấp (Nhà phố/Hẻm lớn)",
                "Phân khúc Cao cấp (Mặt tiền/Kinh doanh)",
            ],
            "Số lượng": [2846, 2987, 1743],
            "Giá TB (Tỷ)": [5.35, 6.27, 11.52],
            "Giá Min (Tỷ)": [0.7, 0.5, 4.8],
            "Giá Max (Tỷ)": [14.8, 14.5, 21.0],
            "Diện tích TB (m²)": [30.5, 57.9, 72.2],
            "Đơn giá TB (tr/m²)": [176.4, 109.9, 168.1],
        })
        st.dataframe(km_spark_df, hide_index=True, use_container_width=True)

        st.markdown("#### Bisecting K-Means (PySpark, K=3)")
        bkm_spark_df = pd.DataFrame({
            "Tên Phân Khúc": [
                "Phân khúc Phổ thông (Nhà hẻm nhỏ)",
                "Phân khúc Trung cấp (Nhà phố/Hẻm lớn)",
                "Phân khúc Cao cấp (Mặt tiền/Kinh doanh)",
            ],
            "Số lượng": [2355, 2921, 2300],
            "Giá TB (Tỷ)": [5.30, 5.72, 10.81],
            "Giá Min (Tỷ)": [1.1, 0.5, 4.0],
            "Giá Max (Tỷ)": [13.2, 11.9, 21.0],
            "Diện tích TB (m²)": [29.5, 52.5, 70.7],
            "Đơn giá TB (tr/m²)": [180.9, 110.4, 162.9],
        })
        st.dataframe(bkm_spark_df, hide_index=True, use_container_width=True)

        st.markdown("#### GMM (PySpark, K=3)")
        gmm_spark_df = pd.DataFrame({
            "Tên Phân Khúc": [
                "Phân khúc Phổ thông (Nhà hẻm nhỏ)",
                "Phân khúc Trung cấp (Nhà phố/Hẻm lớn)",
                "Phân khúc Cao cấp (Mặt tiền/Kinh doanh)",
            ],
            "Số lượng": [3000, 2941, 1635],
            "Giá TB (Tỷ)": [5.52, 6.35, 11.51],
            "Giá Min (Tỷ)": [0.82, 0.89, 0.50],
            "Giá Max (Tỷ)": [12.50, 10.95, 21.00],
            "Diện tích TB (m²)": [32.6, 56.7, 73.9],
            "Đơn giá TB (tr/m²)": [168.9, 112.6, 174.4],
        })
        st.dataframe(gmm_spark_df, hide_index=True, use_container_width=True)

        # So sánh 3 model PySpark (giống Notebook 03 cell 21-23)
        st.markdown("#### So sánh 3 thuật toán PySpark")
        compare_spark_df = pd.DataFrame({
            "Tiêu chí": [
                "Độ tách biệt Diện tích",
                "Phân bổ dải Giá",
                "Số lượng nhà",
                "Đánh giá chung",
            ],
            "K-Means ⭐": [
                "Rất tốt (30.5 → 57.9 → 72.2 m²)",
                "Rõ rệt: phân tách rõ nhóm dưới 6 tỷ và trên 11 tỷ",
                "Cân bằng giữa các cụm",
                "Mô hình tối ưu nhất cho thực tế thị trường",
            ],
            "Bisecting K-Means": [
                "Khá (29.5 → 52.5 → 70.7 m²)",
                "Chồng lấn: Phổ thông (5.3 tỷ) vs Trung cấp (5.7 tỷ) chỉ chênh 400tr",
                "Cụm Phổ thông hơi ít nhà",
                "Phù hợp nếu muốn chia nhỏ từ cụm lớn",
            ],
            "GMM": [
                "Tốt (32.6 → 56.7 → 73.9 m²)",
                "Hợp lý nhưng nhóm Phổ thông diện tích hơi lớn",
                "Tập trung nhiều vào cụm Phổ thông",
                "Phù hợp nếu dữ liệu có tính xác suất cao",
            ],
        })
        st.dataframe(compare_spark_df, hide_index=True, use_container_width=True)

        st.markdown("""
        <div class="info-box">
            🏆 <b>Kết luận Notebook 03:</b> K-Means (PySpark) cho "bước nhảy" diện tích dứt khoát:
            Nhà hẻm nhỏ (30m²) → Nhà phố hẻm xe tải (58m²) → Nhà mặt tiền (72m²).
            Bisecting K-Means chồng lấn về giá, GMM gộp diện tích lớn vào Phổ thông.
            <b>Chọn K-Means (K=3)</b> cho PySpark.
        </div>
        """, unsafe_allow_html=True)

        # So sánh Sklearn vs PySpark
        st.markdown("#### So sánh tổng hợp: Sklearn (K=2) vs PySpark (K=3)")
        overall_df = pd.DataFrame({
            "Môi trường": ["Scikit-learn", "Scikit-learn", "Scikit-learn",
                           "PySpark MLlib", "PySpark MLlib", "PySpark MLlib"],
            "Thuật toán": ["K-Means (k=2) ⭐", "GMM (k=2)", "Agglomerative (k=2)",
                           "K-Means (k=3) ⭐", "Bisecting K-Means (k=3)", "GMM (k=3)"],
            "Phân khúc": ["Phổ thông / Cao cấp", "Phổ thông / Cao cấp", "Phổ thông / Cao cấp",
                          "Phổ thông / Trung cấp / Cao cấp", "Phổ thông / Trung cấp / Cao cấp",
                          "Phổ thông / Trung cấp / Cao cấp"],
            "Nhận xét": ["Ranh giới sắc nhất (giá gấp 2 lần)", "Cụm Cao cấp bị loãng (85/15)",
                         "Bám sát K-Means, validation tốt",
                         "Phát hiện Trung cấp, bước nhảy diện tích dứt khoát",
                         "Phổ thông vs Trung cấp chồng lấn giá",
                         "GMM gộp diện tích lớn vào Phổ thông"],
        })
        st.dataframe(overall_df, hide_index=True, use_container_width=True)

    # ── Tab 4: Dự đoán phân khúc mới ─────────────────────────────────────────
    def _render_prediction(self, km, scaler, feat_cols: list):
        st.subheader("🔍 Dự đoán phân khúc cho căn nhà mới")
        st.markdown(
            '<div class="info-box">Nhập đặc trưng của căn nhà để mô hình K-Means '
            "dự đoán phân khúc thị trường phù hợp.</div>",
            unsafe_allow_html=True,
        )

        col_form, col_result = st.columns([1, 1], gap="large")

        with col_form:
            st.markdown("##### Thông tin căn nhà")
            gia_ty    = st.slider("Giá bán (tỷ đồng)", 0.5, 50.0, 5.0, step=0.5)
            dien_tich = st.number_input("Diện tích (m²)", min_value=1.0, max_value=500.0, value=45.0, step=0.1, format="%.1f")
            width     = st.slider("Chiều ngang (m)", 2.0, 20.0, 4.0, step=0.5)
            col_a, col_b = st.columns(2)
            with col_a:
                n_bed  = st.number_input("Phòng ngủ", 1, 10, 3)
            with col_b:
                n_wc   = st.number_input("Phòng tắm", 0, 6, 2)

            # Thêm input cho categorical features (giống Notebook 02)
            col_c, col_d = st.columns(2)
            with col_c:
                house_type = st.selectbox("Loại nhà", options=[0, 1, 2],
                                          format_func=lambda x: {0: "Mặt tiền", 1: "Hẻm", 2: "Khác"}[x])
                legal_status = st.selectbox("Pháp lý", options=[0, 1, 2, 3],
                                            format_func=lambda x: {0: "Chưa xác định", 1: "Đã có sổ", 2: "Sổ chung", 3: "Khác"}[x])
            with col_d:
                quan = st.selectbox("Quận", options=[0, 1, 2],
                                    format_func=lambda x: {0: "Bình Thạnh", 1: "Gò Vấp", 2: "Phú Nhuận"}[x])
                phuong = st.number_input("Phường (mã số)", 0, 30, 5)

            submitted = st.button("🔍 Dự đoán phân khúc", use_container_width=True)

        with col_result:
            st.markdown("""
            <div class="result-page-header">
                <h3 style="margin:0;color:#1A2744">Kết quả phân khúc</h3>
                <span class="model-badge">K-Means (k=2)</span>
            </div>
            """, unsafe_allow_html=True)

            if submitted:
                gia_trieu  = gia_ty * 1000
                price_m2   = gia_trieu / dien_tich if dien_tich > 0 else 0
                input_dict = {
                    "gia_ban_trieu": gia_trieu,
                    "dien_tich_m2":  dien_tich,
                    "price_per_m2":  price_m2,
                    "width":         width,
                    "n_bedroom":     int(n_bed),
                    "n_toilet":      int(n_wc),
                    "house_type":    int(house_type),
                    "legal_status":  int(legal_status),
                    "quan":          int(quan),
                    "phuong":        int(phuong),
                }
                result = ClusterModelManager.predict_segment(km, scaler, feat_cols, input_dict)

                seg     = result["segment"]
                conf    = result["confidence"]
                color   = SEGMENT_COLORS.get(seg, "#2563EB")
                icon    = "🏠" if seg == "Phổ thông" else "🏰"

                # Score card
                st.markdown(f"""
                <div class="score-card" style="text-align:center">
                    <div style="font-size:48px">{icon}</div>
                    <div style="font-size:26px;font-weight:800;color:{color};margin:12px 0">
                        {seg}
                    </div>
                    <div style="font-size:13px;color:#64748B;margin-bottom:16px">
                        Phân khúc dự đoán bởi K-Means (k=2)
                    </div>
                    <div style="background:#F1F5F9;border-radius:10px;padding:12px">
                        <div style="font-size:12px;color:#94A3B8;text-transform:uppercase;
                                    letter-spacing:.5px">Độ tin cậy</div>
                        <div style="font-size:30px;font-weight:800;color:{color}">{conf:.0f}%</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Input summary
                st.markdown('<div class="section-label" style="margin-top:16px">Thông tin đã nhập</div>',
                            unsafe_allow_html=True)
                summary_df = pd.DataFrame({
                    "Đặc trưng": ["Giá bán", "Diện tích", "Giá/m²", "Chiều ngang",
                                  "Phòng ngủ", "Phòng tắm", "Loại nhà", "Pháp lý", "Quận", "Phường"],
                    "Giá trị":  [
                        f"{gia_ty:.1f} tỷ",
                        f"{dien_tich} m²",
                        f"{price_m2:.0f} triệu/m²",
                        f"{width} m",
                        f"{int(n_bed)}",
                        f"{int(n_wc)}",
                        {0: "Mặt tiền", 1: "Hẻm", 2: "Khác"}[int(house_type)],
                        {0: "Chưa xác định", 1: "Đã có sổ", 2: "Sổ chung", 3: "Khác"}[int(legal_status)],
                        {0: "Bình Thạnh", 1: "Gò Vấp", 2: "Phú Nhuận"}[int(quan)],
                        f"{int(phuong)}",
                    ],
                })
                st.dataframe(summary_df, hide_index=True, use_container_width=True)

            else:
                st.markdown("""
                <div style="text-align:center;padding:80px 20px;color:#ccc">
                    <div style="font-size:52px">🔬</div>
                    <div style="font-size:15px;font-weight:500;color:#999;margin-top:14px">
                        Điền thông tin căn nhà bên trái
                    </div>
                    <div style="font-size:13px;margin-top:6px;color:#bbb">
                        và nhấn <b>Dự đoán phân khúc</b>
                    </div>
                </div>
                """, unsafe_allow_html=True)
