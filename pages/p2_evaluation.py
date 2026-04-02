import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pages.base import BasePage


class P2EvaluationPage(BasePage):
    def render(self) -> None:
        st.title("📊 Evaluation & Report — Project 2")
        st.markdown(
            '<div class="info-box">📌 Đánh giá toàn diện 2 bài toán: '
            "<b>Hệ thống Gợi ý</b> (SBERT Hybrid) và <b>Phân cụm thị trường</b> "
            "(Sklearn K=2 + PySpark K=3).</div>",
            unsafe_allow_html=True,
        )

        tab1, tab2, tab3 = st.tabs([
            "🏠 Recommender System",
            "📊 Clustering",
            "📝 Tổng kết",
        ])

        with tab1:
            self._render_recommender_eval()
        with tab2:
            self._render_clustering_eval()
        with tab3:
            self._render_summary()

    # ── Recommender Evaluation (giống Notebook 01) ───────────────────────────
    def _render_recommender_eval(self):
        st.subheader("Đánh giá Hệ thống Gợi ý (SBERT Hybrid 40-30-30)")

        # So sánh 4 model (giống Notebook 01 cell 39-44)
        st.markdown("#### So sánh 4 Model NLP")
        model_compare = pd.DataFrame({
            "Thuật toán": ["TF-IDF", "BM25", "FastText (Gensim)", "SBERT ⭐"],
            "Loại": ["Sparse (từ khóa)", "Sparse (ranking)", "Dense (word)", "Dense (sentence)"],
            "Sai số Giá (%) ↓": ["~35%", "30.97%", "~40%", "30.70%"],
            "District Match ↑": ["Trung bình", "Khá", "Thấp", "Cao nhất"],
            "Nhận xét": [
                "Baseline tốt, chỉ khớp từ khóa",
                "Thông minh hơn TF-IDF về từ khóa",
                "Bắt quan hệ từ nhưng không hiểu câu",
                "Hiểu ngữ nghĩa toàn câu, cross-lingual",
            ],
        })
        st.dataframe(model_compare, hide_index=True, use_container_width=True)

        st.markdown("""
        <div class="info-box">
            🏆 <b>SBERT</b> cho sai số giá thấp nhất (30.70%) và tỷ lệ gợi ý cùng quận cao nhất.
            Được chọn làm lõi NLP trong mô hình Hybrid: SBERT (40%) + Giá (30%) + Vị trí (30%).
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2, gap="large")
        with col1:
            st.markdown("**3 Metrics đánh giá (Notebook 01)**")
            metrics_df = pd.DataFrame({
                "Metric": [
                    "Price Error (%) ↓",
                    "District Match ↑",
                    "Semantic Relevance ↑",
                ],
                "Ý nghĩa": [
                    "Sai số giá trung bình giữa nhà gợi ý và nhà gốc",
                    "Tỷ lệ gợi ý cùng Quận (vị trí địa lý)",
                    "Độ tương đồng ngữ nghĩa (cosine similarity)",
                ],
                "SBERT Hybrid": [
                    "30.70%",
                    "Cao nhất",
                    "> 0.75",
                ],
            })
            st.dataframe(metrics_df, hide_index=True, use_container_width=True)

        with col2:
            st.markdown("**SBERT vs Hybrid (Notebook 01)**")
            st.markdown("""
            | Tiêu chí | SBERT Only | Hybrid (40-30-30) |
            |----------|-----------|-------------------|
            | Độ thông minh | Tìm nhà mô tả giống nhau | Giống + cùng giá + cùng khu |
            | Trải nghiệm | Có thể gợi ý nhà ở quận khác | Luôn gợi ý nhà gần nhất |
            | Tính ứng dụng | Phù hợp nghiên cứu | Phù hợp thực tế BĐS |
            """)

        st.markdown("**Ví dụ kết quả gợi ý định tính**")
        example_df = pd.DataFrame({
            "Nhà gốc": [
                "Bán nhà Hoàng Hoa Thám, BT, 19m², 3 tầng, 3.9 tỷ",
                "Bán nhà Gò Vấp, 45m², 4PN, mặt tiền, 8.5 tỷ",
            ],
            "Gợi ý Top-1 (SBERT Hybrid)": [
                "Nhà hẻm Bình Thạnh, 22m², 3 tầng, 4.1 tỷ (Sim: 0.83)",
                "Nhà mặt tiền Gò Vấp, 50m², 4PN, 9.0 tỷ (Sim: 0.78)",
            ],
            "Nhận xét": [
                "✅ Cùng quận BT, tương tự diện tích & giá",
                "✅ Cùng quận GV, cùng loại mặt tiền, phân khúc gần",
            ],
        })
        st.dataframe(example_df, hide_index=True, use_container_width=True)

    # ── Clustering Evaluation (số liệu từ Notebook 02 & 03) ─────────────────
    def _render_clustering_eval(self):
        st.subheader("Đánh giá Phân cụm — Sklearn (K=2) & PySpark (K=3)")

        # Bảng metrics K=2→5 (từ Notebook 03 cell 13)
        st.markdown("#### Bảng đánh giá K tối ưu (từ Notebook 03)")
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
                .highlight_min(subset=["DB Index ↓"], color="#EBF7F0"),
            hide_index=True, use_container_width=True,
        )

        col1, col2 = st.columns(2, gap="large")
        with col1:
            st.markdown("**Scikit-learn — Notebook 02 (K=2)**")
            sklearn_df = pd.DataFrame({
                "Thuật toán": ["K-Means (k=2) ⭐", "GMM (k=2)", "Agglomerative (k=2)"],
                "Tỷ lệ": ["70% / 30%", "85% / 15%", "73% / 27%"],
                "Giá TB Phổ thông": ["5.54 tỷ", "6.66 tỷ", "5.93 tỷ"],
                "Giá TB Cao cấp": ["10.91 tỷ", "9.89 tỷ", "10.38 tỷ"],
                "Nhận xét": [
                    "Ranh giới sắc nhất, giá gấp 2 lần",
                    "85% Phổ thông, cụm Cao cấp bị loãng",
                    "Bám sát K-Means, validation tốt",
                ],
            })
            st.dataframe(sklearn_df, hide_index=True, use_container_width=True)

        with col2:
            st.markdown("**PySpark MLlib — Notebook 03 (K=3)**")
            spark_df = pd.DataFrame({
                "Thuật toán": ["K-Means (k=3) ⭐", "Bisecting K-Means (k=3)", "GMM (k=3)"],
                "Phổ thông": ["2846 (5.35 tỷ)", "2355 (5.30 tỷ)", "3000 (5.52 tỷ)"],
                "Trung cấp": ["2987 (6.27 tỷ)", "2921 (5.72 tỷ)", "2941 (6.35 tỷ)"],
                "Cao cấp": ["1743 (11.52 tỷ)", "2300 (10.81 tỷ)", "1635 (11.51 tỷ)"],
                "Nhận xét": [
                    "Bước nhảy diện tích dứt khoát",
                    "Phổ thông & Trung cấp chồng lấn giá",
                    "Gộp diện tích lớn vào Phổ thông",
                ],
            })
            st.dataframe(spark_df, hide_index=True, use_container_width=True)

        # Biểu đồ 4 chỉ số (giống Notebook 03 cell 14)
        st.markdown("#### Biểu đồ 4 chỉ số đánh giá K tối ưu")
        k_vals = [2, 3, 4, 5]
        wcss_vals = [33667.38, 26247.10, 21752.26, 18603.60]
        sil_vals  = [0.341, 0.339, 0.227, 0.236]
        ch_vals   = [2650.68, 2770.29, 2749.77, 2731.39]
        db_vals   = [1.358, 0.906, 1.029, 1.027]

        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        fig.suptitle('ĐÁNH GIÁ TỔNG HỢP 4 CHỈ SỐ XÁC ĐỊNH SỐ CỤM K TỐI ƯU',
                     fontsize=14, fontweight='bold')

        # 1. Elbow
        axes[0, 0].plot(k_vals, wcss_vals, 'o-', color='blue', lw=2)
        axes[0, 0].set_title('1. Elbow Method (WCSS) ↓', fontsize=11)
        axes[0, 0].set_xticks(k_vals)
        axes[0, 0].grid(True, linestyle='--', alpha=0.5)

        # 2. Silhouette
        axes[0, 1].plot(k_vals, sil_vals, 'o-', color='green', lw=2)
        axes[0, 1].set_title('2. Silhouette Score ↑', fontsize=11)
        axes[0, 1].set_xticks(k_vals)
        axes[0, 1].grid(True, linestyle='--', alpha=0.5)

        # 3. CH Score
        axes[1, 0].plot(k_vals, ch_vals, 'o-', color='orange', lw=2)
        axes[1, 0].set_title('3. Calinski-Harabasz Score ↑', fontsize=11)
        axes[1, 0].set_xticks(k_vals)
        axes[1, 0].grid(True, linestyle='--', alpha=0.5)

        # 4. DB Index
        axes[1, 1].plot(k_vals, db_vals, 'o-', color='red', lw=2)
        axes[1, 1].set_title('4. Davies-Bouldin Index ↓', fontsize=11)
        axes[1, 1].set_xticks(k_vals)
        axes[1, 1].grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        st.pyplot(fig)
        plt.close()

        st.markdown("""
        <div class="info-box" style="margin-top:8px">
            🔑 <b>Kết luận:</b><br>
            • Scikit-learn: <b>K=2</b> tối ưu — Phổ thông / Cao cấp. K-Means "nhà vô địch" về độ tách biệt.<br>
            • PySpark: <b>K=3</b> tối ưu — phát hiện thêm phân khúc Trung cấp.
            CH Score đỉnh (2770), DB Index đáy (0.906).<br>
            • Chênh lệch Silhouette giữa K=2 (0.341) và K=3 (0.339) rất nhỏ,
            nhưng K=3 cho DB Index tốt hơn hẳn (0.906 vs 1.358).
        </div>
        """, unsafe_allow_html=True)

    # ── Tổng kết ─────────────────────────────────────────────────────────────
    def _render_summary(self):
        st.subheader("📝 Tổng kết Project 2")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **✅ Kết quả đạt được**

            | Bài toán | Kết quả |
            |---------|--------|
            | Recommender | Hybrid: SBERT (40%) + Price (30%) + Location (30%) |
            | NLP Model | So sánh 4 model → Chọn SBERT (sai số giá 30.70%) |
            | Phân cụm Sklearn | K=2, 3 thuật toán (K-Means ⭐, GMM, Agglomerative) |
            | Phân cụm PySpark | K=3, 3 thuật toán (K-Means ⭐, Bisecting K-Means, GMM) |
            | PySpark K-Means | CH Score = 2770, DB Index = 0.906 |
            | GUI Streamlit | ✅ Hoàn chỉnh, deploy-ready |
            """)

        with col2:
            st.markdown("""
            **📌 Nhận xét & Điểm nhấn**

            | Hạng mục | Nhận xét |
            |---------|---------|
            | Dữ liệu | 3 quận (BT, GV, PN) — ~7,576 records sau lọc IQR |
            | Recommender | SBERT hiểu ngữ nghĩa, không chỉ khớp từ khóa |
            | Sklearn vs PySpark | PySpark phát hiện thêm phân khúc Trung cấp |
            | Agglomerative | Bám sát K-Means → validation cấu trúc phân hóa |
            | GUI | Hoạt động tốt trên local & Streamlit Cloud |
            """)

        st.markdown("**🔮 Hướng phát triển tiếp theo**")
        directions = [
            ("Mở rộng dữ liệu", "Thu thập toàn bộ TP.HCM, cập nhật real-time"),
            ("Deep Learning Rec", "Dùng Sentence-BERT tiếng Việt fine-tuned hoặc Neural CF"),
            ("Real Spark Cluster", "Deploy PySpark trên AWS EMR / Databricks"),
            ("A/B Testing", "Kiểm chứng Recommender với user thực"),
        ]
        cols = st.columns(4)
        for col, (title, desc) in zip(cols, directions):
            with col:
                st.markdown(f"""
                <div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:10px;
                            padding:14px;text-align:center;height:95px">
                    <div style="font-size:11px;font-weight:700;color:#E8512A;
                                text-transform:uppercase;letter-spacing:.5px">{title}</div>
                    <div style="font-size:12px;color:#475569;margin-top:6px;
                                line-height:1.4">{desc}</div>
                </div>
                """, unsafe_allow_html=True)
