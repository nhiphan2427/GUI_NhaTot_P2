import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pages.base import BasePage


class P2BusinessPage(BasePage):
    def render(self) -> None:
        st.title("📋 Business Problem — Project 2")
        self._render_context()
        st.markdown("---")
        self._render_objectives()
        st.markdown("---")
        self._render_data_description()
        st.markdown("---")
        self._render_methodology()

    # ── 1. Bối cảnh ──────────────────────────────────────────────────────────
    def _render_context(self):
        st.subheader("🏙️ Bối cảnh & Đặt vấn đề")
        col1, col2 = st.columns([2, 1], gap="large")

        with col1:
            st.markdown("""
            Trên nền tảng **NhaTot.vn**, hàng chục nghìn tin đăng bất động sản
            được tạo mới mỗi ngày. Người mua nhà phải tự lọc thủ công qua khối lượng
            khổng lồ thông tin, gây mất thời gian và dễ bỏ lỡ căn nhà phù hợp.

            Đồng thời, từ góc nhìn kinh doanh, sàn bất động sản cần **hiểu cấu trúc
            thị trường** để đưa ra chiến lược giá, tiếp thị và tư vấn đầu tư
            phù hợp theo từng phân khúc.

            **=> Hai bài toán cần giải quyết:**
            - **Bài toán 1:** Gợi ý nhà phù hợp dựa trên nội dung mô tả và đặc trưng
              (Hybrid Content-based Recommender System)
            - **Bài toán 2:** Phân cụm thị trường theo đặc trưng nhà
              (K-Means / GMM — Scikit-learn & PySpark)
            """)

        with col2:
            fig, ax = plt.subplots(figsize=(4, 3))
            segments = ["Phổ thông", "Trung cấp", "Cao cấp"]
            counts   = [38, 39, 23]
            colors   = ["#2563EB", "#10B981", "#E8512A"]
            wedges, texts, autotexts = ax.pie(
                counts, labels=segments, colors=colors,
                autopct="%1.0f%%", startangle=90,
                textprops={"fontsize": 9}
            )
            ax.set_title("Phân bố phân khúc thị trường\n(PySpark K=3)", fontsize=10, fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    # ── 2. Mục tiêu ──────────────────────────────────────────────────────────
    def _render_objectives(self):
        st.subheader("🎯 Mục tiêu dự án")
        col1, col2 = st.columns(2, gap="large")
        with col1:
            st.markdown("""
            **Bài toán 1 — Hệ thống gợi ý (Recommender System)**

            | # | Mục tiêu |
            |---|---------|
            | 1 | Tiền xử lý NLP văn bản mô tả tiếng Việt (5 bộ từ điển) |
            | 2 | So sánh 4 model: TF-IDF, BM25, FastText, **SBERT** |
            | 3 | Xây dựng Hybrid: SBERT (40%) + Giá (30%) + Vị trí (30%) |
            | 4 | Gợi ý Top-N căn nhà tương tự cho người dùng |
            """)
        with col2:
            st.markdown("""
            **Bài toán 2 — Phân cụm thị trường (Market Segmentation)**

            | # | Mục tiêu |
            |---|---------|
            | 1 | Scikit-learn: K-Means, GMM, Agglomerative (K=2) |
            | 2 | PySpark MLlib: K-Means, Bisecting K-Means, GMM (K=3) |
            | 3 | Đánh giá bằng Silhouette, CH Score, DB Index |
            | 4 | Trực quan hóa cụm qua PCA 2D |
            """)

    # ── 3. Dữ liệu ───────────────────────────────────────────────────────────
    def _render_data_description(self):
        st.subheader("🗂️ Mô tả dữ liệu")
        col1, col2 = st.columns([1, 1], gap="large")
        with col1:
            st.markdown("**Nguồn & Phạm vi**")
            info_df = pd.DataFrame({
                "Thông tin": ["Nguồn", "Phạm vi", "Số quận", "Tổng tin đăng (sau sạch)"],
                "Giá trị": ["NhaTot.vn", "TP. Hồ Chí Minh", "3 (BT · GV · PN)", "~7.939 records"],
            })
            st.dataframe(info_df, hide_index=True, use_container_width=True)
        with col2:
            st.markdown("**Đặc trưng dùng cho Project 2**")
            feat_df = pd.DataFrame({
                "Tên cột": ["tieu_de", "mo_ta", "cleaned_content", "gia_ban_trieu",
                             "dien_tich_m2", "price_per_m2", "cluster_km", "PCA1/PCA2"],
                "Mô tả":  ["Tiêu đề tin đăng", "Mô tả chi tiết (raw)", "Văn bản đã tiền xử lý NLP",
                            "Giá bán (triệu đồng)", "Diện tích (m²)", "Giá/m² (triệu)",
                            "Nhãn cụm K-Means", "Tọa độ PCA 2 chiều"],
            })
            st.dataframe(feat_df, hide_index=True, use_container_width=True)

        st.markdown("**Quy trình xử lý dữ liệu**")
        steps = [
            ("Thu thập", "Crawl NhaTot.vn (3 quận)"),
            ("NLP Cleaning", "Tách từ, bỏ stopword, chuẩn hóa tiếng Việt"),
            ("TF-IDF", "Vector hóa văn bản mô tả"),
            ("Cosine Sim", "Ma trận độ tương đồng (Hybrid)"),
            ("K-Means/GMM", "Phân cụm đặc trưng số"),
        ]
        cols = st.columns(len(steps))
        for i, (step, desc) in enumerate(steps):
            with cols[i]:
                st.markdown(f"""
                <div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:10px;
                            padding:12px;text-align:center;height:90px">
                    <div style="font-size:11px;font-weight:700;color:#E8512A;
                                text-transform:uppercase;letter-spacing:.5px">{step}</div>
                    <div style="font-size:12px;color:#475569;margin-top:6px;
                                line-height:1.4">{desc}</div>
                </div>
                """, unsafe_allow_html=True)

    # ── 4. Phương pháp ───────────────────────────────────────────────────────
    def _render_methodology(self):
        st.subheader("🔬 Phương pháp tiếp cận")
        col1, col2 = st.columns(2, gap="large")
        with col1:
            st.markdown("**Hệ thống gợi ý — Hybrid Content-based**")
            st.markdown("""
            | Thành phần | Mô tả | Trọng số |
            |------------|-------|---------|
            | SBERT (NLP) | Ngữ nghĩa mô tả (Sentence-BERT) | 40% |
            | Giá tiền | Khoảng cách giá bán (MinMaxScaler) | 30% |
            | Vị trí | Cùng quận/phường (bonus score) | 30% |

            **Output:** Top-N căn nhà tương tự nhất (Hybrid Score)
            """)
        with col2:
            st.markdown("**Phân cụm — Clustering Ensemble**")
            st.markdown("""
            | Thuật toán | Môi trường | K tối ưu |
            |-----------|-----------|---------|
            | K-Means ⭐ | Scikit-learn | 2 cụm |
            | GMM | Scikit-learn | 2 cụm |
            | Agglomerative | Scikit-learn | 2 cụm |
            | K-Means ⭐ | PySpark MLlib | 3 cụm |
            | Bisecting K-Means | PySpark MLlib | 3 cụm |
            | GMM | PySpark MLlib | 3 cụm |

            **Đánh giá:** Silhouette · CH Score · DB Index
            """)
