import streamlit as st
import pandas as pd
from pages.base import BasePage
from data_loader_p2 import DataLoaderP2
from models_p2 import get_recommendations, get_recommendations_by_text


class P2RecommenderPage(BasePage):
    def render(self) -> None:
        st.title("🏠 Hệ thống Gợi ý Nhà (Hybrid Model)")
        st.markdown(
            '<div class="info-box">🤖 Mô hình <b>Hybrid Recommender</b>: '
            "So sánh 4 model (TF-IDF, BM25, FastText, <b>SBERT</b>) → Chọn SBERT làm lõi NLP. "
            "Hybrid Score = SBERT (40%) + Giá tiền (30%) + Vị trí (30%).</div>",
            unsafe_allow_html=True,
        )

        with st.spinner("⏳ Đang tải dữ liệu và mô hình..."):
            df_houses  = DataLoaderP2.load_houses()
            cosine_sim = DataLoaderP2.load_cosine_sim()

        if df_houses.empty:
            st.error("Không tải được dữ liệu nhà. Vui lòng kiểm tra file house_samples.csv.")
            return

        self._render_selector(df_houses, cosine_sim)
        st.markdown("---")
        self._render_model_comparison()
        st.markdown("---")
        self._render_model_info()

    # ── Selector + Result ─────────────────────────────────────────────────────
    def _render_selector(self, df: pd.DataFrame, cosine_sim):
        id_col    = "id" if "id" in df.columns else df.columns[0]
        title_col = next((c for c in ["tieu_de", "title", "name"] if c in df.columns), df.columns[1])

        tab_list, tab_manual = st.tabs(["📋 Chọn từ danh sách", "✏️ Nhập mô tả thủ công"])

        # ── Tab 1: Chọn từ danh sách ──────────────────────────────────────────
        with tab_list:
            search_kw = st.text_input(
                "🔍 Lọc nhanh (gõ từ khoá):",
                placeholder="Ví dụ: Bình Thạnh, 4 tầng, hẻm…",
                key="kw_filter",
            )

            if search_kw.strip():
                mask = df[title_col].str.contains(search_kw, case=False, na=False)
                filtered_df = df[mask]
            else:
                filtered_df = df

            house_options = [
                (row[title_col], row[id_col])
                for _, row in filtered_df.iterrows()
            ]

            if not house_options:
                st.warning("Không tìm thấy nhà khớp với từ khoá. Thử từ khoá khác.")
            else:
                st.caption(f"Hiển thị {len(house_options):,} căn nhà{' (đã lọc)' if search_kw.strip() else ''}")
                selected = st.selectbox(
                    "Chọn nhà bạn quan tâm:",
                    options=house_options,
                    format_func=lambda x: x[0],
                    key="house_select",
                )

                if selected:
                    house_id     = selected[1]
                    selected_row = df[df[id_col] == house_id]

                    if selected_row.empty:
                        st.warning(f"Không tìm thấy nhà với ID: {house_id}")
                    else:
                        col_info, col_recs = st.columns([1, 1], gap="large")
                        with col_info:
                            self._render_selected_house(selected_row.iloc[0], title_col)
                        with col_recs:
                            self._render_recommendations(df, house_id, cosine_sim, title_col)

        # ── Tab 2: Nhập mô tả thủ công ────────────────────────────────────────
        with tab_manual:
            st.markdown("Nhập mô tả căn nhà bạn muốn tìm (tiêu đề, khu vực, số tầng, giá…):")
            query_text = st.text_area(
                "Mô tả căn nhà:",
                placeholder="Ví dụ: nhà 4 tầng hẻm xe hơi Bình Thạnh gần trung tâm 5 tỷ",
                height=100,
                key="manual_query",
                label_visibility="collapsed",
            )

            num_results = st.slider("Số căn gợi ý:", min_value=3, max_value=15, value=5, key="num_manual")

            if st.button("🔎 Tìm nhà tương tự", type="primary", key="btn_manual"):
                if not query_text.strip():
                    st.warning("Vui lòng nhập mô tả trước khi tìm kiếm.")
                else:
                    with st.spinner("Đang phân tích và tìm kiếm…"):
                        recs = get_recommendations_by_text(df, query_text.strip(), nums=num_results)

                    if recs.empty:
                        st.info("Không tìm thấy gợi ý phù hợp.")
                    else:
                        st.markdown("""
                        <div class="result-page-header">
                            <h3 style="margin:0;color:#1A2744">Kết quả tìm kiếm</h3>
                            <span class="model-badge">TF-IDF Text Search (fallback)</span>
                        </div>
                        """, unsafe_allow_html=True)
                        self._render_rec_cards(recs, title_col)

    # ── Hiển thị nhà được chọn ───────────────────────────────────────────────
    def _render_selected_house(self, row: pd.Series, title_col: str):
        title = row.get(title_col, "Không có tiêu đề")
        gia   = row.get("gia_ban", row.get("gia_ban_trieu", "N/A"))
        dt    = row.get("dien_tich", row.get("dien_tich_m2", "N/A"))
        addr  = row.get("dia_chi", "")
        mo_ta = str(row.get("mo_ta", ""))
        short = " ".join(mo_ta.split()[:80]) + ("..." if len(mo_ta.split()) > 80 else "")

        st.markdown(f"""
        <div class="price-card" style="margin-bottom:16px">
            <div style="font-size:11px;color:rgba(255,255,255,0.5);
                        text-transform:uppercase;letter-spacing:1px">Bạn vừa chọn</div>
            <div style="font-size:16px;font-weight:700;color:white;
                        margin:8px 0;line-height:1.4">{title}</div>
            <div style="display:flex;gap:16px;margin-top:8px;
                        border-top:1px solid rgba(255,255,255,0.15);padding-top:8px">
                <div>
                    <div style="font-size:10px;color:rgba(255,255,255,0.5)">GIÁ BÁN</div>
                    <div style="font-size:18px;font-weight:800;color:#E8512A">{gia}</div>
                </div>
                <div>
                    <div style="font-size:10px;color:rgba(255,255,255,0.5)">DIỆN TÍCH</div>
                    <div style="font-size:18px;font-weight:800;color:white">{dt}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if addr:
            st.markdown(f"📍 **{addr}**")

        with st.expander("📄 Mô tả chi tiết", expanded=False):
            st.write(short)

    # ── Gợi ý ───────────────────────────────────────────────────────────────
    def _render_recommendations(self, df: pd.DataFrame, house_id, cosine_sim, title_col: str):
        st.markdown("""
        <div class="result-page-header">
            <h3 style="margin:0;color:#1A2744">Căn nhà gợi ý</h3>
            <span class="model-badge">SBERT Hybrid (40-30-30)</span>
        </div>
        """, unsafe_allow_html=True)

        if cosine_sim is None:
            st.warning("⚠ Chưa tải được ma trận cosine similarity. Đang dùng gợi ý ngẫu nhiên.")
            recs = df[df["id"] != house_id].sample(min(5, len(df) - 1)).copy()
            recs["do_tuong_dong"] = 0.0
        else:
            recs = get_recommendations(df, house_id, cosine_sim, nums=5)

        if recs.empty:
            st.info("Không tìm thấy gợi ý phù hợp.")
            return

        self._render_rec_cards(recs, title_col)

    def _render_rec_cards(self, recs: pd.DataFrame, title_col: str):
        for _, rec in recs.iterrows():
            title   = rec.get(title_col, "Không có tiêu đề")
            gia     = rec.get("gia_ban", rec.get("gia_ban_trieu", "N/A"))
            dt      = rec.get("dien_tich", rec.get("dien_tich_m2", "N/A"))
            sim_pct = rec.get("do_tuong_dong", 0)
            mo_ta   = str(rec.get("mo_ta", ""))
            short   = " ".join(mo_ta.split()[:30]) + "..."

            if sim_pct >= 70:
                sim_color, sim_bg = "#059669", "#ECFDF5"
            elif sim_pct >= 40:
                sim_color, sim_bg = "#D97706", "#FFFBEB"
            else:
                sim_color, sim_bg = "#2563EB", "#EFF6FF"

            st.markdown(f"""
            <div style="background:white;border:1px solid #E8EEF6;border-radius:12px;
                        padding:14px 16px;margin-bottom:10px;
                        box-shadow:0 2px 8px rgba(0,0,0,0.05)">
                <div style="display:flex;justify-content:space-between;align-items:flex-start">
                    <div style="font-size:13px;font-weight:700;color:#1E293B;
                                flex:1;line-height:1.4">{title}</div>
                    <div style="background:{sim_bg};color:{sim_color};
                                padding:3px 10px;border-radius:20px;
                                font-size:12px;font-weight:700;white-space:nowrap;
                                margin-left:10px">
                        {sim_pct:.0f}% tương đồng
                    </div>
                </div>
                <div style="display:flex;gap:20px;margin-top:8px">
                    <div style="font-size:13px;color:#E8512A;font-weight:700">{gia}</div>
                    <div style="font-size:13px;color:#64748B">{dt}</div>
                </div>
                <div style="font-size:12px;color:#94A3B8;margin-top:6px;line-height:1.4">{short}</div>
            </div>
            """, unsafe_allow_html=True)

    # ── So sánh 4 Model (giống Notebook 01) ──────────────────────────────────
    def _render_model_comparison(self):
        st.subheader("📊 So sánh hiệu quả 4 Model NLP (Notebook 01)")
        st.markdown("""
        Notebook 01 thực hiện so sánh chéo 4 mô hình Content-based Filtering.
        Kết quả cho thấy **SBERT** vượt trội về khả năng hiểu ngữ nghĩa mô tả nhà.
        """)

        compare_df = pd.DataFrame({
            "Thuật toán": ["TF-IDF (Scikit-learn)", "BM25 (Rank-BM25)", "FastText (Gensim)", "SBERT (Sentence-Transformers) ⭐"],
            "Loại": ["Từ khóa (Sparse)", "Từ khóa (Ranking)", "Word Embedding", "Sentence Embedding"],
            "Ưu điểm": [
                "Nhanh, đơn giản, baseline tốt",
                "Xử lý từ khóa thông minh hơn TF-IDF, có tính tần suất",
                "Bắt được quan hệ ngữ nghĩa giữa từ",
                "Hiểu toàn bộ câu, ngữ cảnh sâu, cross-lingual",
            ],
            "Hạn chế": [
                "Chỉ khớp từ khóa, không hiểu ngữ nghĩa",
                "Vẫn dựa trên từ riêng lẻ",
                "Không hiểu thứ tự từ trong câu",
                "Cần GPU, tốn tài nguyên hơn",
            ],
        })
        st.dataframe(compare_df, hide_index=True, use_container_width=True)

        st.markdown("""
        <div class="info-box">
            🏆 <b>Kết luận:</b> SBERT cho sai số giá thấp nhất (~30.70%) và khả năng gợi ý
            "cùng quận" tốt nhất. Được chọn làm lõi NLP cho mô hình Hybrid.
        </div>
        """, unsafe_allow_html=True)

    # ── Thông tin mô hình Hybrid (giống Notebook 01: 40-30-30) ───────────────
    def _render_model_info(self):
        st.subheader("ℹ️ Cách hoạt động của mô hình Hybrid (40-30-30)")
        st.markdown("""
        Bất động sản là tài sản đặc thù gắn liền với **vị trí** và **giá tiền**.
        Mô hình Hybrid kết hợp ngữ nghĩa từ SBERT với các yếu tố thực tế.
        """)
        col1, col2, col3 = st.columns(3)
        cards = [
            ("🧠 SBERT (40%)", "blue",
             "Sentence-BERT hiểu được ý nghĩa mô tả nhà thay vì chỉ khớp từ khóa. "
             "Encode toàn bộ đoạn mô tả thành vector ngữ nghĩa → Cosine Similarity."),
            ("💰 Giá tiền (30%)", "yellow",
             "So sánh giá bán giữa các căn nhà. Sử dụng MinMaxScaler chuẩn hóa → "
             "tính khoảng cách giá. Ưu tiên gợi ý căn nhà cùng tầm giá."),
            ("📍 Vị trí địa lý (30%)", "red",
             "Bonus score nếu cùng quận/phường. Bất động sản gắn liền với vị trí — "
             "người mua thường tìm nhà trong cùng khu vực."),
        ]
        color_map = {
            "blue":   ("method-card-blue",   "mb-blue"),
            "yellow": ("method-card-yellow", "mb-yellow"),
            "red":    ("method-card-red",    "mb-red"),
        }
        for col, (title, color, desc) in zip([col1, col2, col3], cards):
            card_cls, badge_cls = color_map[color]
            with col:
                st.markdown(f"""
                <div class="method-card {card_cls}">
                    <div class="method-tag">THÀNH PHẦN MÔ HÌNH</div>
                    <div class="method-title">{title}</div>
                    <div class="method-desc">{desc}</div>
                    <span class="method-badge {badge_cls}">Hybrid Model</span>
                </div>
                """, unsafe_allow_html=True)
