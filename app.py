import io

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sentiment_engine import (
    SEGMENT_COLORS,
    analyze_dataframe,
    clean_text,
    classify,
    find_text_column,
    lexicon_score,
    load_data_from_upload,
)
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon=" ",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    .stApp { font-family: 'Inter', sans-serif; }
    .hero {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 18px; padding: 26px 30px; margin-bottom: 18px;
        color: white; box-shadow: 0 8px 24px rgba(102,126,234,.35);
    }
    .hero h1 { margin: 0; font-size: 30px; font-weight: 700; letter-spacing: .5px; }
    .hero p { margin: 6px 0 0; font-size: 15px; opacity: .92; }
    .kpi {
        border-radius: 14px; padding: 18px 20px; color: white;
        box-shadow: 0 6px 18px rgba(0,0,0,.12); text-align: center;
    }
    .kpi .num { font-size: 30px; font-weight: 700; line-height: 1.1; }
    .kpi .lbl { font-size: 14px; opacity: .95; margin-top: 2px; }
    .kpi .sub { font-size: 12px; opacity: .85; margin-top: 4px; }
    .seg-card { border-radius: 12px; padding: 14px 18px; margin: 6px 0;
                border-left: 5px solid; background: #f8f9fc; }
    div[data-testid="stSidebar"] { background: #f4f5fb; }
    .stTabs [data-baseweb="tab-list"] { gap: 6px; }
    .stTabs [data-baseweb="tab"] { border-radius: 8px 8px 0 0; padding: 8px 18px; font-weight: 600; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def kpi_html(label, count, pct, color):
    return (
        f'<div class="kpi" style="background:linear-gradient(135deg,{color},'
        f'{color}bb);">'
        f'<div class="num">{count}</div>'
        f'<div class="lbl">{label}</div>'
        f'<div class="sub">{pct:.1f}% of reviews</div></div>'
    )


def load_source():
    uploaded = st.sidebar.file_uploader(
        "Upload a CSV / Excel file",
        type=["csv", "xlsx", "xls"],
        help="Your file can be about movies, products, services, feedback, tweets, anything.",
    )
    if uploaded is not None:
        try:
            df = load_data_from_upload(uploaded)
            return df, uploaded.name
        except Exception as exc:
            st.sidebar.error(f"Could not read file: {exc}")
            return None, None
    sample = st.sidebar.radio(
        "â€¦or try a built-in sample",
        ["sample_movie_reviews.csv (movies)",
         "sample_product_reviews.xlsx (products / services)"],
    )
    path = "data/" + ("sample_movie_reviews.csv" if sample.startswith("sample_movie") else "sample_product_reviews.xlsx")
    try:
        if path.endswith(".csv"):
            df = pd.read_csv(path)
        else:
            df = pd.read_excel(path)
        return df, path
    except FileNotFoundError:
        st.sidebar.warning("Sample files not found. Please upload a file.")
        return None, None


def live_review_tab():
    st.subheader("Test any single review instantly")
    text = st.text_area(
        "Write a review / sentence below",
        placeholder="e.g. The service was amazing and the staff was very helpful.",
        height=110,
    )
    if st.button("Analyze this text", type="primary"):
        if text.strip():
            analyzer = SentimentIntensityAnalyzer()
            cleaned = clean_text(text)
            compound = analyzer.polarity_scores(cleaned)["compound"]
            if abs(compound) < 0.01 and lexicon_score(cleaned) != 0:
                compound = lexicon_score(cleaned)
            sentiment = classify(compound)
            color = SEGMENT_COLORS[sentiment]
            st.markdown(
                f"""
                <div class="seg-card" style="border-left-color:{color};">
                <b>Segment:</b>
                <span style="color:{color};font-weight:700;text-transform:capitalize;">
                {sentiment}</span>
                &nbsp;|&nbsp; <b>Score:</b> {compound:.3f}
                <div style="font-size:13px;color:#555;margin-top:6px;">{cleaned}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.info("Type something to analyze.")


def main():
    st.markdown(
        """
        <div class="hero">
            <h1>Sentiment Analysis Studio</h1>
            <p>Upload a CSV / Excel of reviews and instantly segment every row into
            <b>positive</b>, <b>negative</b> or <b>neutral</b> â€” works for movies,
            products, services, feedback and any other text.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("## Data source")
        df, source_name = load_source()

        if df is None:
            st.stop()

        st.markdown(f"**File:** `{source_name}`")
        st.markdown(f"**Rows:** {len(df)} &nbsp;|&nbsp; **Columns:** {len(df.columns)}")

        detected = find_text_column(df)
        options = list(df.columns)
        default_idx = options.index(detected) if detected in options else 0
        text_col = st.selectbox(
            "Review / text column",
            options,
            index=default_idx,
            help="Auto-detected. Change it if the wrong column was picked.",
        )
        keep_extra = st.checkbox(
            "Include original (non-review) columns in results",
            value=False,
            help="When off, only the review text + sentiment columns are kept.",
        )
        run = st.button("Run sentiment analysis", type="primary", width='stretch')

    if not run:
        st.info("ðŸ‘ˆ Pick a file (or use a sample) and press **Run sentiment analysis**.")
        st.stop()

    prog = st.progress(0.0, text="Scoring reviewsâ€¦")

    def tick(fraction):
        prog.progress(fraction, text=f"Scoring reviewsâ€¦ {fraction*100:.0f}%")

    out, stats = analyze_dataframe(df, text_col, progress_cb=tick, keep_extra_cols=keep_extra)
    prog.empty()

    count_map = {r["segment"]: int(r["count"]) for _, r in stats.iterrows()}
    pct_map = {r["segment"]: float(r["percentage"]) for _, r in stats.iterrows()}

    col1, col2, col3 = st.columns(3)
    col1.markdown(kpi_html("POSITIVE", count_map["positive"], pct_map["positive"], "#2ecc71"),
                  unsafe_allow_html=True)
    col2.markdown(kpi_html("NEUTRAL", count_map["neutral"], pct_map["neutral"], "#f39c12"),
                  unsafe_allow_html=True)
    col3.markdown(kpi_html("NEGATIVE", count_map["negative"], pct_map["negative"], "#e74c3c"),
                  unsafe_allow_html=True)

    tab_overview, tab_reviews, tab_segments, tab_live = st.tabs(
        ["Overview", "All Reviews", "By Segment", "Single Review"])

    with tab_overview:
        c1, c2 = st.columns([3, 2])
        with c1:
            fig_bar = px.bar(
                stats,
                x="segment",
                y="count",
                color="segment",
                text=stats["percentage"].map(lambda p: f"{p:.1f}%"),
                color_discrete_map=SEGMENT_COLORS,
                category_orders={"segment": ["positive", "neutral", "negative"]},
                title="Review volume per segment",
            )
            fig_bar.update_traces(textposition="outside")
            fig_bar.update_layout(
                showlegend=False, yaxis_title="Number of reviews",
                xaxis_title="", title_font_size=16,
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_bar, width='stretch')
        with c2:
            fig_pie = px.pie(
                stats,
                names="segment",
                values="count",
                color="segment",
                color_discrete_map=SEGMENT_COLORS,
                hole=0.55,
                title="Share of sentiment",
            )
            fig_pie.update_traces(textinfo="label+percent")
            fig_pie.update_layout(showlegend=False, title_font_size=16,
                                  paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_pie, width='stretch')

        st.markdown("### Downloads")
        dl1, dl2, dl3 = st.columns(3)
        csv_bytes = out.to_csv(index=False).encode("utf-8-sig")
        xlsx_buf = io.BytesIO()
        out.to_excel(xlsx_buf, index=False)
        sum_csv = stats.to_csv(index=False).encode("utf-8-sig")
        dl1.download_button("Download results (CSV)", csv_bytes,
                            file_name=f"{source_name.split('.')[0]}_sentiment.csv",
                            mime="text/csv", width='stretch')
        dl2.download_button("Download results (Excel)", xlsx_buf.getvalue(),
                            file_name=f"{source_name.split('.')[0]}_sentiment.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            width='stretch')
        dl3.download_button("Download summary (CSV)", sum_csv,
                            file_name="sentiment_summary.csv", mime="text/csv",
                            width='stretch')

        st.markdown("### Sample reviews per segment")
        s1, s2, s3 = st.columns(3)
        for col, seg, color in (
            (s1, "positive", "#2ecc71"),
            (s2, "negative", "#e74c3c"),
            (s3, "neutral", "#f39c12"),
        ):
            sub = out[out["sentiment"] == seg]
            with col:
                st.markdown(f"**{seg.upper()}** ({len(sub)})")
                preview = sub[text_col].head(4).tolist()
                for i, txt in enumerate(preview):
                    st.markdown(
                        f'<div class="seg-card" style="border-left-color:{color};'
                        f'font-size:13px;">{i+1}. {txt}</div>',
                        unsafe_allow_html=True,
                    )

    with tab_reviews:
        search = st.text_input("Search reviews", placeholder="Type a keyword to filterâ€¦")
        seg_filter = st.multiselect(
            "Filter by segment",
            ["positive", "negative", "neutral"],
            default=["positive", "negative", "neutral"],
        )
        view = out[out["sentiment"].isin(seg_filter)].copy()
        if search.strip():
            mask = view[text_col].astype(str).str.contains(search.strip(), case=False, na=False)
            view = view[mask]
        st.markdown(f"**Showing {len(view)} of {len(out)} reviews**")
        cols = [text_col, "sentiment", "sentiment_score", "cleaned_text"]
        cols = [c for c in cols if c in view.columns] + [c for c in view.columns if c not in cols]
        st.dataframe(view[cols], width='stretch', hide_index=True)

    with tab_segments:
        for seg, color in (("positive", "#2ecc71"), ("neutral", "#f39c12"), ("negative", "#e74c3c")):
            sub = out[out["sentiment"] == seg]
            with st.expander(
                f"{seg.upper()} â€” {len(sub)} reviews ({pct_map[seg]:.1f}%)",
                expanded=(seg == "positive"),
            ):
                if sub.empty:
                    st.write("No reviews in this segment.")
                else:
                    st.dataframe(
                        sub[[text_col, "sentiment_score"]].reset_index(drop=True),
                        width='stretch', hide_index=True,
                    )

    with tab_live:
        live_review_tab()


if __name__ == "__main__":
    main()
