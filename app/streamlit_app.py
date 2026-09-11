"""
AI-Based Document Intelligence Platform — Streamlit Dashboard
SparkIIT ML with Python Project

Run with:
    streamlit run app/streamlit_app.py
"""

import sys
import time
import uuid
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.ocr.extractor import DocumentExtractor
from src.nlp.classifier import DocumentClassifier
from src.nlp.ner import EntityExtractor
from src.nlp.summarizer import DocumentSummarizer
from src.search.semantic_search import SemanticSearchEngine
from src.utils.preprocessor import clean_text, extract_keywords

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="📄 Document Intelligence Platform",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .info-card {
        background: linear-gradient(135deg, #1e293b, #0f172a);
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 20px 28px;
        margin: 12px 0 20px 0;
    }
    .result-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 18px;
        margin: 10px 0;
    }
    .entity-tag {
        display: inline-block;
        background: #3b82f6;
        color: white;
        border-radius: 8px;
        padding: 3px 10px;
        margin: 3px;
        font-size: 0.85em;
    }
    .section-header {
        color: #94a3b8;
        font-size: 0.85em;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ── Model Loading (cached) ─────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_extractor():
    return DocumentExtractor()

@st.cache_resource(show_spinner=False)
def load_classifier():
    return DocumentClassifier()

@st.cache_resource(show_spinner=False)
def load_ner():
    return EntityExtractor()

@st.cache_resource(show_spinner=False)
def load_summarizer():
    return DocumentSummarizer()

@st.cache_resource(show_spinner=False)
def load_search():
    return SemanticSearchEngine()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Features")
    do_classify  = st.toggle("📂 Document Classification", value=True)
    do_ner       = st.toggle("🧠 Named Entity Recognition", value=True)
    do_summarize = st.toggle("✍️ Text Summarization", value=True)
    do_search    = st.toggle("🔎 Add to Search Index", value=True)
    st.divider()
    st.markdown("### 📋 About")
    st.markdown("""
    This platform processes uploaded documents and:
    - Extracts text (PDF / image / scanned)
    - Classifies document type
    - Extracts named entities
    - Generates AI summaries
    - Indexes for semantic search

    **Models used:**
    - BART-large-MNLI (classification)
    - T5-small (summarization)
    - spaCy en_core_web_sm (NER)
    - all-MiniLM-L6-v2 (search)
    """)

# ── Header / Landing Page ──────────────────────────────────────────────────────
st.markdown("# 📄 AI-Based Document Intelligence Platform")
st.markdown("### Intelligent document processing with NLP, OCR & Semantic Search")

st.markdown("""
<div class="info-card">
    <table style="width:100%; border-collapse:collapse; color:#e2e8f0;">
        <tr>
            <td style="padding:5px 0; font-size:1.05em; width:200px;">
                📌 <strong style="color:#94a3b8;">Project Topic</strong>
            </td>
            <td style="padding:5px 0; font-size:1.05em; color:#f1f5f9;">
                AI-Based Document Intelligence Platform
            </td>
        </tr>
        <tr>
            <td style="padding:5px 0; font-size:1.05em;">
                👤 <strong style="color:#94a3b8;">Full Name</strong>
            </td>
            <td style="padding:5px 0; font-size:1.05em; color:#f1f5f9;">
                Varad Sachin Kale
            </td>
        </tr>
        <tr>
            <td style="padding:5px 0; font-size:1.05em;">
                📧 <strong style="color:#94a3b8;">Registered Email</strong>
            </td>
            <td style="padding:5px 0; font-size:1.05em; color:#f1f5f9;">
                varadk1120@gmail.com
            </td>
        </tr>
        <tr>
            <td style="padding:5px 0; font-size:1.05em;">
                🏫 <strong style="color:#94a3b8;">Program</strong>
            </td>
            <td style="padding:5px 0; font-size:1.05em; color:#f1f5f9;">
                SparkIIT — ML with Python
            </td>
        </tr>
    </table>
</div>
""", unsafe_allow_html=True)

st.divider()

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["📤 Process Document", "🔎 Semantic Search"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Document Processing
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("## 📤 Upload a Document")
    uploaded_file = st.file_uploader(
        "Upload PDF or Image",
        type=["pdf", "png", "jpg", "jpeg", "tiff"],
        help="Supports digital PDFs, scanned PDFs, and images"
    )

    if uploaded_file:
        file_bytes = uploaded_file.read()
        filename   = uploaded_file.name
        file_size  = len(file_bytes) / 1024

        st.success(f"✅ Uploaded: **{filename}** ({file_size:.1f} KB)")

        # ── Step 1: Text Extraction ────────────────────────────────────────────
        with st.spinner("🔍 Extracting text..."):
            extractor = load_extractor()
            extraction = extractor.extract_from_bytes(file_bytes, filename)
            raw_text = extraction["text"]
            clean   = clean_text(raw_text)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("📄 Pages", extraction["n_pages"])
        col2.metric("📝 Words", f"{extraction['word_count']:,}")
        col3.metric("🔤 Characters", f"{extraction['char_count']:,}")
        col4.metric("⚙️ Method", extraction["method"].replace("_", " ").title())

        with st.expander("📜 View Extracted Text", expanded=False):
            st.text_area("Raw Text", clean[:3000] + ("..." if len(clean) > 3000 else ""),
                         height=250, disabled=True)

        st.divider()

        # ── Step 2: Classification ─────────────────────────────────────────────
        if do_classify:
            with st.spinner("🏷️ Classifying document..."):
                classifier   = load_classifier()
                class_result = classifier.classify(clean)

            col_a, col_b = st.columns([1, 2])
            with col_a:
                st.markdown("### 🏷️ Document Type")
                st.markdown(f"""
                <div class="result-card" style="text-align:center;">
                    <h2 style="color:#60a5fa; margin:0">{class_result['predicted_label']}</h2>
                    <p style="color:#94a3b8; margin:4px 0">
                        Confidence: <strong style="color:#f1f5f9">
                        {class_result['confidence']*100:.1f}%</strong>
                    </p>
                </div>
                """, unsafe_allow_html=True)

            with col_b:
                st.markdown("### 📊 Classification Scores")
                scores_df = pd.DataFrame(
                    [(k, round(v * 100, 2)) for k, v in
                     sorted(class_result["all_scores"].items(), key=lambda x: x[1], reverse=True)],
                    columns=["Document Type", "Confidence (%)"]
                )
                fig = px.bar(scores_df, x="Confidence (%)", y="Document Type",
                             orientation="h", color="Confidence (%)",
                             color_continuous_scale="Blues",
                             template="plotly_dark")
                fig.update_layout(height=280, margin=dict(l=0, r=0, t=20, b=0),
                                  showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

            st.divider()

        # ── Step 3: NER ────────────────────────────────────────────────────────
        if do_ner:
            with st.spinner("🧠 Extracting entities..."):
                ner_extractor = load_ner()
                ner_result    = ner_extractor.extract(clean)

            st.markdown("### 🧠 Named Entities")

            if ner_result["total"] == 0:
                st.info("No named entities detected in this document.")
            else:
                stats = ner_extractor.get_summary_stats(ner_result)
                stats_df = pd.DataFrame(stats)

                col_ner1, col_ner2 = st.columns([1, 2])
                with col_ner1:
                    st.markdown(f"**{ner_result['total']} entities found**")
                    st.dataframe(stats_df, use_container_width=True, hide_index=True)

                with col_ner2:
                    for label, entities in list(ner_result["grouped"].items())[:6]:
                        icon = entities[0]["icon"]
                        desc = entities[0]["desc"]
                        st.markdown(f"**{icon} {label}** — *{desc}*")
                        tags_html = " ".join(
                            f'<span class="entity-tag">{e["text"]}</span>'
                            for e in entities[:8]
                        )
                        st.markdown(tags_html, unsafe_allow_html=True)

            st.divider()

        # ── Step 4: Summarization ──────────────────────────────────────────────
        if do_summarize:
            with st.spinner("✍️ Generating summary..."):
                summarizer   = load_summarizer()
                summary_result = summarizer.summarize(clean)

            st.markdown("### ✍️ AI-Generated Summary")
            st.markdown(f"""
            <div class="result-card">
                <p style="color:#e2e8f0; font-size:1.05em; line-height:1.7; margin:0">
                    {summary_result['summary']}
                </p>
            </div>
            """, unsafe_allow_html=True)

            s_col1, s_col2, s_col3 = st.columns(3)
            s_col1.metric("Original Words", f"{summary_result['original_length']:,}")
            s_col2.metric("Summary Words", summary_result['summary_length'])
            s_col3.metric("Compression", f"{(1-summary_result['compression_ratio'])*100:.0f}%")

            st.divider()

        # ── Step 5: Keywords & Index ───────────────────────────────────────────
        keywords = extract_keywords(clean, top_n=10)
        st.markdown("### 🔑 Key Terms")
        kw_html = " ".join(
            f'<span style="background:#1e3a5f; color:#93c5fd; border-radius:8px; '
            f'padding:4px 12px; margin:3px; display:inline-block">{kw}</span>'
            for kw in keywords
        )
        st.markdown(kw_html, unsafe_allow_html=True)

        if do_search:
            st.divider()
            with st.spinner("📥 Adding to search index..."):
                search_engine = load_search()
                doc_id = str(uuid.uuid4())[:8]
                meta = {
                    "filename": filename,
                    "doc_type": class_result.get("predicted_label", "Unknown") if do_classify else "Unknown",
                    "word_count": extraction["word_count"],
                    "keywords": ", ".join(keywords),
                }
                search_engine.add_document(doc_id, clean, meta)

            st.success(f"✅ Document indexed! Total in search: **{search_engine.get_index_size()}** docs")

    else:
        st.info("👆 Upload a PDF or image document to start intelligent processing.")
        st.markdown("""
        ### 💡 What this platform does:
        | Step | Feature | Description |
        |------|---------|-------------|
        | 1️⃣ | **OCR Extraction** | Extract text from PDFs and scanned images |
        | 2️⃣ | **Classification** | Identify document type (Invoice, Resume, etc.) |
        | 3️⃣ | **NER** | Extract people, orgs, dates, locations |
        | 4️⃣ | **Summarization** | Generate AI summary in seconds |
        | 5️⃣ | **Semantic Search** | Index for natural language search |
        """)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Semantic Search
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("## 🔎 Semantic Document Search")
    st.markdown("Search across all indexed documents using natural language.")

    search_engine = load_search()
    index_size = search_engine.get_index_size()

    if index_size == 0:
        st.warning("⚠️ No documents indexed yet. Upload and process documents in the **Process Document** tab first.")
    else:
        st.info(f"📚 **{index_size} document(s)** in the search index")

        query = st.text_input(
            "Enter your search query",
            placeholder="e.g. 'employment terms and salary', 'patient diagnosis report'..."
        )
        # Slider requires max > min; show fixed value when only 1 doc indexed
        if index_size > 1:
            top_k = st.slider("Number of results", 1, min(10, index_size), min(5, index_size))
        else:
            top_k = 1
            st.info("Only 1 document indexed — returning top result.")

        if query:
            with st.spinner("🔍 Searching..."):
                results = search_engine.search(query, top_k=top_k)

            if not results:
                st.warning("No results found.")
            else:
                st.markdown(f"### 📋 Top {len(results)} Results")
                for r in results:
                    with st.expander(
                        f"**#{r['rank']}** {r.get('filename', r['id'])} "
                        f"— Score: {r['similarity_score']:.3f} "
                        f"| Type: {r.get('doc_type', 'Unknown')}"
                    ):
                        st.markdown(f"**Snippet:** {r.get('text_snippet', '')[:400]}...")
                        cols = st.columns(3)
                        cols[0].metric("Similarity", r['similarity_score'])
                        cols[1].metric("Words", r.get('word_count', 'N/A'))
                        cols[2].metric("Keywords", r.get('keywords', 'N/A')[:30])

        if st.button("🗑️ Clear Search Index", type="secondary"):
            search_engine.clear_index()
            st.success("Index cleared!")
            st.rerun()
