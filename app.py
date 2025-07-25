import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import json
import os
import PyPDF2
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
import warnings
from pythainlp.tokenize import word_tokenize, sent_tokenize
from pythainlp.util import normalize
import plotly.graph_objects as go
import plotly.express as px

warnings.filterwarnings('ignore')

# Import our custom modules
from rag_system import ExperimentalThaiTextProcessor, ExperimentalVectorDatabase, ExperimentalCAMTLlamaRAG
from utils import load_css, initialize_session_state

# Page configuration
st.set_page_config(
    page_title="Finddy - Financial Knowledge Assistant",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
load_css()

# Initialize session state
initialize_session_state()

# Sidebar navigation
with st.sidebar:
    st.image("https://via.placeholder.com/150x50/8B7355/FFFFFF?text=Finddy", width=150)
    st.markdown("---")

    page = st.radio(
        "เมนูหลัก",
        ["🔍 ค้นหาความรู้", "📊 แผงควบคุม", "📤 อัปโหลดเอกสาร"],
        index=0
    )

    st.markdown("---")
    st.markdown("### 📊 สถิติการใช้งาน")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("การค้นหาวันนี้", st.session_state.get('searches_today', 0))
    with col2:
        st.metric("เอกสารทั้งหมด", st.session_state.get('total_docs', 0))


# Initialize RAG components
@st.cache_resource
def initialize_rag_system():
    """Initialize the RAG system components"""
    processor = ExperimentalThaiTextProcessor()
    vector_db = ExperimentalVectorDatabase("streamlit_app")

    # Load existing database if available
    if os.path.exists("data/vector_db.json"):
        vector_db.load()

    rag_system = ExperimentalCAMTLlamaRAG(vector_db)
    return processor, vector_db, rag_system


processor, vector_db, rag_system = initialize_rag_system()

# Main content based on selected page
if page == "🔍 ค้นหาความรู้":
    st.title("🔍 Finddy - ระบบค้นหาความรู้ทางการเงิน")
    st.markdown("Financial Knowledge Assistant")

    # Search interface
    col1, col2, col3 = st.columns([3, 1, 1])

    with col1:
        query = st.text_input(
            "พิมพ์คำถามของคุณ",
            placeholder="เช่น วิธีการยืมเงินสดย่อย, วงเงินกู้สูงสุด...",
            key="search_query"
        )

    with col2:
        departments = ["ทุกแผนก", "การเงิน", "ทรัพยากรบุคคล", "ปฏิบัติการ", "ขาย", "ไอที"]
        selected_dept = st.selectbox("แผนก", departments, key="dept_filter")

    with col3:
        doc_types = ["ทุกประเภท", "นโยบาย", "แนวปฏิบัติ", "รายงาน", "คำถามที่พบบ่อย", "ขั้นตอน", "คู่มือ", "แบบฟอร์ม",
                     "ประกาศ"]
        selected_type = st.selectbox("ประเภท", doc_types, key="type_filter")

    # Quick search tags
    st.markdown("### คำค้นหายอดนิยม")
    col1, col2, col3, col4, col5 = st.columns(5)

    quick_searches = [
        ("💰 เงินกู้สวัสดิการ", "เงินกู้สวัสดิการ 50000"),
        ("💵 เงินสดย่อย", "เงินสดย่อย 3000"),
        ("📊 ดอกเบี้ย", "ดอกเบี้ย 0.5%"),
        ("💻 Smart Office", "Smart Office"),
        ("🧾 ค่ารับรอง", "ค่ารับรอง")
    ]

    for col, (label, search_text) in zip([col1, col2, col3, col4, col5], quick_searches):
        with col:
            if st.button(label, use_container_width=True):
                st.session_state['search_query'] = search_text
                st.rerun()

    # Search button
    if st.button("🔍 ค้นหาเอกสาร", type="primary", use_container_width=True):
        if query:
            with st.spinner("กำลังค้นหา..."):
                # Update search count
                st.session_state['searches_today'] = st.session_state.get('searches_today', 0) + 1

                # Get available sources
                available_sources = vector_db.get_available_sources()

                if not available_sources:
                    st.error("ไม่พบเอกสารในระบบ กรุณาอัปโหลดเอกสารก่อน")
                else:
                    # Search across all sources
                    all_results = []
                    for source in available_sources:
                        results = vector_db.search_by_source_with_metadata(
                            query, source, k=3
                        )
                        if results:
                            # Group results by source
                            source_result = {
                                'source': source,
                                'title': results[0]['title'],
                                'department': results[0]['department'],
                                'type': results[0]['type'],
                                'chunks': results,
                                'avg_similarity': np.mean([r['similarity'] for r in results])
                            }
                            all_results.append(source_result)

                    # Sort by average similarity
                    all_results.sort(key=lambda x: x['avg_similarity'], reverse=True)

                    # Apply filters
                    if selected_dept != "ทุกแผนก":
                        all_results = [r for r in all_results if r['department'] == selected_dept]

                    if selected_type != "ทุกประเภท":
                        all_results = [r for r in all_results if r['type'] == selected_type]

                    # Display results
                    if all_results:
                        st.markdown("### 📚 ผลการค้นหา")

                        for idx, result in enumerate(all_results[:5]):  # Show top 5
                            with st.expander(f"{idx + 1}. {result['title']}", expanded=(idx == 0)):
                                col1, col2 = st.columns([3, 1])

                                with col1:
                                    st.markdown(f"**แผนก:** {result['department']}")
                                    st.markdown(f"**ประเภท:** {result['type']}")
                                    st.markdown(f"**ความเชื่อมั่น:** {result['avg_similarity'] * 100:.1f}%")

                                with col2:
                                    if st.button(f"ดูคำตอบ", key=f"view_{idx}"):
                                        # Get answer from RAG
                                        answer_result = rag_system.ask_with_enhanced_retrieval(
                                            query, result['source']
                                        )
                                        st.session_state['current_answer'] = answer_result
                                        st.session_state['show_answer'] = True

                                # Show preview of relevant chunks
                                st.markdown("**ข้อความที่เกี่ยวข้อง:**")
                                for chunk in result['chunks'][:2]:
                                    st.info(chunk['text'][:200] + "...")
                    else:
                        st.warning("ไม่พบเอกสารที่ตรงกับคำค้นหาของคุณ")

    # Display answer if available
    if st.session_state.get('show_answer', False) and 'current_answer' in st.session_state:
        st.markdown("---")
        st.markdown("### 💡 คำตอบ")

        answer_data = st.session_state['current_answer']

        # Answer card
        with st.container():
            st.markdown(f"**คำถาม:** {answer_data['query']}")
            st.markdown(f"**เอกสารอ้างอิง:** {answer_data.get('selected_title', 'ไม่ระบุ')}")

            # Display answer
            st.markdown("**คำตอบ:**")
            st.success(answer_data['answer'])

            # Metadata
            if answer_data.get('metadata'):
                with st.expander("ข้อมูลเพิ่มเติม"):
                    meta = answer_data['metadata']
                    st.markdown(f"- **แผนก:** {meta.get('department', 'ไม่ระบุ')}")
                    st.markdown(f"- **ประเภท:** {meta.get('type', 'ไม่ระบุ')}")
                    st.markdown(f"- **คำสำคัญ:** {', '.join(meta.get('keywords', []))}")

            # Feedback
            col1, col2, col3 = st.columns([1, 1, 3])
            with col1:
                if st.button("👍 มีประโยชน์"):
                    st.session_state['feedback_submitted'] = True
                    st.success("ขอบคุณสำหรับความคิดเห็น!")
            with col2:
                if st.button("👎 ไม่มีประโยชน์"):
                    st.session_state['feedback_submitted'] = True
                    st.info("ขอบคุณสำหรับความคิดเห็น เราจะปรับปรุงให้ดีขึ้น")

elif page == "📊 แผงควบคุม":
    st.title("📊 แผงควบคุม")

    # Dashboard metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "📄 เอกสารทั้งหมด",
            st.session_state.get('total_docs', 0),
            "อัปเดตล่าสุด"
        )

    with col2:
        st.metric(
            "🔍 การค้นหาวันนี้",
            st.session_state.get('searches_today', 0),
            "+12% จากเมื่อวาน"
        )

    with col3:
        st.metric(
            "😊 ความพึงพอใจ",
            "92%",
            "+3% จากเดือนที่แล้ว"
        )

    with col4:
        st.metric(
            "⚡ เวลาตอบสนอง",
            "187ms",
            "-23ms จากเดือนที่แล้ว"
        )

    # Charts
    st.markdown("### 📈 กราฟแสดงผล")

    col1, col2 = st.columns(2)

    with col1:
        # Search trends
        dates = pd.date_range(end=datetime.now(), periods=7).to_list()
        searches = np.random.randint(200, 400, size=7)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates,
            y=searches,
            mode='lines+markers',
            name='การค้นหา',
            line=dict(color='#8B7355', width=3),
            marker=dict(size=8)
        ))
        fig.update_layout(
            title="แนวโน้มการค้นหา 7 วันย้อนหลัง",
            xaxis_title="วันที่",
            yaxis_title="จำนวนการค้นหา",
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Document distribution
        doc_types = ["แนวปฏิบัติ", "นโยบาย", "คู่มือ", "ประกาศ", "อื่นๆ"]
        doc_counts = [7, 5, 4, 3, 2]

        fig = go.Figure(data=[go.Pie(
            labels=doc_types,
            values=doc_counts,
            hole=.3,
            marker_colors=['#8B7355', '#A68B6F', '#C4A57B', '#E2C488', '#F0DBA5']
        )])
        fig.update_layout(
            title="การกระจายประเภทเอกสาร",
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)

    # Document table
    st.markdown("### 📚 รายการเอกสาร")

    # Get documents from vector database
    if hasattr(vector_db, 'data') and vector_db.data:
        # Create document list from vector database
        documents = []
        seen_sources = set()

        for doc in vector_db.data:
            if doc['source'] not in seen_sources:
                seen_sources.add(doc['source'])
                documents.append({
                    'ชื่อเอกสาร': doc['title'],
                    'แผนก': doc['department'],
                    'ประเภท': doc['type'],
                    'วันที่': doc['date'],
                    'ไฟล์': doc['source']
                })

        if documents:
            df = pd.DataFrame(documents)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("ยังไม่มีเอกสารในระบบ")
    else:
        st.info("ยังไม่มีเอกสารในระบบ")

elif page == "📤 อัปโหลดเอกสาร":
    st.title("📤 อัปโหลดเอกสาร")

    # File upload
    uploaded_file = st.file_uploader(
        "เลือกไฟล์ PDF",
        type=['pdf'],
        help="รองรับเฉพาะไฟล์ PDF"
    )

    if uploaded_file:
        # Document metadata form
        with st.form("upload_form"):
            st.markdown("### ข้อมูลเอกสาร")

            col1, col2 = st.columns(2)

            with col1:
                title = st.text_input(
                    "ชื่อเอกสาร",
                    value=uploaded_file.name.replace('.pdf', ''),
                    help="ชื่อเอกสารที่จะแสดงในระบบ"
                )

                department = st.selectbox(
                    "แผนก",
                    ["การเงิน", "ทรัพยากรบุคคล", "ปฏิบัติการ", "ขาย", "ไอที"]
                )

                doc_type = st.selectbox(
                    "ประเภท",
                    ["นโยบาย", "แนวปฏิบัติ", "รายงาน", "คำถามที่พบบ่อย", "ขั้นตอน", "คู่มือ", "แบบฟอร์ม", "ประกาศ"]
                )

            with col2:
                doc_date = st.date_input(
                    "วันที่",
                    value=datetime.now()
                )

                keywords = st.text_input(
                    "คำสำคัญ",
                    placeholder="เช่น เงินกู้, สวัสดิการ, ดอกเบี้ย",
                    help="คั่นด้วยเครื่องหมายจุลภาค"
                )

                doc_number = st.text_input(
                    "เลขที่เอกสาร",
                    placeholder="เช่น อว 1234/2567"
                )

            # Submit button
            submitted = st.form_submit_button("อัปโหลดเอกสาร", type="primary", use_container_width=True)

            if submitted:
                with st.spinner("กำลังประมวลผลเอกสาร..."):
                    try:
                        # Extract text from PDF
                        pdf_reader = PyPDF2.PdfReader(uploaded_file)
                        full_text = ""

                        for page_num in range(len(pdf_reader.pages)):
                            page = pdf_reader.pages[page_num]
                            page_text = page.extract_text()
                            full_text += page_text + "\n"

                        # Normalize text
                        full_text = processor.normalize_thai_text(full_text)

                        # Create document metadata
                        metadata = {
                            'title': title,
                            'department': department,
                            'type': doc_type,
                            'date': doc_date.strftime("%Y-%m-%d"),
                            'keywords': [k.strip() for k in keywords.split(',') if k.strip()],
                            'doc_number': doc_number,
                            'filename': uploaded_file.name
                        }

                        # Chunk text using whitespace method with 300 chunk size
                        chunks = processor.chunk_by_whitespace(
                            full_text,
                            title,
                            chunk_size=300,
                            overlap_ratio=0.2
                        )

                        # Create documents for vector database
                        documents = []
                        for i, chunk_data in enumerate(chunks):
                            doc = {
                                'id': f"{uploaded_file.name}_{i}",
                                'source': uploaded_file.name,
                                'text': chunk_data['text'],
                                'title': metadata['title'],
                                'department': metadata['department'],
                                'type': metadata['type'],
                                'date': metadata['date'],
                                'keywords': metadata['keywords'],
                                'doc_number': metadata['doc_number'],
                                'chunk_index': i,
                                'total_chunks': len(chunks),
                                'chunk_method': 'whitespace',
                                'chunk_size': 300
                            }
                            documents.append(doc)

                        # Embed and add to vector database
                        vector_db.embed_documents(documents)
                        vector_db.save()

                        # Update document count
                        st.session_state['total_docs'] = st.session_state.get('total_docs', 0) + 1

                        st.success(f"✅ อัปโหลดเอกสารสำเร็จ! สร้าง {len(chunks)} chunks")

                        # Clear form
                        st.rerun()

                    except Exception as e:
                        st.error(f"❌ เกิดข้อผิดพลาด: {str(e)}")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>Finddy - Financial Knowledge Assistant © 2024</p>
        <p>Powered by RAG System with Thai Text Processing</p>
    </div>
    """,
    unsafe_allow_html=True
)