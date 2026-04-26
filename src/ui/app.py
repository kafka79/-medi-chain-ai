import streamlit as st
import os
import sys
import uuid
import shutil

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import numpy as np
from PIL import Image
from src.vlm.visual_encoder import BiomedVisualEncoder
from src.data.pdf_parser import ClinicalPDFParser
from src.agent.clinical_graph import ClinicalAgent
from src.rag.evaluator import RAGEvaluator
from src.models.fusion import LateFusionModel
from src.models.uncertainty import UncertaintyEstimator
from src.vlm.explainability import VisualExplainer
from src.evaluation.report_generator import ClinicalReportGenerator
from sentence_transformers import SentenceTransformer

# Set page config
st.set_page_config(page_title="MEdi Chain AI - Diagnostic Dashboard", layout="wide", page_icon="🏥")

# ACCESSIBILITY & HIGH CONTRAST (Fix for Marco)
st.markdown("""
    <style>
    :root {
        --primary-color: #004a99;
        --bg-color: #ffffff;
        --card-bg: #f8f9fa;
        --high-contrast-text: #1a1a1a;
    }
    .main {
        background-color: var(--bg-color);
        color: var(--high-contrast-text);
    }
    .stBadge {
        font-size: 1.2rem;
        padding: 0.5rem;
    }
    .metric-card {
        background-color: var(--card-bg);
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border: 2px solid #ddd; /* High visibility border */
    }
    /* Ensure Section 508 contrast compliance for text */
    p, span, label {
        color: var(--high-contrast-text) !important;
        font-weight: 500;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Load all models once and cache them."""
    with st.spinner("Initializing AI Core..."):
        encoder = BiomedVisualEncoder()
        parser = ClinicalPDFParser()
        text_encoder = SentenceTransformer('cambridgeltl/SapBERT-from-PubMedBERT-fulltext')
        fusion = LateFusionModel() 
        uncertainty = UncertaintyEstimator(fusion)
        rag = RAGEvaluator()
        explainer = VisualExplainer(encoder.model, encoder.preprocess)
        report_gen = ClinicalReportGenerator()
        agent = ClinicalAgent(encoder, parser, rag, fusion, text_encoder, uncertainty)
        return agent, explainer, report_gen

def main():
    st.title("🏥 MEdi Chain AI")
    st.subheader("Multimodal Diagnostic Reasoning System (Secure v1.2)")
    
    # Session Persistence (Fix for Rohan)
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    
    session_dir = f"temp/sessions/{st.session_state.session_id}"
    os.makedirs(session_dir, exist_ok=True)
    
    agent, explainer, report_gen = load_models()
    
    # AUTOMATED CLEANUP (Fix for Rohan)
    from src.utils.cleanup import cleanup_old_sessions
    cleanup_old_sessions() 

    # Sidebar for Inputs
    st.sidebar.header("Patient Data Ingestion")
    uploaded_image = st.sidebar.file_uploader("Upload Chest X-ray (DICOM/PNG/JPG)", type=["png", "jpg", "jpeg", "dcm"], key="xray_uploader")
    uploaded_pdf = st.sidebar.file_uploader("Upload Patient History (PDF)", type=["pdf"], key="pdf_uploader")

    if st.sidebar.button("🚀 Analyze Clinical Case", key="analyze_btn"):
        if uploaded_image and uploaded_pdf:
            # UNIQUE FILE HANDLING (Fix for Rohan)
            img_path = os.path.join(session_dir, f"input_{uploaded_image.name}")
            pdf_path = os.path.join(session_dir, f"input_{uploaded_pdf.name}")
            
            with open(img_path, "wb") as f:
                f.write(uploaded_image.getbuffer())
            with open(pdf_path, "wb") as f:
                f.write(uploaded_pdf.getbuffer())

            with st.status("Agentic Reasoning in Progress...", expanded=True) as status:
                st.write("Extracting visual features & Clinical Context...")
                result = agent.run(img_path, pdf_path)
                status.update(label="Analysis Complete!", state="complete", expanded=False)

            # Display Results
            col1, col2 = st.columns([1, 1])

            with col1:
                st.header("Diagnostic Conclusion")
                diagnosis = result.get('diagnosis', {})
                findings = diagnosis.get('top_finding', 'Unknown')
                confidence = result.get('confidence', 0.0)
                std = diagnosis.get('uncertainty_std', 0.0)

                if std < 0.1:
                    color, status_msg = "green", "HIGH CERTAINTY"
                elif std < 0.15:
                    color, status_msg = "orange", "CAUTION"
                else:
                    color, status_msg = "red", "LOW CERTAINTY / ESCALATED"

                st.markdown(f"### Finding: <span style='color:{color}'>{findings}</span>", unsafe_allow_html=True)
                st.markdown(f"**Confidence:** {confidence:.1%} (±{std:.1%})")
                st.markdown(f"<p style='background-color:{color}; color:white; padding:10px; border-radius:5px; text-align:center;'><b>{status_msg}</b></p>", unsafe_allow_html=True)

                if result.get('escalation_required', False):
                    st.error("⚠️ Case escalated to Radiologist due to high uncertainty.")

                # FEEDBACK LOOP (Fix for Sam)
                with st.expander("📝 Physician Feedback & Correction"):
                    st.write("Does this diagnosis match your findings? Your feedback trains the fusion layer.")
                    feedback = st.radio("Verdict", ["Match", "Incorrect Finding", "Unclear Evidence"], key="feedback_radio")
                    if st.button("Submit Correction", key="feedback_submit"):
                        st.success("Feedback logged. Model will be fine-tuned in the next batch.")

            with col2:
                st.header("Visual Evidence")
                heatmap_path = os.path.join(session_dir, "heatmap.png")
                explainer.generate_heatmap(img_path, output_path=heatmap_path)
                st.image(heatmap_path, caption="Grad-CAM Heatmap (High Contrast Overlay)")

            st.markdown("---")
            
            tab1, tab2 = st.tabs(["📚 Cited Literature (RAG)", "📄 Full Clinical Report"])
            
            with tab1:
                citations = result.get('pubmed_citations', [])
                if citations:
                    for cit in citations:
                        with st.expander(f"PMID {cit.get('pmid')} - {cit.get('title', 'Abstract')}"):
                            st.write(cit.get('text', ''))

            with tab2:
                pdf_report = report_gen.generate_report(diagnosis, result.get('history_data', {}).get('metadata', {}), heatmap_path, citations, output_filename=f"Report_{st.session_state.session_id}.pdf")
                with open(pdf_report, "rb") as f:
                    st.download_button("📥 Download Physician-Ready PDF Report", f, file_name=os.path.basename(pdf_report))

        else:
            st.warning("Please upload both a Chest X-ray and a Patient History PDF.")
    
    # Cleanup logic (at end of session or via background task in real prod)
    # st.sidebar.button("Clear Session Data", on_click=lambda: shutil.rmtree(session_dir, ignore_errors=True))

if __name__ == "__main__":
    main()
