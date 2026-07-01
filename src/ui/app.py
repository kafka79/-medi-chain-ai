import os
import sys
import uuid

import streamlit as st
import requests
from src.evaluation.report_generator import ClinicalReportGenerator
from src.utils.feedback_logger import FeedbackLogger
from src.utils.cleanup import cleanup_old_sessions


st.set_page_config(
    page_title="MEdi Chain AI - Diagnostic Dashboard",
    layout="wide",
    page_icon="hospital",
)

st.markdown(
    """
    <style>
    .stBadge {
        font-size: 1.2rem;
        padding: 0.5rem;
    }
    .metric-card {
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border: 2px solid #ddd;
    }
    /* Flaw 7 Fix: Prevent aspect-ratio warping of clinical scans/heatmaps under viewport scaling */
    img {
        object-fit: contain !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_tools():
    """Load lightweight reporting and logging tools."""
    report_gen = ClinicalReportGenerator()
    feedback_logger = FeedbackLogger()
    return report_gen, feedback_logger


@st.cache_resource
def run_startup_cleanup():
    """Run TTL session directory cleanup once on application startup."""
    try:
        cleanup_old_sessions()
    except Exception:
        pass


def main():
    st.title("MEdi Chain AI")
    st.subheader("Multimodal Diagnostic Reasoning System")

    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    session_dir = os.path.join("temp", "sessions", st.session_state.session_id)
    os.makedirs(session_dir, exist_ok=True)

    report_gen, feedback_logger = load_tools()

    run_startup_cleanup()

    st.sidebar.header("Patient Data Ingestion")
    uploaded_image = st.sidebar.file_uploader(
        "Upload Chest X-ray (DICOM/PNG/JPG)",
        type=["png", "jpg", "jpeg", "dcm"],
        key="xray_uploader",
    )
    uploaded_pdf = st.sidebar.file_uploader(
        "Upload Patient History (PDF)",
        type=["pdf"],
        key="pdf_uploader",
    )

    if st.sidebar.button("Analyze Clinical Case", key="analyze_btn"):
        if uploaded_image and uploaded_pdf:
            img_path = os.path.join(session_dir, f"input_{uploaded_image.name}")
            pdf_path = os.path.join(session_dir, f"input_{uploaded_pdf.name}")

            with open(img_path, "wb") as image_handle:
                image_handle.write(uploaded_image.getbuffer())
            with open(pdf_path, "wb") as pdf_handle:
                pdf_handle.write(uploaded_pdf.getbuffer())

            with st.status("Requesting remote analysis...", expanded=True) as status:
                st.write("Sending data to inference cluster...")
                try:
                    # In a real setup, configure API_URL via env var
                    api_url = os.getenv("API_URL", "http://medi-api:8000")
                    headers = {"X-API-Key": os.getenv("API_KEY", "dev-secret-key-123")}
                    
                    with open(img_path, "rb") as img_file, open(pdf_path, "rb") as pdf_file:
                        files = {
                            "image": ("image.jpg", img_file, "image/jpeg"),
                            "history": ("history.pdf", pdf_file, "application/pdf")
                        }
                        response = requests.post(f"{api_url}/analyze", files=files, headers=headers)
                    response.raise_for_status()
                    result = response.json()
                    import time
                    st.session_state.analysis_timestamp = time.time()
                    status.update(label="Analysis complete", state="complete", expanded=False)
                except Exception as e:
                    status.update(label="Analysis failed", state="error", expanded=False)
                    st.error(f"API Request Failed: {e}")
                    return

            col1, col2 = st.columns([1, 1])

            with col1:
                st.header("Diagnostic Conclusion")
                diagnosis = result.get("diagnosis", {})
                findings = diagnosis.get("top_finding", "Unknown")
                confidence = result.get("confidence", 0.0)
                std = diagnosis.get("uncertainty_std", 0.0)

                if std < 0.1:
                    color, status_msg = "green", "HIGH CERTAINTY"
                elif std < 0.15:
                    color, status_msg = "orange", "CAUTION"
                else:
                    color, status_msg = "red", "LOW CERTAINTY / ESCALATED"

                st.markdown(
                    f"### Finding: <span style='color:{color}'>{findings}</span>",
                    unsafe_allow_html=True,
                )
                st.markdown(f"**Confidence:** {confidence:.1%} (+/- {std:.1%})")
                st.markdown(
                    f"<p style='background-color:{color}; color:white; padding:10px; "
                    "border-radius:5px; text-align:center;'><b>"
                    f"{status_msg}</b></p>",
                    unsafe_allow_html=True,
                )

                if result.get("escalation_required", False):
                    st.error("Case escalated to a radiologist due to high uncertainty.")

                with st.expander("Physician Feedback & Correction"):
                    st.write(
                        "Feedback is stored for audit and future retraining. "
                        "It does not update the model live."
                    )
                    feedback = st.radio(
                        "Verdict",
                        ["Match", "Incorrect Finding", "Unclear Evidence"],
                        key="feedback_radio",
                    )
                    feedback_notes = st.text_area(
                        "Clinical note (optional)",
                        key="feedback_notes",
                    )
                    if st.button("Submit Correction", key="feedback_submit"):
                        try:
                            api_url = os.getenv("API_URL", "http://medi-api:8000")
                            headers = {"X-API-Key": os.getenv("API_KEY", "dev-secret-key-123")}
                            
                            # Track start time for backend latency telemetry
                            start_time = st.session_state.get("analysis_timestamp", None)
                            
                            payload = {
                                "session_id": st.session_state.session_id,
                                "verdict": feedback,
                                "notes": feedback_notes,
                                "diagnosis": diagnosis,
                                "history_metadata": result.get("history_data", {}).get("metadata", {}),
                                "doctor_id": "dr-authorized-1",
                                "start_time": start_time
                            }
                            fb_resp = requests.post(f"{api_url}/feedback", json=payload, headers=headers)
                            fb_resp.raise_for_status()
                            st.success("Feedback securely logged via Authenticated API.")
                        except Exception as e:
                            st.error(f"Failed to securely transmit feedback: {e}")

            with col2:
                st.header("Visual Evidence")
                heatmap_base64 = result.get("heatmap_base64", "")
                if heatmap_base64:
                    import base64
                    try:
                        st.image(base64.b64decode(heatmap_base64), use_container_width=True, caption="BiomedCLIP Grad-CAM Attention Heatmap")
                    except Exception as e:
                        st.error(f"Failed to render heatmap: {e}")
                else:
                    st.info("No heatmap was returned by the inference API.")

            st.markdown("---")

            tab1, tab2 = st.tabs(["Cited Literature (RAG)", "Clinical Report"])

            with tab1:
                citations = result.get("pubmed_citations", [])
                if citations:
                    st.markdown("##### Clinical Literature Reference Cards")
                    for cit in citations:
                        title = cit.get('title', 'Abstract')
                        pmid = cit.get('pmid')
                        text = cit.get("text", "")
                        
                        # Limit to first 2 sentences to reduce cognitive load
                        import re
                        sentences = re.split(r'(?<=[.!?])\s+', text)
                        tldr = " ".join(sentences[:2]) if len(sentences) > 0 else text
                        
                        with st.expander(f"PMID {pmid} - {title}"):
                            st.markdown(f"**Key Findings:** *{tldr}*")
                            if len(sentences) > 2:
                                if st.checkbox("Show Full Abstract", key=f"show_abstract_{pmid}"):
                                    st.write(text)
                else:
                    st.info("No citations were returned for this case.")

            with tab2:
                diagnosis_for_export = {
                    **diagnosis,
                    "confidence": confidence,
                    "escalation_required": result.get("escalation_required", False),
                }
                report_bundle = report_gen.generate_report(
                    diagnosis_for_export,
                    result.get("history_data", {}).get("metadata", {}),
                    "", # heatmap_path is omitted until API provides it
                    citations,
                    output_filename=f"Report_{st.session_state.session_id}.pdf",
                )
                with open(report_bundle["pdf_path"], "rb") as report_handle:
                    st.download_button(
                        "Download Physician-Ready PDF Report",
                        report_handle,
                        file_name=os.path.basename(report_bundle["pdf_path"]),
                    )
                st.success("Report Generated!")
                
                # Push report to EHR via Gateway
                st.write("Transmitting to Hospital EHR...")
                try:
                    from src.data.fhir_formatter import EHRGateway
                    ehr_gateway = EHRGateway()
                    with open(report_bundle['fhir_path'], "r", encoding="utf-8") as f:
                        fhir_json = f.read()
                    success = ehr_gateway.push_report(fhir_json)
                    if success:
                        st.success("Successfully transmitted to EHR!")
                    else:
                        st.warning("EHR Transmission failed. Saved to Dead Letter Queue.")
                except Exception as e:
                    st.error(f"EHR Integration Error: {e}")
                
                with open(report_bundle["fhir_path"], "rb") as fhir_handle:
                    st.download_button(
                        "Download FHIR DiagnosticReport JSON",
                        fhir_handle,
                        file_name=os.path.basename(report_bundle["fhir_path"]),
                    )
        else:
            st.warning("Please upload both a chest X-ray and a patient history PDF.")


if __name__ == "__main__":
    main()
