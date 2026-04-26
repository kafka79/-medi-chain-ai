# 🏥 MEdi Chain AI: Multimodal Agentic Medical Diagnostics

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python) ![PyTorch](https://img.shields.io/badge/PyTorch-2.2-red?logo=pytorch) ![LangGraph](https://img.shields.io/badge/LangGraph-Agentic-orange) ![Milvus](https://img.shields.io/badge/VectorDB-Milvus-blue) ![FHIR](https://img.shields.io/badge/Medical-FHIR--R4-green)

> **v1.3: Edge Optimized** - Now supporting automated session cleanup and high-efficiency CPU inference via ONNX for rural clinical deployment.

---

## 🏗️ System Architecture

```mermaid
graph TD
    A[Chest X-ray / DICOM] --> B(BiomedCLIP Visual Encoder)
    C[Patient History PDF] --> D(pdfplumber Clinical Parser)
    
    B --> E{LangGraph Agent}
    D --> E
    
    E --> F[Query PubMed RAG]
    F --> G[Milvus Vector DB]
    G --> F
    
    F --> H[Late Fusion MLP]
    H --> I[MC Dropout Uncertainty]
    
    I --> J{Self-Verify Node}
    J -- Low Confidence --> F
    J -- High Confidence --> K[Physician PDF Report]
    J -- High Uncertainty --> L[Escalate to Radiologist]
    
    K --> M[DiagnosticReport FHIR-R4]
```

---

## 📊 Benchmark Results (IU-Xray Test Split)

| Metric | System Result | Baseline (Zero-shot CLIP) | Target |
|---|---|---|---|
| **AUROC (Macro)** | **0.842** | 0.612 | ≥ 0.82 |
| **Macro-F1** | **0.781** | 0.455 | — |
| **Hit-Rate@5 (RAG)** | **0.810** | — | ≥ 0.75 |
| **ECE (Calibration)** | **0.064** | 0.215 | < 0.08 |
| **VRAM Footprint** | **~4.2 GB** | — | < 16 GB |

> **Innovation:** Integrated Monte Carlo Dropout enables calibrated uncertainty quantification (ECE < 0.08), signaling "I don't know" when a case is ambiguous.

---

## 🔬 Key Features

1.  **Agentic Self-Correction**: Using LangGraph, the system recursively refines its PubMed query if initial diagnosis confidence is below 60%.
2.  **Explainable AI (XAI)**: Grad-CAM heatmap generation on the last ViT block of BiomedCLIP to audit anatomical findings.
3.  **Privacy by Design**: Training loop protected by **Differential Privacy (Opacus)** with ε=8.0, delta=1e-5.
4.  **Clinical Interoperability**: Ingests DICOM and outputs structured HL7 FHIR-R4 DiagnosticReports.
5.  **Graceful Degradation**: Triggers an escalation path when uncertainty std deviation > 0.15.

---

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Setup Docker infrastructure (Milvus, etcd, minio)
make setup
```

### 2. Edge/CPU Optimization (Sam's Fix)
For deployment in hardware-constrained rural clinics:
```bash
# Export Fusion model to ONNX for 4x faster CPU inference
python src/utils/export_onnx.py
```
*Supports OpenVINO runtime for Intel-based medical tablets.*

### 3. Run Diagnostic UI
```bash
# Launch Streamlit dashboard
streamlit run src/ui/app.py
```

### 3. API Access
```bash
# Launch FastAPI production endpoint
python deployment/api/main.py
```

---

## 🧑‍⚕️ Differential Case: Silicosis Walkthrough
For a synthetic case with "15 years silica dust exposure" and lower-lobe opacities:
- **Visual Encoder** detects reticular patterns.
- **Parser** extracts "Occupation: Stone Driller" and "Exposure: 15 years".
- **Agent** queries PubMed for "Silicosis vs Pneumonia radiographic findings".
- **Result**: `Silicosis: 72% ± 8%`, `Pneumonia: 18% ± 5%`.
- **XAI**: Grad-CAM highlights the periphery of lower lobes.

---

## 🛠️ Tech Stack
- **Deep Learning**: PyTorch, BiomedCLIP, SapBERT.
- **Agentic Logic**: LangGraph, LangChain.
- **Data Infra**: Milvus (Vector DB), Pydicom (Medical Imaging).
- **Compliance**: Opacus (Differential Privacy), fhir.resources.
- **Reporting**: ReportLab (PDF), Streamlit (UI), FastAPI (API).

---

## 🤝 Design Decisions
- **Late Fusion**: Concatenating [512D Image + 768D Text] features allows independent training objectives for vision and language encoders while retaining task-specific fine-tuning.
- **Milvus Standalone**: Chosen over serverless solutions (Pinecone) to ensure HIPAA data residency requirements can be met in local hospital infrastructure.
