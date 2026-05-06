# MEdi Chain AI - Enterprise Clinical Intelligence

Advanced multimodal diagnostic workflow for clinical imaging and history, built for production scale and reliability.

## Key Advancements

- **Distributed Architecture:** Moved from single-node to a multi-replica, load-balanced serving stack using Gunicorn/Uvicorn.
- **Clinical Monitoring:** Integrated Kolmogorov-Smirnov drift detection to monitor model performance over time.
- **Interoperability:** Built-in EHR Gateway supporting FHIR R4 standard for seamless integration with systems like Epic and Cerner.
- **Cloud-Native Storage:** Abstracted storage layer ready for S3/Object-store migration to support horizontal scaling.
- **Attention-Driven Fusion:** Upgraded from simple late fusion to cross-modal attention for superior feature alignment.

## System Components

- `deployment/api/main.py`: Enterprise FastAPI service with concurrency control and drift monitoring.
- `src/monitoring/drift_detector.py`: Automated statistical monitoring of model predictions.
- `src/utils/storage.py`: Pluggable storage provider for ephemeral and persistent artifacts.
- `src/data/fhir_formatter.py`: FHIR R4 DiagnosticReport generator with EHR Gateway mediator.
- `src/models/fusion.py`: Multi-head cross-attention fusion model.

## Running in Production

```bash
docker-compose -f deployment/docker-compose.yml up -d --scale medi-api=3
```

- **Health Monitoring:** Each node performs proactive health checks before accepting clinical traffic.
- **Load Balancing:** Docker Compose handles internal balancing across the 3 API replicas.

## Testing & Evaluation

```bash
# Run the new benchmark suite
python src/evaluation/benchmark.py
```

The benchmark now tracks:
- Classification Precision/Recall/F1
- P95 Inference Latency
- Calibration Error (ECE)

---

## Technical Standards

- **Safety:** MC Dropout for epistemic uncertainty; cases with high variance are automatically escalated to radiologists.
- **Privacy:** Request-scoped storage with background scavenger tasks ensures no PII leaks between sessions.
- **Scalability:** Stateless API design with shared volume mounting for local dev, ready for S3 in production.
