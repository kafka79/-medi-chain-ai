# AI Safety & Failure Modes [Tara's Critique]

## Attention Saturation in Rare Pathologies
**The Problem:** Transformers trained on PubMed (like BiomedCLIP) can suffer from "Attention Saturation" when encountering rare conditions like Mesothelioma. The model may over-attend to common features (Pneumonia) and miss the subtle marginal thickening characteristic of asbestos-related cancer.

## The Uncertainty Circuit-Breaker
To prevent lethal misdiagnosis, we implemented **MC Dropout**.
- **First-Principles Reasoning:** We do not trust a single forward pass. By running 20 passes with stochastic dropout, we measure the **variance of the attention map**.
- **Intervention:** If `std_deviation > 0.15`, the system **force-escalates** to a human. This ensures the AI never "confidently hallunincates" a diagnosis in the presence of out-of-distribution (OOD) data.
