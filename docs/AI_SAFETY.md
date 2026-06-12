# AI Safety & Failure Modes [Tara's Critique]

## Attention Saturation in Rare Pathologies
**The Problem:** Transformers trained on PubMed (like BiomedCLIP) can suffer from "Attention Saturation" when encountering rare conditions like Mesothelioma. The model may over-attend to common features (Pneumonia) and miss the subtle marginal thickening characteristic of asbestos-related cancer.

## The Uncertainty Circuit-Breaker
To prevent lethal misdiagnosis, we implemented **MC Dropout**.
- **First-Principles Reasoning:** We run 20 passes with stochastic dropout on the fusion head to measure the **fusion head variance**.
- **Limitation on Backbone Uncertainty:** Note that because the visual encoder (BiomedCLIP) is run once to extract static feature embeddings and its vision tower has a dropout probability of `0.0` at inference time, visual backbone epistemic uncertainty is bypassed by the MC Dropout loop.
- **Visual Safety Mitigation:** To compensate for this visual backbone limitation, the system implements a multi-layered safety strategy:
  1. **Out-of-Distribution (OOD) Softmax Check:** Any case with `max_softmax < 0.4` is rejected at the agent level as potentially out of distribution.
  2. **Uncertainty Escalation:** If the scaled standard deviation of the fusion head exceeds `0.15` (or average confidence drops below `0.6`), the system automatically **force-escalates** to a radiologist.
  3. **Visual Rationale (Heatmaps):** The visual explainer generates similarity-based heatmaps directly targeting the class logits to provide a transparent visual explanation of the specific diagnostic findings.
