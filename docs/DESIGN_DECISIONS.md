# Design Decisions & Retrospectives

## Failed Experiment: Early Fusion Transformers [Dana's Critique]
**Duration:** 4 days in Development Phase.
**Hypothesis:** By feeding raw image patches and text tokens into a single transformer backbone (Early Fusion), the model would learn cross-modal attention directly.
**Outcome:** **FAILED.** 
**Why:** The noise in clinical PDF reports (parsing errors, irrelevant boilerplate) polluted the visual attending mechanism. The model suffered from "modality dominance" where it ignored text entirely because ImageNet-prebuilt features were stronger.
**Pivot:** Switched to **Late Fusion** with LayerNorm (v1.2). This allowed the encoders to stay specialized while the MLP learned the specific diagnostic mapping.

## Why LayerNorm over BatchNorm?
Switched during live development to handle single-patient inference. BatchNorm relies on moving averages which collapse at batch_size=1, causing diagnostic inaccuracies.
