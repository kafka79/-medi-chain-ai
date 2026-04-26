# Teaching & Onboarding Guide [Anika's Critique]

## Mentorship Philosophy
This project is designed with **modular isolation** to help junior engineers onboard.

### Step 1: Modality Encoders
Start the junior dev on the `VisualEncoder` and `PDFParser`. These are standard CV/NLP tasks.
### Step 2: The Agentic State
Explain LangGraph. Show how the `AgentState` dict acts as the "source of truth".
### Step 3: Deployment
Guided walkthrough of the `Dockerfile` and `Makefile` to ensure they understand reproducible environments.

**Quality Rule:** "Don't just fix a bug; document why the fix was necessary in the context of clinical safety."
