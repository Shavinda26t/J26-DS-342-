# AI-Based HDD Failure Prediction and Self-Healing Framework

## Project Overview
**HDD_Failure_Prediction_System** is a final-year research framework integrating four independent research components into a unified storage health monitoring and self-healing intelligence ecosystem.

### Research Components
1. **Component 1**: Behavioral Drift & GenAI Intelligence for Enterprise HDD Fleets
2. **Component 2**: Adaptive AI-Driven Digital Twin for HDD Health, Degradation, and Remaining Useful Life
3. **Component 3 (Lead Research Contribution)**: Causal Explainable AI (CXAI) for HDD Failure Analysis
4. **Component 4**: Safe Deep Reinforcement Learning for Autonomous Storage Optimization

---

## Architectural Principles
- **Strict Component Isolation**: Each component maintains its own dataset pipeline (`data/raw`, `interim`, `processed`, `features`, `splits`), models, experiments, figures, and configuration. Shared preprocessed datasets across components are prohibited.
- **Microservice Integration**: Research components remain decoupled and communicate exclusively via the **Backend API Service (`05_backend`)** and **Shared Schema Contracts (`07_shared_contracts`)**.
- **Unified Frontend**: A modern **React + Vite Frontend (`06_frontend`)** dashboard visualizes fleet health, digital twins, causal failure explanations, and safe optimization policies.

---

## Local Development Setup

### 1. Prerequisites
- Python 3.10+
- Node.js 18+
- Docker & Docker Compose (optional)

### 2. Backend Setup
```bash
cd 05_backend
python -m venv venv
# On Windows PowerShell:
.env\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### 3. Frontend Setup
```bash
cd 06_frontend
npm install
npm run dev
```

### 4. Running Docker Environment
```bash
docker-compose up --build
```

---

## Component 3: Causal XAI Workflow
Component 3 analyzes Backblaze Hard Drive Stats (Q4 2025) through:
1. Dataset Audit & Model Selection
2. SMART Attribute Selection & Temporal Feature Engineering
3. Reference Prediction Model Training
4. SHAP Explanation & Temporal Causal Discovery
5. Causal Effect Estimation & Root Cause Analysis
6. SHAP vs. Causal Validation & Robustness Evaluation

---

## Git Workflow
- Standard branching: `main`, `feature/component-3-cxai`, `feature/backend-api`, etc.
- No dataset files (`.csv`, `.parquet`) or model binaries (`.pkl`, `.joblib`) are committed to Git.
