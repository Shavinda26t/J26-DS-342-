# Component 3: Causal Explainable AI for HDD Failure Analysis

## Research Contribution
Primary research contribution focusing on:
1. Global and local feature explanations (SHAP)
2. Temporal causal discovery (PCMCI / Granger Causal Graphs)
3. Causal effect estimation (DoWhy / EconML)
4. Causally informed root-cause analysis
5. Comparative analysis of predictive feature importance vs. causal evidence
6. Explanation robustness and stability evaluation

## Initial Dataset
- Backblaze Hard Drive Stats Q4 2025

## Execution Sequence
```bash
python scripts/01_dataset_audit.py
python scripts/02_select_hdd_model.py
python scripts/03_smart_attribute_audit.py
python scripts/04_data_preprocessing.py
python scripts/05_temporal_feature_engineering.py
python scripts/06_create_dataset_splits.py
python scripts/07_train_reference_model.py
python scripts/08_shap_analysis.py
python scripts/09_temporal_causal_discovery.py
python scripts/10_causal_effect_estimation.py
python scripts/11_root_cause_analysis.py
python scripts/12_shap_vs_causal_analysis.py
python scripts/13_cxai_integration.py
python scripts/14_robustness_analysis.py
python scripts/15_evaluate_cxai.py
```
