# Q4 research decision and integration handoff

Frozen primary: `transformer_sector_error` at the 14-day horizon. Q4 scored endpoints: 1,758,728. Reserved first failures: 51; complete warning histories: 34; recalled: 11.

Primary pooled average precision: 0.02019; false alerts per 1,000 observed negative endpoints: 2.052. See `q4_14day_method_review.csv` for every model and frozen comparator.

Decision: integrate the frozen daily scorer in shadow display with the team agent. Do not show failure probabilities, enable healing, or claim drift detection. Verify the real agent schema and 30-day replay against Notebook 24 first. The Q4 findings do not change the candidate or thresholds.

Limitations: 17 recorded failures lack complete 14-day warning histories; per-model event counts are small. The ST4000DM000 synthetic sector weakness remains unresolved. No SSD or validated regime/drift result is established.
