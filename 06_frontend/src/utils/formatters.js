export const formatEnumLabel = (str) => {
  if (!str) return 'Not Available';
  
  const mapping = {
    'HIGH_PRIORITY_ROOT_CAUSE_CANDIDATE': 'High-Priority Root-Cause Candidate',
    'SUPPORTING_DEGRADATION_FACTOR': 'Supporting Degradation Factor',
    'EXPLORATORY_ROOT_CAUSE_CANDIDATE': 'Exploratory Root-Cause Candidate',
    'PREDICTIVE_TEMPORAL_CONTEXT': 'Predictive Temporal Context',
    'PREDICTIVE_SIGNAL_REQUIRES_SEMANTIC_REVIEW': 'Predictive Signal (Requires Semantic Review)',
    'PREDICTIVE_ONLY': 'Predictive Only',
    'STRONG_MATCHED_FAILURE_RISK_EVIDENCE': 'Strong Matched Failure Risk Evidence',
    'MODERATE_MATCHED_FAILURE_RISK_EVIDENCE': 'Moderate Matched Failure Risk Evidence',
    'UNCERTAIN_MATCHED_FAILURE_RISK_EVIDENCE': 'Uncertain Matched Failure Risk Evidence',
    'SPARSE_MATCHED_FAILURE_RISK_EVIDENCE': 'Sparse / Exploratory Risk Evidence',
    'TOWARD_FAILURE_OUTPUT': 'Toward Failure Prediction',
    'AWAY_FROM_FAILURE_OUTPUT': 'Away From Failure Prediction',
    'MIXED_LOCAL_EFFECT': 'Mixed Local Effect'
  };

  if (mapping[str]) return mapping[str];

  return str
    .replace(/_/g, ' ')
    .toLowerCase()
    .replace(/\b\w/g, (char) => char.toUpperCase());
};

export const formatNullable = (val, suffix = '') => {
  if (val === null || val === undefined || val === '') {
    return 'Not available in representative research export';
  }
  return `${val}${suffix}`;
};

export const formatPercentage = (val) => {
  if (val === null || val === undefined) return 'Not available in representative research export';
  return `${Number(val).toFixed(2)}%`;
};

export const formatPP = (val) => {
  if (val === null || val === undefined) return 'Not available in representative research export';
  const num = Number(val);
  const sign = num > 0 ? '+' : '';
  return `${sign}${num.toFixed(2)} pp`;
};
