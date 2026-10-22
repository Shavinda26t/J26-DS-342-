import { apiClient } from './client';

// Component 2: Adaptive Digital Twin
// All endpoints serve the component's held-out validation artifacts.
// NOTE: the axios client's baseURL is the backend root, so the /api/v1
// prefix must be explicit here (same convention as component4Api).
export const getComponent2Overview = () => apiClient.get('/api/v1/component2/overview');
export const getComponent2Detections = () => apiClient.get('/api/v1/component2/detections');
export const getComponent2LeadTimeDistribution = () =>
  apiClient.get('/api/v1/component2/lead-time-distribution');
export const getComponent2Sweep = () => apiClient.get('/api/v1/component2/sweep');
export const getComponent2ConfigComparison = () =>
  apiClient.get('/api/v1/component2/config-comparison');
export const getComponent2RulMetrics = () => apiClient.get('/api/v1/component2/rul-metrics');
export const getComponent2Twin = (serial) =>
  apiClient.post('/api/v1/component2/twin', { serial_number: serial });

export const postComponent2BatchInference = (file) => {
  const formData = new FormData();
  formData.append('file', file);
  return apiClient.post('/api/v1/component2/inference/batch', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
};

// --- Live Digital Twin: synchronization + enhanced features ---
export const postComponent2TwinSync = (serialNumber, observation) =>
  apiClient.post('/api/v1/component2/twin/sync', {
    serial_number: serialNumber,
    observation,
  });

export const getComponent2TwinState = (serialNumber) =>
  apiClient.get(`/api/v1/component2/twin/${serialNumber}/state`);

export const postComponent2TwinLearn = (serialNumber, actualRulDays) =>
  apiClient.post(`/api/v1/component2/twin/${serialNumber}/learn`, {
    actual_rul_days: actualRulDays,
  });

export const deleteComponent2Twin = (serialNumber) =>
  apiClient.delete(`/api/v1/component2/twin/${serialNumber}`);

export const getComponent2TwinCalibration = () =>
  apiClient.get('/api/v1/component2/twin/calibration');

export const getComponent2TwinRiskGate = () =>
  apiClient.get('/api/v1/component2/twin/risk-gate');

export const getComponent2TwinSerials = () =>
  apiClient.get('/api/v1/component2/twin/serials');
