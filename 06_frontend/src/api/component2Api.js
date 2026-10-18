import { apiClient } from './client';

// Component 2: Adaptive Digital Twin
// All endpoints serve the component's held-out validation artifacts.
export const getComponent2Overview = () => apiClient.get('/component2/overview');
export const getComponent2Detections = () => apiClient.get('/component2/detections');
export const getComponent2LeadTimeDistribution = () =>
  apiClient.get('/component2/lead-time-distribution');
export const getComponent2Sweep = () => apiClient.get('/component2/sweep');
export const getComponent2ConfigComparison = () =>
  apiClient.get('/component2/config-comparison');
export const getComponent2RulMetrics = () => apiClient.get('/component2/rul-metrics');
export const getComponent2Twin = (serial) =>
  apiClient.post('/component2/twin', { serial_number: serial });
