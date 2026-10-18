import { apiClient } from './client';
export const getComponent1Drift = (fleetId) => apiClient.post('/api/v1/component1/analyze', { fleet_id: fleetId });
