import { apiClient } from './client';
export const getComponent1Drift = (fleetId) => apiClient.post('/component1/analyze', { fleet_id: fleetId });
