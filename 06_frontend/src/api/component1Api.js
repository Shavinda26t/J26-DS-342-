import { apiClient } from './client';
export const getComponent1Drift = (fleetId) => apiClient.post('/api/v1/component1/analyze', { fleet_id: fleetId });
export const getShadowStates = () => apiClient.get('/api/v1/component1/shadow');
export const getShadowHistory = (serialNumber) => apiClient.get('/api/v1/component1/shadow/history', { params: { serial_number: serialNumber } });
