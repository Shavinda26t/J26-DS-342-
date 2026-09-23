import { apiClient } from './client';
export const getComponent4Optimize = (driveId, state) => apiClient.post('/component4/optimize', { drive_id: driveId, current_state: state });
