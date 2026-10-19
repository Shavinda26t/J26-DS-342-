import { apiClient } from './client';
export const getComponent2Twin = (serial) => apiClient.post('/component2/twin', { serial_number: serial });
