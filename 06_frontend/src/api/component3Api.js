import { apiClient } from './client';
export const getComponent3Explain = (serial) => apiClient.post('/component3/explain', { serial_number: serial });
export const getComponent3RootCause = (serial) => apiClient.post('/component3/root-cause', { serial_number: serial });
