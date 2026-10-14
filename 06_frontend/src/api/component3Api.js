import { apiClient, API_BASE_URL } from './client';

// Utility to handle prefix whether client baseURL includes /api/v1 or root
const getEndpointPath = (path) => {
  if (apiClient.defaults.baseURL.endsWith('/api/v1')) {
    return `/component3${path}`;
  }
  return `/api/component3${path}`;
};

export const getComponent3Health = async () => {
  const res = await apiClient.get(getEndpointPath('/health'));
  return res.data;
};

export const getComponent3Summary = async () => {
  const res = await apiClient.get(getEndpointPath('/summary'));
  return res.data;
};

export const getComponent3Dashboard = async () => {
  const res = await apiClient.get(getEndpointPath('/dashboard'));
  return res.data;
};

export const getComponent3Factors = async () => {
  const res = await apiClient.get(getEndpointPath('/factors'));
  return res.data;
};

export const getComponent3Factor = async (factorId) => {
  const res = await apiClient.get(getEndpointPath(`/factors/${factorId}`));
  return res.data;
};

export const getComponent3RootCauses = async () => {
  const res = await apiClient.get(getEndpointPath('/root-causes'));
  return res.data;
};

export const getComponent3TemporalPathways = async () => {
  const res = await apiClient.get(getEndpointPath('/temporal-pathways'));
  return res.data;
};

export const getComponent3MatchedEffects = async () => {
  const res = await apiClient.get(getEndpointPath('/matched-effects'));
  return res.data;
};

export const getComponent3Cases = async () => {
  const res = await apiClient.get(getEndpointPath('/cases'));
  return res.data;
};

export const getComponent3Case = async (caseType) => {
  const res = await apiClient.get(getEndpointPath(`/cases/${caseType}`));
  return res.data;
};

export const getComponent3Figures = async () => {
  const res = await apiClient.get(getEndpointPath('/figures'));
  return res.data;
};

export const getComponent3FigureUrl = (figureKey) => {
  const path = getEndpointPath(`/figures/${figureKey}`);
  return `${API_BASE_URL}${path}`;
};

// Legacy endpoints
export const getComponent3Explain = (serial) => 
  apiClient.post(getEndpointPath('/explain'), { serial_number: serial });

export const getComponent3RootCause = (serial) => 
  apiClient.post(getEndpointPath('/root-cause'), { serial_number: serial });
