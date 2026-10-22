import { apiClient } from './client';
export const getComponent4Optimize = (driveId, state) => apiClient.post('/api/v1/component4/optimize', { drive_id: driveId, current_state: state });


/**
 * Component 4: Safety-checked HDD mitigation recommendation.
 *
 * The backend returns a DRY_RUN_ONLY decision.
 * This function does not execute any storage operation.
 */
export const getComponent4SafeDecision = async (request) => {
  const response = await apiClient.post(
    '/api/v1/component4/decide',
    request
  );

  return response.data;
};


/**
 * Component 4 - Actual trained DQN inference.
 *
 * Runs the saved DQN model through FastAPI.
 * The backend independently validates the proposed action.
 */
export const getComponent4RLDecision = async (request) => {
  const response = await apiClient.post(
    '/api/v1/component4/rl-decide',
    request
  );

  return response.data;
};
