const API_BASE_URL = 'http://35.184.237.121:5000';

export const API_ENDPOINTS = {
  HEALTH: `${API_BASE_URL}/api/health`,
  START_CASE: `${API_BASE_URL}/api/start-case`,
  ADD_EVENT_PREDICT: `${API_BASE_URL}/api/add-event-predict`
};

export default API_BASE_URL;