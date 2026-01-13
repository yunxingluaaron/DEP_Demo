const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://35.184.237.121:5000';

// Debug logging - this will show in browser console
console.log('=== API CONFIG DEBUG ===');
console.log('process.env.REACT_APP_API_URL:', process.env.REACT_APP_API_URL);
console.log('Final API_BASE_URL:', API_BASE_URL);
console.log('=======================');

export const API_ENDPOINTS = {
  HEALTH: `${API_BASE_URL}/api/health`,
  START_CASE: `${API_BASE_URL}/api/start-case`,
  ADD_EVENT_PREDICT: `${API_BASE_URL}/api/add-event-predict`
};

// Log the endpoints
console.log('API_ENDPOINTS configured:', API_ENDPOINTS);

export default API_BASE_URL;