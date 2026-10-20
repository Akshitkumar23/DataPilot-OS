const rawApi = import.meta.env.VITE_API_URL
  || (typeof window !== 'undefined' && window.electron?.apiBaseUrl)
  || (typeof window !== 'undefined' && window.ENV_API_URL)
  || 'http://127.0.0.1:8011';
export const API = rawApi.replace(/\/+$/, '');

let sessionToken = typeof window !== 'undefined' ? (window.electron?.apiToken || window.DATAPILOT_API_TOKEN || localStorage.getItem('datapilot_api_token') || '') : '';

export function setApiToken(token) {
  if (!token) return;
  sessionToken = token;
  if (typeof window !== 'undefined') {
    window.DATAPILOT_API_TOKEN = token;
    localStorage.setItem('datapilot_api_token', token);
  }
}

export function getApiToken() {
  if (sessionToken) return sessionToken;
  if (typeof window !== 'undefined') {
    if (window.electron?.apiToken) {
      sessionToken = window.electron.apiToken;
      return sessionToken;
    }
    if (window.DATAPILOT_API_TOKEN) {
      sessionToken = window.DATAPILOT_API_TOKEN;
      return sessionToken;
    }
    const stored = localStorage.getItem('datapilot_api_token');
    if (stored) {
      sessionToken = stored;
      return sessionToken;
    }
  }
  return '';
}

export async function fetchApiToken() {
  if (typeof window !== 'undefined' && window.electron?.getApiToken) {
    try {
      const tok = await window.electron.getApiToken();
      if (tok) {
        setApiToken(tok);
        return tok;
      }
    } catch (e) {
      console.warn('Failed to fetch API token via Electron IPC:', e);
    }
  }
  return getApiToken();
}

export function getAuthHeaders(extraHeaders = {}) {
  const token = getApiToken();
  const headers = { ...extraHeaders };
  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }
  return headers;
}

export function formatApiError(errJson, fallbackStatus) {
  if (!errJson) return fallbackStatus ? `HTTP ${fallbackStatus}` : 'Unknown Server Error';
  const detail = errJson.detail || errJson.message;
  if (typeof detail === 'string') return detail;
  if (Array.isArray(detail)) {
    return detail.map(item => item.msg || item.message || JSON.stringify(item)).join('; ');
  }
  if (typeof detail === 'object' && detail !== null) {
    return detail.msg || detail.message || JSON.stringify(detail);
  }
  return fallbackStatus ? `HTTP ${fallbackStatus}` : 'Server Error';
}
