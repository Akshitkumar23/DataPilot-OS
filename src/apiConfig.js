export const API = import.meta.env.VITE_API_URL || window.ENV_API_URL || 'http://localhost:8000';

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
